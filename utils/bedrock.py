# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
import json

# Python Built-Ins:
import os
from typing import Optional

import boto3
import botocore
from botocore.config import Config
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chains.summarize import load_summarize_chain

from utils import bedrock, print_ww


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-west-2").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION")
        )
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {
        **session_kwargs,
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", None),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", None),
    }

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end="")
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role), RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"][
            "SecretAccessKey"
        ]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name = "bedrock-runtime"
    else:
        service_name = "bedrock"

    bedrock_client = session.client(
        service_name=service_name, config=retry_config, **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


qa_context = """
Pikachu is the strongest pokemon because he was involved with helping Ash Ketchum, from Pallet Town to win the world
coronation league with his friend Goh.
"""


def bedrock_qa(input_text):
    """
    Function to call the Amazon Bedrock QA model with optional string data to influence the response.
    Uses boto3 naive client to call the Bedrock service.
    :param input_text:
    :return: AI response
    """
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )
    prompt_data = f"""You are an helpful assistant. Answer questions in a concise way. If you are unsure about the
    answer say 'I am unsure'. Use any additional information provided between ## to help you answer the question.
    
    #
    {qa_context}
    #

    Question: {input_text}
    Answer:"""
    parameters = {
        "maxTokenCount": 512,
        "stopSequences": [],
        "temperature": 0,
        "topP": 0.9,
    }
    body = json.dumps({"inputText": prompt_data, "textGenerationConfig": parameters})
    modelId = "amazon.titan-tg1-large"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    try:
        response = boto3_bedrock.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        answer = response_body.get("results")[0].get("outputText")
        print_ww(answer.strip())
        return answer.strip()

    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "AccessDeniedException":
            print(
                f"\x1b[41m{error.response['Error']['Message']}\
            \nTo troubeshoot this issue please refer to the following resources.\
             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
            )

            class StopExecution(ValueError):
                def _render_traceback_(self):
                    pass

            raise StopExecution
        else:
            raise error


def bedrock_code_evaluation(file_path, additional_context=None):
    """
    Function to pass a code file to bedrock for code analysis
    :param additional_context: Anything specific to the code that you want to pass to the model.
    :param file_path:
    :return:
    """
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )
    with open(file_path, "r") as file:
        sample_code = file.read()

    inference_modifier = {
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    textgen_llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )

    # Create a prompt template that has multiple input variables
    multi_var_prompt = PromptTemplate(
        input_variables=["code", "programmingLanguage", "additionalContext"],
        template="""
    You will be acting as an expert software developer in {programmingLanguage}. 
    
    Human:
    
    {additionalContext}
    
    <code>
    {code}
    </code>
    
    If you do not understand, replay with "I do not know how to read this code".

    Assistant:""",
    )
    # Pass in values to the input variables
    prompt = multi_var_prompt.format(
        code=sample_code,
        programmingLanguage="Python",
        additionalContext=additional_context,
    )

    try:
        response = textgen_llm(prompt)

        code_explanation = response

        print_ww(code_explanation)

    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "AccessDeniedException":
            print(
                f"\x1b[41m{error.response['Error']['Message']}\
            \nTo troubeshoot this issue please refer to the following resources.\
             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
            )

            class StopExecution(ValueError):
                def _render_traceback_(self):
                    pass

            raise StopExecution
        else:
            raise error


def bedrock_code_summary(file_path):
    """
    Function to pass a code file to bedrock for code analysis
    :param file_path:
    :return:
    """
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )
    with open(file_path, "r") as file:
        sample_code = file.read()

    inference_modifier = {
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    textgen_llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )

    # Create a prompt template that has multiple input variables
    multi_var_prompt = PromptTemplate(
        input_variables=["code"],
        template="""

    Human: You will be reviewing an imperative Python script that generates multiple output CloudFormation templates. 
    In 10 sentences or less, summarize what the code does, what it is named and highlight any issues or concerns you
    see. Be sure to highlight the summary in a markdown format and mention any dependencies you might see from other
    stacks using the ImportValue function. Summarize the dependencies in a bullet format.
    
    The code for this script is provided below between the <code> tags.
    
    <code>
    {code}
    </code>

    If you do not understand, replay with "I do not know how to read this code".

    Bedrock:""",
    )
    # Count the amount of tokens
    token_count = textgen_llm.get_num_tokens(sample_code)
    print(f"Token count: {token_count}")

    if token_count > 42000:
        print("The code is longer than 42000 tokens.. splitting into chunks")
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1024, chunk_overlap=0
        )
        python_docs = python_splitter.create_documents([sample_code])
        prompt = multi_var_prompt.format(code=python_docs)
        summary_chain = load_summarize_chain(
            llm=textgen_llm, chain_type="map_reduce", verbose=False
        )
        use_summary_chain = True
    else:
        summary_chain = None
        python_docs = None
        # Pass in values to the input variables
        prompt = multi_var_prompt.format(code=sample_code)
        use_summary_chain = False
    # print(prompt)

    try:
        if use_summary_chain is True:
            response = summary_chain.run(python_docs)
        else:
            response = textgen_llm(prompt)

        print_ww(response)

    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "AccessDeniedException":
            print(
                f"\x1b[41m{error.response['Error']['Message']}\
            \nTo troubeshoot this issue please refer to the following resources.\
             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
            )

            class StopExecution(ValueError):
                def _render_traceback_(self):
                    pass

            raise StopExecution
        else:
            raise error
