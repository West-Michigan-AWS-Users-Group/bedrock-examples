import os
import sys
import time
from urllib.request import urlretrieve

import numpy as np
# import pinecone
import psycopg2
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from utils import bedrock, print_ww


# docker run --name postgres -e POSTGRES_PASSWORD=bedrockworkshop! -p 5432:5432 -v ./init.sql:/docker-entrypoint-initdb.d/init.sql -d ankane/pgvector

def bedrock_connection():
    return bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", 'us-west-2')
    )


def init_bedrock_claude_client():
    return Bedrock(
        model_id="anthropic.claude-v1", client=bedrock_connection(), model_kwargs={"max_tokens_to_sample": 200}
    )


pg_params = {
    "host": "127.0.0.1",
    "database": os.environ.get("PGVECTOR_DATABASE", "postgres"),
    "user": os.environ.get("PGVECTOR_USER", "postgres"),
    "password": os.environ.get("PGVECTOR_PASSWORD", "bedrockworkshop!")
}


def create_vector_datastore_connection_string(host, database, user, password):
    DRIVER = os.environ.get("PGVECTOR_DRIVER", "psycopg2", )
    HOST = os.environ.get("PGVECTOR_HOST", host)
    PORT = os.environ.get("PGVECTOR_PORT", "5432")
    DATABASE = os.environ.get("PGVECTOR_DATABASE", database)
    USER = os.environ.get("PGVECTOR_USER", user)
    PASSWORD = os.environ.get("PGVECTOR_PASSWORD", password)

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=DRIVER,
        host=HOST,
        port=PORT,
        database=DATABASE,
        user=USER,
        password=PASSWORD
    )
    return CONNECTION_STRING


def fetch_irs_documents():
    os.makedirs("data", exist_ok=True)
    files = [
        "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
        "https://www.irs.gov/pub/irs-pdf/p15.pdf",
        "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
    ]
    for url in files:
        file_path = os.path.join("data", url.rpartition("/")[2])
        urlretrieve(url, file_path)


def load_vector_db(database, user, host, password, connection_string):
    module_path = ".."
    sys.path.append(os.path.abspath(module_path))
    claud_llm = init_bedrock_claude_client()
    bedrock_emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_connection())

    fetch_irs_documents()

    loader = PyPDFDirectoryLoader("./data/")

    documents = loader.load()
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a tiny chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=100,
    )
    docs = text_splitter.split_documents(documents)

    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(
        documents
    )
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(docs)
    print(f"Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.")
    print(f"After the split we have {len(docs)} documents more than the original {len(documents)}.")
    print(
        f"Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters."
    )

    sample_embedding = np.array(bedrock_emb.embed_query(docs[0].page_content))
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)

    conn = psycopg2.connect(dbname=database, user=user, host=host, password=password)
    cur = conn.cursor()
    conn.commit()

    os_var = '05a2cea7-667f-4932-bc34-b6f98b8c22f0'

#     pinecone.init(api_key=os_var, environment='gcp-starter')
# 
#     index_name = 'vectorstore-embeddings'
# 
#     try:
#         index = pinecone.Index(index_name)
#         pinecone.describe_index(index_name)
#     except TypeError:
#         print("Index does not exist yet, creating it now")
#         pinecone.create_index(index_name, metric='cosine', dimension=768)
#         # wait for index to be ready before connecting
#         while not pinecone.describe_index(index_name).status['ready']:
#             print(pinecone.describe_index(index_name))
#             time.sleep(1)

    db = PGVector.from_documents(
        embedding=bedrock_emb,
        documents=docs,
        collection_name='tax_info',
        connection_string=connection_string
    )
    return bedrock_emb, claud_llm, docs


# bedrock_embeddings, qa_llm, qa_docs = load_vector_db(**pg_params,
#                                                      connection_string=create_vector_datastore_connection_string(
#                                                          **pg_params)
#                                                      )

## Create a PGVector Store. Helpful if we didn't create the store in this session
tax_vectorstore = PGVector(
    collection_name="tax_info",
    connection_string=create_vector_datastore_connection_string(**pg_params),
    embedding_function=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_connection()),
)

claude_llm = init_bedrock_claude_client()

# query = "What website can I visit for help with my taxes?"
# query = "When does a business have to report tips and what are the guidelines?"
# query = "Am I required to have an EIN to file taxes?"
# query = "What is the difference between a sole proprietorship and a partnership?"
query = "Is it possible that I get sentenced to jail due to failure in filings?"
similar_documents = tax_vectorstore.similarity_search(query, k=3)  # our search query  # return 3 most relevant docs
qa = RetrievalQA.from_chain_type(llm=claude_llm,
                                 chain_type="stuff",
                                 retriever=tax_vectorstore.as_retriever(),
                                 )

qa_result = qa(query)
print_ww(f'qa query standalone result: {qa_result["result"]}')

print_ww('***************************************************************************************')
print_ww('***************************************************************************************')

# qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
#    llm=claude_llm, chain_type="stuff", retriever=tax_vectorstore.as_retriever()
# )

# qa_with_sources.run(query)
#


qa_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=claude_llm,
    chain_type="stuff",
    retriever=tax_vectorstore.as_retriever(),
    return_source_documents=True,
    # chain_type_kwargs={"prompt": PROMPT},
)

qa_source_result = qa_sources(query)
print_ww(f'qa with sources answer: {qa_source_result["answer"]}')
try:
    print_ww(f'qa with sources sources: {qa_source_result["source_documents"]}')
except KeyError:
    pass
# uncomment this and the above argument to return_source_documents to see the source documents
# print_ww(f'qa query source result: {qa_source_result["source_documents"]}')

# result = qa({"query with sources": query})
# print_ww(result["result"])


# print_ww('***************************************************************************************')
# print_ww('***************************************************************************************')
# prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
# 
# {context}
# 
# Question: {question}
# Assistant:"""
# 
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", query])
# 
# qa_source_custom = RetrievalQA.from_chain_type(
#     llm=claude_llm,
#     chain_type="stuff",
#     retriever=tax_vectorstore.as_retriever(),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT},
# )
# qa_source_custom_result = qa_source_custom({"query": query})
# print_ww(f'qa query answer custom result: {qa_source_custom_result["result"]}')
# print(qa_source_custom_result["source_documents"])