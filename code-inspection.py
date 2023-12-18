from utils import bedrock

file_path = "/Users/trentnielsen/dev/yam-technology/cf-modules/cf_ec2.py"
ec2_module_context = """
This file contains code that is not dry and a lot of code is repeated in nested loops. Detailed information on how
to fix these nested loops, making the code DRY should be a primary focus. 
"""

# print("######### Starting code snipped evaluation #########")
# bedrock.bedrock_code_evaluation(file_path, additional_context=ec2_module_context)
bedrock.bedrock_code_evaluation(
    file_path,
    additional_context="""'
You will explain the below and highlight if there are any red flags or where best practices are not being followed.
Please provide detailed examples of how to fix any issues, including the line numbers on which the problems exist.
""",
)
# print("######### Ending code snipped evaluation #########")

# print("######### Starting code summary evaluation #########")
# bedrock.bedrock_code_summary(file_path)
bedrock.bedrock_code_evaluation(
    file_path,
    additional_context="""'
Explain to me what this code is doing in 10 sentences or less, using laymans terms.
""",
)
# print("######### Ending code summary evaluation #########")
