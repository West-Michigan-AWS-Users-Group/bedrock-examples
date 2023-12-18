import os
import sys

from flask import Flask, jsonify, render_template, request

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/bedrock_text", methods=["POST"])
def bedrock_text():
    try:
        input_text = request.form["input_text"]
        # Call your Python function to process the input_text
        result = bedrock.bedrock_qa(input_text)
        return result
    except Exception as e:
        return jsonify({"error": str(e)})


def process_input_text(input_text):
    # Your Python logic here to process the input_text
    # Replace this with your actual processing code
    return f"Processed: {input_text}"


if __name__ == "__main__":
    app.run(debug=True)
