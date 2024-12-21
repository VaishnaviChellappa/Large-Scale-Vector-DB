import boto3
import ir_datasets
import json
from io import BytesIO

# AWS S3 configuration
BUCKET_NAME = "msmarcobucket"  #  S3 bucket name
S3_PREFIX = "msmarco-passages/"  # Folder name in the bucket

# Initialize the S3 client
s3 = boto3.client("s3")

# Load the MS MARCO Passage v2 dataset
dataset = ir_datasets.load("msmarco-passage-v2")

def upload_passage_to_s3(doc):
    """
    Upload a single document (passage) to S3 as a JSON file.
    :param doc: The namedtuple containing doc_id and text.
    """
    # Prepare the JSON content
    content = {
        "doc_id": doc.doc_id,
        "text": doc.text,
        "spans": doc.spans,
        "msmarco_document_id": doc.msmarco_document_id
    }
    
    # Convert JSON to string and then to bytes
    json_content = json.dumps(content)
    file_obj = BytesIO(json_content.encode("utf-8"))
    
    # Define the S3 object key (file name)
    s3_key = f"{S3_PREFIX}{doc.doc_id}.json"
    
    # Upload to S3
    s3.upload_fileobj(file_obj, BUCKET_NAME, s3_key)
    print(f"Uploaded {doc.doc_id} to s3://{BUCKET_NAME}/{s3_key}")

# Stream and upload documents to S3
for doc in dataset.docs_iter():
    upload_passage_to_s3(doc)

print("All passages uploaded successfully.")
