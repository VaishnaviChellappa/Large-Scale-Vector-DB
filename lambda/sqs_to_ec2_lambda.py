import json
import os
import requests

#  public DNS or IP of the EC2 instance where the Flask server is running
EC2_ENDPOINT = "http://ec2-endpoint:5000/search"

def lambda_handler(event, context):
    # Parse the incoming event, which should contain a body with { "query": "<the query>" }
    # If using API Gateway with a Lambda proxy integration, the request payload will be in event['body']
    if 'body' not in event or event['body'] is None:
        return {
            "statusCode": 400,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps({"error": "No request body provided."})
        }

    # parse the request body as JSON
    try:
        data = json.loads(event['body'])
    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps({"error": "Invalid JSON in request body."})
        }

    # Extract the query parameter
    query = data.get('query')
    if not query:
        return {
            "statusCode": 400,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps({"error": "No 'query' field found in request."})
        }

    # Make a POST request to the EC2-based Flask service
    try:
        response = requests.post(EC2_ENDPOINT, json={"query": query})
        response.raise_for_status()  # Raises an HTTPError if response is not 2xx

        # The backend should return a JSON array of passages
        passages = response.json()

        # Return the results back to the caller
        return {
            "statusCode": 200,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps(passages)
        }

    except requests.exceptions.RequestException as e:
        # If there's an error calling the backend, return a dummy data response similar to the front end fallback
        dummy_data = [
            { "id": "0", "passage": "This is a dummy passage. Backend is unreachable." }
        ]
        return {
            "statusCode": 200,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps(dummy_data)
        }
