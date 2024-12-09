import numpy as np
import pandas as pd
import faiss
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load the MSMARCO MiniLM model
model_name = "msmarco-MiniLM-L6-cos-v5"
model = SentenceTransformer(model_name)

# Index
index_path = "/Users/sahilfaizal/Desktop/BigData/project/backend/hnsw_index.bin"
index = faiss.read_index(index_path)

def load_tsv(file_path):
    """
    Load a TSV file into a Pandas DataFrame.

    Parameters:
    - file_path: Path to the TSV file.

    Returns:
    - DataFrame with TSV contents.
    """
    return pd.read_csv(file_path, sep="\t", header=None, names=["id", "text"])

def encode_texts(texts, model):
    """
    Encode a list of texts using a SentenceTransformer model.

    Parameters:
    - texts: List of strings to encode.
    - model: Preloaded SentenceTransformer model.

    Returns:
    - Numpy array of embeddings.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def query_index(index, query_embeddings, top_k=10):
    """
    Query the FAISS index with query embeddings.

    Parameters:
    - index: Loaded FAISS index.
    - query_embeddings: Query embeddings (NumPy array).
    - top_k: Number of nearest neighbors to retrieve.

    Returns:
    - distances: Distance scores of the nearest neighbors.
    - indices: Indices of the nearest neighbors in the index.
    """
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices

# Collection
collection_path = "/Users/sahilfaizal/Desktop/BigData/project/backend/collection.tsv"
collection_df = load_tsv(collection_path)

@app.route('/search', methods=['POST'])
def retrieve_items():
    input = request.get_json()
    response_data = []
    # Generate embeddings
    query_embeddings = encode_texts([input['query']], model)
    query_embeddings = query_embeddings.reshape(1, -1) if query_embeddings.ndim == 1 else query_embeddings
    # Query the index
    top_k = 3  # Retrieve top 3 results per query
    distances, indices = query_index(index, query_embeddings, top_k)
    # Map retrieved indices to passage IDs and texts
    print("Mapping results to passages...")
    results = []
    for query_idx, retrieved_indices in enumerate(indices):
        # retrieved_passages = collection_df.iloc[retrieved_indices]
        retrieved_passages = collection_df.iloc[retrieved_indices.tolist()]
        results.append({
        "query_text": input['query'],
        "retrieved_ids": retrieved_passages["id"].tolist(),
        "retrieved_texts": retrieved_passages["text"].tolist(),
        "distances": distances[query_idx].tolist()
    })
    for result in results:
        for i, passage_id in enumerate(result['retrieved_ids']):
            response_data.append({"id": passage_id, "passage": result['retrieved_texts'][i]})
    return jsonify(response_data), 200

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
