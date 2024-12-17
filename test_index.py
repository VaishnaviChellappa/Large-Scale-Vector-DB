import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
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

# File paths
queries_path = "queries.tsv"
collection_path = "collection.tsv"
index_path = "hnsw_index.bin"

# Load queries and collection
print("Loading queries and collection...")
queries_df = load_tsv(queries_path)
collection_df = load_tsv(collection_path)

# Select 5 random queries for testing
selected_queries = queries_df.sample(2, random_state=42)
print(f"Selected Queries:\n{selected_queries}")

# Encode selected queries using SentenceTransformer
print("Encoding queries...")
model = SentenceTransformer("msmarco-MiniLM-L6-cos-v5")
query_embeddings = encode_texts(selected_queries["text"].tolist(), model)

# Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index(index_path)

# Query the index
print("Querying the index...")
top_k = 5  # Retrieve top 5 results per query
distances, indices = query_index(index, query_embeddings, top_k)

# Map retrieved indices to passage IDs and texts
print("Mapping results to passages...")
results = []
for query_idx, retrieved_indices in enumerate(indices):
    retrieved_passages = collection_df.iloc[retrieved_indices]
    results.append({
        "query_id": selected_queries["id"].iloc[query_idx],
        "query_text": selected_queries["text"].iloc[query_idx],
        "retrieved_ids": retrieved_passages["id"].tolist(),
        "retrieved_texts": retrieved_passages["text"].tolist(),
        "distances": distances[query_idx].tolist()
    })

# Display results
print("\nQuery Results:")
for result in results:
    print(f"Query ID: {result['query_id']}")
    print(f"Query Text: {result['query_text']}")
    print("Top Retrieved Passages:")
    for i, passage_id in enumerate(result['retrieved_ids']):
        print(f"  - Passage ID: {passage_id}, Text: {result['retrieved_texts'][i]}, Distance: {result['distances'][i]}")
    print("-" * 50)
