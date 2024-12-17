import h5py
import numpy as np
import faiss

def create_hnsw_index_streaming(file_path, index_path, batch_size=200000, ef_construction=200, M=8):
    """
    Create an HNSW index by streaming embeddings from an HDF5 file.

    Parameters:
    - file_path: Path to the HDF5 file.
    - index_path: Path to save the FAISS index.
    - batch_size: Number of embeddings to load per batch.
    - ef_construction: HNSW construction parameter for accuracy (default 200).
    - M: HNSW graph connectivity (default 32).
    """
    print(f"Creating HNSW index for embeddings in {file_path}...")

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        embedding_dataset = f['embedding']
        num_embeddings, embedding_dim = embedding_dataset.shape

        print(f"Total embeddings: {num_embeddings}, Dimension: {embedding_dim}")

        # Create the HNSW index
        index = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction

        # Stream embeddings in batches
        for start in range(0, num_embeddings, batch_size):
            end = min(start + batch_size, num_embeddings)
            embeddings_batch = embedding_dataset[start:end]
            index.add(embeddings_batch)
            print(f"Processed batch {start} to {end}")

        # Save the index
        faiss.write_index(index, index_path)
        print(f"Index saved to {index_path}")

# File paths
file_path = "embeddings.h5"
index_path = "hnsw_index.bin"

# Create HNSW index
create_hnsw_index_streaming(file_path, index_path)
