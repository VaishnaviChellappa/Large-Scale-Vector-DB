from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
import numpy as np
import h5py
import s3fs
import io

# Initialize Spark
spark = SparkSession.builder.appName("MSMARCO_Embeddings").getOrCreate()

# S3 Paths
s3_input_path = "s3a://msmarcobucket/collection.tsv"
s3_output_path = "s3://msmarcobucket/embeddings/collection_embeddings.hdf5"

# read the MS MARCO collection from S3
# collection.tsv is typically: id \t text
df = spark.read.csv(s3_input_path, sep="\t", header=False).toDF("id", "text")

# Broadcast model name so that all executors know what to load
model_name = "msmarco-MiniLM-L6-cos-v5"
bc_model_name = spark.sparkContext.broadcast(model_name)

def embed_partitions(iter_records):
    # Load the model once per partition
    model = SentenceTransformer(bc_model_name.value)
    
    # Collect texts
    records = list(iter_records)
    if len(records) == 0:
        return []
    
    ids = [r[0] for r in records]
    texts = [r[1] for r in records]
    
    # Encode all texts in the partition
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Return (id, embedding) pairs
    # Convert embedding to a list so it can be serialized easily
    return zip(ids, embeddings.tolist())

# Apply the mapPartitions function
id_embeddings_rdd = df.rdd.mapPartitions(embed_partitions)

all_data = id_embeddings_rdd.collect()

# all_data is now a list of (id, embedding) pairs on the driver.
all_data.sort(key=lambda x: x[0])

ids = [x[0] for x in all_data]
embeddings = np.array([x[1] for x in all_data], dtype='float32')

# Write to HDF5 on S3
# use s3fs + h5py to write directly to S3
fs = s3fs.S3FileSystem()

# Open an S3 file for writing
with fs.open(s3_output_path, 'wb') as f:
    with h5py.File(f, 'w') as hf:
        # Create a dataset for ids and embeddings
        # Store ids as strings (variable length)
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('ids', data=np.array(ids, dtype=object), dtype=dt)
        hf.create_dataset('embeddings', data=embeddings)

spark.stop()
