import numpy as np
import faiss
import os

# Configuration.
embedding_file = "normalized_embeddings.npy"
index_file = "faiss_index.index"

# Load embeddings.
print(f"Loading embeddings from '{embedding_file}'...")
embeddings = np.load(embedding_file).astype("float32")

# Build the FAISS index.
dimension = embeddings.shape[1]
print(f"Creating FAISS index with dimension {dimension} and {len(embeddings)} vectors...")

# Use Flat Inner Product for cosine similarity because vectors are normalized.
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("Index created successfully.")

# Save the index.
faiss.write_index(index, index_file)
print(f"Index saved to '{index_file}'.")

# Check that the file exists.
if os.path.exists(index_file):
    size = os.path.getsize(index_file) / (1024 * 1024)
    print(f"File size: {size:.2f} MB")
else:
    print("Error: file was not found.")
