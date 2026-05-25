import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Input text.
query_text = (
    "A drone that automatically follows a vehicle using AI-based image "
    "recognition for security applications. The drone has a camera module "
    "and uses image recognition to detect and track the vehicle."
)

# Load the embedding model.
print("Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Embed and normalize the input text.
query_embedding = model.encode([query_text], convert_to_numpy=True)
faiss.normalize_L2(query_embedding)

# Load the already normalized vector matrix.
print("Loading vector matrix from 'normalized_embeddings.npy' ...")
embeddings = np.load("normalized_embeddings.npy")

# Prepare the FAISS index.
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Search for the top 10 similar patents.
print("Searching for the 10 most similar patents...")
scores, indices = index.search(query_embedding, k=10)

# Load metadata.
metadata = pd.read_csv("metadata.csv")

# Print results.
print("\nThese 10 patents are closest to the patent idea:\n")

for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
    entry = metadata.iloc[idx]
    print(f"{rank}. Publication {entry['publication_number']}, score {score:.4f}")
    print(f"    Title: {entry['title']}")
    print(f"    Date: {entry.get('publication_date', 'n/a')}")
    abstract = entry.get("abstract", "")
    if isinstance(abstract, str):
        print(f"    Abstract: {abstract[:300]}...\n")
    else:
        print("    Abstract: not available\n")
