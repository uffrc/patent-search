from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configuration.
embedding_file = "normalized_embeddings.npy"
index_file = "faiss_index.index"
top_k = 5

# Load embeddings to check their dimension.
print(f"Loading sample embeddings from '{embedding_file}'...")
embeddings = np.load(embedding_file).astype("float32")
dimension = embeddings.shape[1]
print(f"Stored embeddings: {len(embeddings)}, dimension: {dimension}")

# Load the FAISS index.
print(f"Loading FAISS index from '{index_file}'...")
index = faiss.read_index(index_file)
print(f"Index contains {index.ntotal} vectors.")

# Load the model.
print("Loading SentenceTransformer model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Example patent text.
example_text = "a bicycle light powered by a wheel-mounted dynamo"
print(f"\nSearch text: '{example_text}'")

# Convert text to a normalized vector.
query_vector = model.encode([example_text], normalize_embeddings=True).astype("float32")

# Run the FAISS search.
print(f"\nSearching for the top {top_k} most similar embeddings...")
scores, indices = index.search(query_vector, top_k)

# Print result indices.
print("\nTop results by FAISS position:")
for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
    print(f"{rank}. Position: {idx}, Score: {score:.4f}")
