import numpy as np
import faiss

# Load embeddings.
print("Loading embeddings.npy ...")
embeddings = np.load("embeddings.npy")

# Check shape and a sample norm before normalization.
print(f"Shape before normalization: {embeddings.shape}")
print(f"First vector norm before normalization: {np.linalg.norm(embeddings[0]):.4f}")

# Normalize with L2 norm.
faiss.normalize_L2(embeddings)

# Check normalization.
print(f"First vector norm after normalization: {np.linalg.norm(embeddings[0]):.4f}")

# Save normalized embeddings.
np.save("normalized_embeddings.npy", embeddings)
print("Normalized embeddings saved as 'normalized_embeddings.npy'")
