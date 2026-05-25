import os
import numpy as np

# Path to the embedding folder.
FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Embeddings")

# Files in sorted order from embedding_01.npy through embedding_24.npy.
embedding_files = [f"embedding_{i:02d}.npy" for i in range(1, 25)]

all_embeddings = []

for filename in embedding_files:
    file_path = os.path.join(FOLDER_PATH, filename)
    if not os.path.exists(file_path):
        print(f"Missing file: {filename}")
        continue
    print(f"Loading: {filename}")
    embeddings = np.load(file_path)
    all_embeddings.append(embeddings)

# Concatenate all arrays.
merged_embeddings = np.concatenate(all_embeddings, axis=0)

# Save merged embeddings.
np.save("embeddings.npy", merged_embeddings)
print(f"\nMerged file 'embeddings.npy' saved with shape {merged_embeddings.shape}")
