import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuration.
FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bulk_downloads")
FILE_PREFIX = "de_patente_"
FILE_SUFFIX = ".csv"
START_INDEX = 2
END_INDEX = 24
BATCH_SIZE = 32
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Load the model.
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Process all configured files.
for i in range(START_INDEX, END_INDEX + 1):
    filename = f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
    file_path = os.path.join(FOLDER_PATH, filename)
    output_file = f"embedding_{i:02d}.npy"

    if os.path.exists(output_file):
        print(f"Skipping {output_file} because it already exists")
        continue

    print(f"\nProcessing: {filename}")

    try:
        df = pd.read_csv(file_path, usecols=["title", "abstract"])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    # Prepare texts.
    embedding_inputs = []
    print("Filtering texts...")
    for idx in tqdm(range(len(df)), desc=f"Preparing texts from {filename}"):
        row = df.iloc[idx]
        title = str(row["title"]).strip() if pd.notna(row["title"]) else ""
        abstract = str(row["abstract"]).strip() if pd.notna(row["abstract"]) else ""

        if title and abstract:
            text = f"{title}. {abstract}"
        elif title:
            text = title
        else:
            continue

        embedding_inputs.append(text)

    print(f"Computing embeddings for {len(embedding_inputs)} texts...")
    embeddings = model.encode(
        embedding_inputs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save embeddings.
    np.save(output_file, embeddings)
    print(f"Embeddings saved as '{output_file}' with shape {embeddings.shape}")
