from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration.
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "bulk_downloads"
OUTPUT_DIR = BASE_DIR / "Embeddings"
FILE_PREFIX = "de_patente_"
FILE_SUFFIX = ".csv"
START_INDEX = 1
END_INDEX = 24
BATCH_SIZE = 32
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the model.
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Process all configured files.
for i in range(START_INDEX, END_INDEX + 1):
    filename = f"{FILE_PREFIX}{i}{FILE_SUFFIX}"
    file_path = SOURCE_DIR / filename
    output_file = OUTPUT_DIR / f"embedding_{i:02d}.npy"

    if output_file.exists():
        print(f"Skipping {output_file.name} because it already exists")
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
    print(f"Embeddings saved as '{output_file.name}' with shape {embeddings.shape}")
