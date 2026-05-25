import os
import pandas as pd

# Path to the folder with cleaned CSV files.
FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bulk_downloads_stripped")

# Files from de_patente_1.csv through de_patente_24.csv.
csv_files = [f"de_patente_{i}.csv" for i in range(1, 25)]

# Store all DataFrames before concatenation.
all_dataframes = []

for filename in csv_files:
    file_path = os.path.join(FOLDER_PATH, filename)
    try:
        df = pd.read_csv(file_path)
        all_dataframes.append(df)
        print(f"Read {filename} with {len(df)} patents")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Concatenate all metadata.
metadata = pd.concat(all_dataframes, ignore_index=True)

# Save metadata.
metadata.to_csv("metadata.csv", index=False)
print(f"\nmetadata.csv saved with {len(metadata)} entries")
