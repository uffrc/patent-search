#!/usr/bin/env python3
import argparse
import bisect
import csv
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 10

# FAISS and PyTorch can trip over each other through OpenMP on macOS when the
# large index and the transformer are loaded in the same process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@dataclass(frozen=True)
class PatentSource:
    csv_path: Path
    start_index: int
    row_count: int

    @property
    def end_index(self):
        return self.start_index + self.row_count


def load_required_modules():
    missing = []

    try:
        import faiss
    except ModuleNotFoundError:
        faiss = None
        missing.append("faiss-cpu")

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
        missing.append("numpy")

    try:
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging
    except ModuleNotFoundError:
        SentenceTransformer = None
        missing.append("sentence-transformers")

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing Python package(s): "
            + ", ".join(missing)
            + "\nInstall them with:\n\n"
            + f"    python -m pip install {packages}\n"
        )

    faiss.omp_set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    transformers_logging.disable_progress_bar()

    return faiss, SentenceTransformer


def npy_row_count(path):
    import numpy as np

    return int(np.load(path, mmap_mode="r").shape[0])


def csv_row_count(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def discover_sources(base_dir):
    csv_dir = base_dir / "bulk_downloads_stripped"
    embeddings_dir = base_dir / "Embeddings"

    if not csv_dir.exists():
        raise FileNotFoundError(f"Could not find CSV folder: {csv_dir}")

    sources = []
    start_index = 0

    for number in range(1, 25):
        csv_path = csv_dir / f"de_patente_{number}.csv"
        embedding_path = embeddings_dir / f"embedding_{number:02d}.npy"

        if not csv_path.exists():
            continue

        if embedding_path.exists():
            row_count = npy_row_count(embedding_path)
        else:
            row_count = csv_row_count(csv_path)

        sources.append(
            PatentSource(
                csv_path=csv_path,
                start_index=start_index,
                row_count=row_count,
            )
        )
        start_index += row_count

    if not sources:
        raise FileNotFoundError(f"No de_patente_*.csv files found in {csv_dir}")

    return sources


def source_for_index(sources, global_index):
    ends = [source.end_index for source in sources]
    source_position = bisect.bisect_right(ends, global_index)
    if source_position >= len(sources):
        raise IndexError(f"Result index {global_index} is outside the patent data")

    source = sources[source_position]
    return source, global_index - source.start_index


def read_patents(sources, global_indices):
    wanted_by_file = defaultdict(set)
    local_to_global = {}

    for global_index in global_indices:
        source, local_index = source_for_index(sources, int(global_index))
        wanted_by_file[source.csv_path].add(local_index)
        local_to_global[(source.csv_path, local_index)] = int(global_index)

    patents = {}

    for csv_path, wanted_rows in wanted_by_file.items():
        remaining = set(wanted_rows)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for local_index, row in enumerate(reader):
                if local_index not in remaining:
                    continue

                global_index = local_to_global[(csv_path, local_index)]
                patents[global_index] = row
                remaining.remove(local_index)

                if not remaining:
                    break

        if remaining:
            missing = ", ".join(str(row_number) for row_number in sorted(remaining))
            raise IndexError(f"Could not find row(s) {missing} in {csv_path}")

    return patents


def shorten(text, width=380):
    text = " ".join(str(text or "").split())
    if not text:
        return "not available"
    return textwrap.shorten(text, width=width, placeholder="...")


def print_results(indices, scores, patents):
    print("\nTop matches:\n")

    for rank, (global_index, score) in enumerate(zip(indices, scores), start=1):
        patent = patents.get(int(global_index), {})
        publication_number = patent.get("publication_number") or "unknown publication"
        publication_date = patent.get("publication_date") or "unknown date"
        title = patent.get("title") or "-"
        abstract = patent.get("abstract") or ""

        print(f"{rank}. Publication {publication_number}, score {score:.4f}")
        print(f"   Date: {publication_date}")
        print(f"   Title: {title}")
        print(f"   Abstract: {shorten(abstract)}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search similar German patents from a terminal prompt."
    )
    parser.add_argument(
        "--index",
        default="faiss_index.index",
        help="path to the FAISS index file",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="sentence-transformers model used for the query embedding",
    )
    parser.add_argument(
        "--online-model",
        action="store_true",
        help="allow downloading/checking model files from Hugging Face",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    index_path = base_dir / args.index

    if not index_path.exists():
        raise SystemExit(f"Could not find FAISS index: {index_path}")

    print("Loading search data...")
    sources = discover_sources(base_dir)

    faiss, SentenceTransformer = load_required_modules()

    print(f"Loading FAISS index from {index_path.name}...")
    index = faiss.read_index(str(index_path))

    source_row_count = sum(source.row_count for source in sources)
    if index.ntotal != source_row_count:
        print(
            "Warning: FAISS index contains "
            f"{index.ntotal} vectors, but the CSV/embedding sources contain "
            f"{source_row_count} rows.",
            file=sys.stderr,
        )

    print(f"Loading transformer model: {args.model}")
    model = SentenceTransformer(args.model, local_files_only=not args.online_model)

    print("\ntype your idea and hit enter to search")
    print("Press Enter on an empty line, or type q, to quit.\n")

    while True:
        try:
            query = input("Idea: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() in {"q", "quit", "exit"}:
            break

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        if query_vector.shape[1] != index.d:
            raise SystemExit(
                f"Model produced {query_vector.shape[1]} dimensions, "
                f"but the FAISS index expects {index.d}."
            )

        top_k = min(DEFAULT_TOP_K, index.ntotal)
        scores, indices = index.search(query_vector, top_k)
        result_indices = [int(index_number) for index_number in indices[0]]
        patents = read_patents(sources, result_indices)
        print_results(result_indices, scores[0], patents)

    print("Goodbye.")


if __name__ == "__main__":
    main()
