# German Patent Semantic Search

Local semantic search over roughly 2.2 million German patent records. A
multilingual Sentence Transformer converts an invention description into an
embedding, and FAISS retrieves semantically similar patents. The project
provides a terminal interface and a Streamlit interface.

## Setup

Create the Conda environment:

```sh
conda env create -f environment.yml
conda activate patent-search
```

The local dataset was downloaded from Google's patent database and filtered to
German records. It is not a complete patent collection: the search corpus
contains titles and abstracts rather than full patent documents, and some
abstracts are missing. The data and generated search assets are intentionally
not included in this repository.

The search expects 24 CSV files named `de_patente_1.csv` through
`de_patente_24.csv` in `bulk_downloads_stripped/`. Each file must contain
`publication_number`, `publication_date`, `title`, and `abstract` columns. A
compatible `faiss_index.index` must be stored in the project directory. Users
who recreate the dataset are responsible for complying with Google's
applicable terms.

The embedding model is
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
published under the Apache-2.0 model license.

## Run

For the first terminal run, allow the embedding model to be downloaded:

```sh
python search.py --online-model
```

Later terminal runs can use `python search.py`. The terminal displays ten
results by default; this is configured by `DEFAULT_TOP_K` in `search.py`.

After the model is available locally, start the web interface with:

```sh
streamlit run streamlit_app.py
```

The FAISS index and transformer model are loaded into memory. At least 8 GB of
free RAM is required, and 16 GB or more is recommended.
