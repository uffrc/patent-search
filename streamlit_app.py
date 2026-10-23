#!/usr/bin/env python3
import logging
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

from search import (
    MODEL_NAME,
    discover_sources,
    load_required_modules,
    search_patents,
    shorten,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_COUNT = 10
MAX_RESULT_COUNT = 30
MAX_CANDIDATES = 10000
LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="Patent Semantic Search",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --accent: #2563eb;
        --border: #d9e1ec;
        --muted: #617086;
        --surface: #ffffff;
        --surface-soft: #f7f9fc;
        --text: #172033;
    }

    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f6f8fb 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: var(--text);
        font-size: 2.35rem !important;
        letter-spacing: 0;
        margin-bottom: 0.2rem;
    }

    .intro {
        color: var(--muted);
        font-size: 1.02rem;
        margin-bottom: 1.4rem;
    }

    .result-card,
    .prompt-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: 0 14px 36px rgba(22, 35, 58, 0.07);
    }

    .prompt-card {
        padding: 0.95rem 1rem;
        margin: 1.2rem 0 1rem;
    }

    .prompt-label,
    .result-meta {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .prompt-text {
        margin-top: 0.25rem;
        color: var(--text);
        font-style: italic;
    }

    .result-card {
        padding: 1rem 1.05rem;
        margin-bottom: 0.8rem;
    }

    .result-title {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .score {
        display: inline-flex;
        align-items: center;
        min-height: 1.75rem;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        background: #e8f1ff;
        color: #1155cc;
        font-weight: 700;
    }

    .abstract {
        color: #2d3748;
        line-height: 1.55;
        margin-top: 0.75rem;
    }

    .stButton > button {
        min-height: 2.8rem;
        border-radius: 7px;
        border: 1px solid var(--accent);
        background: var(--accent);
        color: white;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #1d4ed8;
        background: #1d4ed8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_backend():
    index_path = BASE_DIR / "faiss_index.index"
    if not index_path.exists():
        raise FileNotFoundError(f"Could not find FAISS index: {index_path}")

    sources = discover_sources(BASE_DIR)
    faiss, SentenceTransformer = load_required_modules()
    index = faiss.read_index(str(index_path))
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return index, model, sources


def date_filter_controls(mode):
    lower_bound = None
    upper_bound = None

    if mode in {"After", "Between"}:
        lower_bound = st.date_input(
            "Publication date lower bound",
            value=None,
            min_value=date(1877, 1, 1),
            max_value=date.today(),
            format="YYYY-MM-DD",
        )
    if mode in {"Before", "Between"}:
        upper_bound = st.date_input(
            "Publication date upper bound",
            value=None,
            min_value=date(1877, 1, 1),
            max_value=date.today(),
            format="YYYY-MM-DD",
        )

    return lower_bound, upper_bound


def format_patent_date(raw_date):
    value = "".join(character for character in str(raw_date or "") if character.isdigit())
    if len(value) >= 8:
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return "not available"


def find_results(query, result_count, lower_bound, upper_bound):
    index, model, sources = load_backend()
    candidate_limit = max(result_count, 100)

    while candidate_limit <= MAX_CANDIDATES:
        results = search_patents(
            query,
            index,
            model,
            sources,
            top_k=result_count,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            candidate_limit=candidate_limit,
        )
        if len(results) >= result_count or candidate_limit == MAX_CANDIDATES:
            return results, candidate_limit
        candidate_limit = min(candidate_limit * 3, MAX_CANDIDATES)

    return results, candidate_limit


st.title("Patent Semantic Search")
st.markdown(
    '<p class="intro">Describe an invention idea and find semantically similar German patents.</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    query = st.text_area(
        "Describe your invention idea",
        height=140,
        placeholder=(
            "Example: A bicycle light powered by a wheel-mounted dynamo that turns "
            "on automatically while the bicycle is moving."
        ),
    )

    filter_column, result_column, button_column = st.columns([2.2, 1.1, 1])
    with filter_column:
        date_mode = st.selectbox(
            "Publication date filter",
            ["No date filter", "After", "Before", "Between"],
        )
        lower_bound, upper_bound = date_filter_controls(date_mode)

    with result_column:
        result_count = st.number_input(
            "Patents to show",
            min_value=1,
            max_value=MAX_RESULT_COUNT,
            value=DEFAULT_RESULT_COUNT,
            step=1,
        )

    with button_column:
        st.write("")
        st.write("")
        submitted = st.button("Search", use_container_width=True)

if submitted:
    cleaned_query = " ".join(query.split())
    if not cleaned_query:
        st.warning("Please enter an invention idea before searching.")
    elif lower_bound and upper_bound and lower_bound > upper_bound:
        st.warning("The lower publication date must be before the upper publication date.")
    else:
        st.session_state["last_query"] = cleaned_query
        with st.spinner("Searching patents..."):
            try:
                results, candidate_limit = find_results(
                    cleaned_query,
                    int(result_count),
                    lower_bound,
                    upper_bound,
                )
            except Exception:
                LOGGER.exception("Patent search failed")
                st.error(
                    "Search failed. Verify that the local model and patent search "
                    "assets are available."
                )
                results = []
                candidate_limit = None
        st.session_state["last_results"] = results
        st.session_state["last_candidate_limit"] = candidate_limit
        st.session_state["last_requested_count"] = int(result_count)

if "last_query" in st.session_state:
    st.markdown(
        f"""
        <div class="prompt-card">
            <div class="prompt-label">Your search</div>
            <div class="prompt-text">{escape(st.session_state["last_query"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

results = st.session_state.get("last_results")
if results is not None:
    if not results:
        st.info("No matching patents were found for the current date filter.")
    else:
        requested_count = st.session_state.get("last_requested_count", DEFAULT_RESULT_COUNT)
        candidate_limit = st.session_state.get("last_candidate_limit")
        st.caption(
            f"Showing {len(results)} of {requested_count} requested patents"
            + (f" from the top {candidate_limit} semantic candidates." if candidate_limit else ".")
        )

        for rank, result in enumerate(results, start=1):
            patent = result.patent
            publication_number = patent.get("publication_number") or "unknown publication"
            publication_date = format_patent_date(patent.get("publication_date"))
            title = patent.get("title") or "Untitled patent"
            abstract = shorten(patent.get("abstract"), width=520)
            escaped_publication_number = escape(publication_number)
            escaped_publication_date = escape(publication_date)
            escaped_title = escape(title)
            escaped_abstract = escape(abstract)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-meta">#{rank} &middot; {escaped_publication_number} &middot; {escaped_publication_date}</div>
                    <div class="result-title">{escaped_title}</div>
                    <div class="result-meta">Similarity <span class="score">{result.score:.4f}</span></div>
                    <div class="abstract">{escaped_abstract}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
