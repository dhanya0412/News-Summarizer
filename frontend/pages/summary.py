# frontend/pages/summary.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root so Streamlit uses same env as terminal
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

import streamlit as st
from pymongo import MongoClient
import importlib
import importlib.util
import io
import contextlib

st.set_page_config(page_title="News Summarizer", layout="wide")

# Ensure project root is on sys.path so imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import summarizer and search modules from project root
try:
    import summarizer as summ_mod
except Exception:
    st.error("Could not import summarizer.py from project root. Check file and imports.")
    st.stop()

try:
    import search as search_mod
except Exception:
    st.error("Could not import search.py from project root. Check file and exports.")
    st.stop()

# Validate summarizer functions exist
for fn in ("generate_summary", "answer_followup"):
    if not hasattr(summ_mod, fn):
        st.error(f"summarizer.py missing required function: {fn}")
        st.stop()

# DB config (keep your variable names)
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")
COLLECTION = "final_dataset"

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
    return client[DB_NAME]

db = get_db()
collection = db[COLLECTION]

# ------------------------------------------------------------
# Top search bar (allows new queries from this page) + Back button
# ------------------------------------------------------------
# determine current query from params/session
raw_q = st.query_params.get("q", None)
if isinstance(raw_q, list) and raw_q:
    current_query = raw_q[0].strip()
else:
    current_query = (raw_q or "").strip()

# fallback to session_state if needed
if not current_query:
    qs_state = st.session_state.get("q", None)
    if isinstance(qs_state, list) and qs_state:
        current_query = qs_state[0].strip()
    elif isinstance(qs_state, str) and qs_state.strip():
        current_query = qs_state.strip()

# Top UI row: search input + buttons
top_col1, top_col2 = st.columns([6, 1])
with top_col1:
    new_search = st.text_input("Search the corpus", value=current_query or "", placeholder="Ask a question...")
with top_col2:
    if st.button("Search"):
        if (new_search or "").strip():
            # set query param (ensures URL updates) and rerun page
            st.query_params = {"q": new_search.strip()}
            st.experimental_rerun()

# Back to dashboard button (small and accessible)
st.write("\n")
# use a simple link that navigates to the app root (works reliably)
st.markdown("[⬅ Back to Dashboard](/)")

# Page title and description
st.title("News Summarizer")
st.markdown("Search the corpus, get a concise AI-generated summary, and ask follow-ups.")

# Sidebar: optional image (keeps UI polished)
uploaded_image_path = "/mnt/data/759abac5-1d8f-4513-90c1-bd517debbf69.png"
if Path(uploaded_image_path).exists():
    with st.sidebar:
        st.image(uploaded_image_path, use_column_width=True)
        st.caption("Project debug image")

# Re-evaluate query after any potential rerun/param change
raw_q = st.query_params.get("q", None)
if isinstance(raw_q, list) and raw_q:
    query = raw_q[0].strip()
else:
    query = (raw_q or "").strip()

# fallback to session_state if needed
if not query:
    qs_state = st.session_state.get("q", None)
    if isinstance(qs_state, list) and qs_state:
        query = qs_state[0].strip()
    elif isinstance(qs_state, str) and qs_state.strip():
        query = qs_state.strip()

if not query:
    st.info("No query provided. Use the search box above or go back to the dashboard.")
    st.stop()

# Show query
st.header("Query")
st.markdown(f"**{query}**")

# Step 1 — Retrieval (use your `search.py` exactly like the CLI)
st.header("Results")
with st.spinner("Running retrieval..."):
    try:
        results = search_mod.search(query, k=10, prune=True, debug=False)
    except Exception:
        st.error("Retrieval failed. See server logs for details.")
        st.stop()

if not results:
    st.warning("No results returned by retrieval. Please try a different query.")
    st.stop()

# Display retrieved documents in a clean list
cols = st.columns([3,1])
with cols[0]:
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)") if isinstance(r, dict) else getattr(r, "title", "(no title)")
        snippet = (r.get("content") or r.get("snippet") or "") if isinstance(r, dict) else (getattr(r, "content", "") or "")
        url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
        with st.expander(f"{i}. {title}", expanded=False):
            if snippet:
                st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))
            if url:
                st.markdown(f"[Source]({url})")
with cols[1]:
    st.metric("Retrieved", f"{len(results)} docs")
    st.markdown("")  # spacing
    if st.button("Refresh Summary"):
        # simply rerun to refresh
        st.experimental_rerun()

# Cache docs for follow-ups (exact CLI behavior)
st.session_state["cached_docs"] = results
st.session_state["original_query"] = query

# Step 2 — Summarization
st.header("Summary")
with st.spinner("Generating summary..."):
    try:
        summary, cited = summ_mod.generate_summary(query, results)
    except Exception:
        st.error("Failed to generate summary. Check the summarizer implementation or API keys.")
        st.stop()

# Nicely show summary and sources
st.write(summary or "No summary generated.")

# Step 3 — Follow-up interaction
st.header("Follow-up")
followup = st.text_input("Ask a follow-up question:", key="followup_input")

if st.button("Get Follow-up Answer"):
    fu = (followup or "").strip()
    if not fu:
        st.warning("Please enter a follow-up question.")
    else:
        with st.spinner("Generating follow-up answer..."):
            try:
                ans_text, ans_cited = summ_mod.answer_followup(fu, st.session_state["cached_docs"], st.session_state["original_query"])
            except Exception:
                st.error("Failed to generate follow-up answer. Check summarizer implementation.")
                st.stop()

        st.markdown("**Answer**")
        st.write(ans_text or "No answer generated.")

        if ans_cited:
            st.markdown("**Cited documents for follow-up:**")
            for idx in ans_cited:
                if 1 <= idx <= len(st.session_state["cached_docs"]):
                    doc = st.session_state["cached_docs"][idx-1]
                    t = doc.get("title", "(no title)") if isinstance(doc, dict) else getattr(doc, "title", "(no title)")
                    st.markdown(f"- Doc {idx}: {t}")
                else:
                    st.markdown(f"- Doc {idx}: (index out of range)")

# Small footer
st.write("---")
st.caption("Summarizer UI — powered by your backend summarizer.py. No debug prints shown.")
