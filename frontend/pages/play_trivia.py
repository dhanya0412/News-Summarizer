# frontend/pages/play_trivia.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import random

# Load .env from project root so it works the same as your CLI
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="Play Trivia", layout="wide")

# Ensure project root is importable if needed (not required but safe)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mongo config
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
    return client[DB_NAME]

db = get_db()

st.title("🎲 Play Trivia")
st.markdown("Choose a topic and press **Play**. You'll get 5 random questions (answers are hidden until you reveal them).")

# Category mapping same as your CLI
categories = {
    "Sports": "sports",
    "Entertainment": "entertainment",
    "Politics": "politics",
    "Health": "health",
    "Technology": "technology",
    "India": "india",
    "Business": "business"
}

# Controls: select box and play button
col1, col2, col3 = st.columns([3, 2, 5])
with col1:
    selected_label = st.selectbox("Select category", list(categories.keys()))
    selected = categories[selected_label]
with col2:
    if st.button("Play"):
        st.session_state["play_trivia_start"] = True
        st.session_state["trivia_category"] = selected
        # refresh page to show questions
        st.rerun()



# If user already started or pressing Play now, fetch questions
if st.session_state.get("play_trivia_start"):
    category = st.session_state.get("trivia_category", selected)
    coll_name = f"trivia_{category}"
    coll = db[coll_name]

    # fetch data
    try:
        docs = list(coll.find())  # get all docs in collection
    except Exception as e:
        st.error(f"Could not fetch trivia data from collection '{coll_name}': {e}")
        st.stop()

    if not docs:
        st.warning(f"No trivia entries found for category '{category}'.")
        st.stop()

    # shuffle and pick first 5
    random.shuffle(docs)
    questions = docs[:5] if len(docs) >= 5 else docs

    st.markdown(f"### Category: **{selected_label}** — Showing {len(questions)} questions")

    # Render the questions with hidden answers that reveal on click
    for i, doc in enumerate(questions, start=1):
        # Support both dict and BSON documents
        qtext = doc.get("question") if isinstance(doc, dict) else doc["question"]
        ans = doc.get("answer") if isinstance(doc, dict) else doc["answer"]

        # Remove potential leading numbering or whitespace (you had [3:] slice earlier)
        # But don't assume; show as-is with small cleaning
        qtext_clean = qtext.strip()
        # If the question starts like "Qn. " or "1. " remove first 0-3 chars if necessary
        # (You can remove the [3:] slicing unless your data always uses it)
        # qtext_clean = qtext_clean[3:] if len(qtext_clean) > 3 and qtext_clean[:3].isdigit() else qtext_clean

        with st.expander(f"Q{i}. {qtext_clean}", expanded=False):
            # show "Show answer" button per question
            key = f"reveal_{category}_{i}"
            if st.button("Show answer", key=key):
                st.markdown(f"**Answer:** {ans}")
            else:
                st.markdown("_Answer hidden — click 'Show answer' to reveal._")

    # Option to play again (reshuffle)
    st.markdown("---")
    if st.button("Play again (shuffle)"):
        st.session_state["play_trivia_start"] = True
        st.rerun()

else:
    st.info("Pick a category and click Play to begin the trivia.")

# Footer small help text
st.caption("Trivia data is pulled from MongoDB collections named `trivia_<category>` (for example `trivia_sports`).")
