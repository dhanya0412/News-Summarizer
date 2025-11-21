# frontend/pages/realorfake.py
import os
import random
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient

# -------------------- LOAD ENV --------------------
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
db = client[DB_NAME]

final_coll = db["final_dataset"]
fake_coll = db["fake_news_dataset"]

st.set_page_config(page_title="Real or Fake Quiz", layout="wide")

st.title("📰 REAL or FAKE — Headline Challenge")
st.caption("Guess which headline is REAL. Two are fake!")

# Optional: show uploaded debug image in sidebar (path from conversation)
uploaded_image_path = "/mnt/data/759abac5-1d8f-4513-90c1-bd517debbf69.png"
if Path(uploaded_image_path).exists():
    with st.sidebar:
        st.image(uploaded_image_path, use_column_width=True)
        st.caption("Debug image")

# -------------------- QUIZ LOADER --------------------
def load_quiz():
    """Fetch 1 real + 2 fake headlines from MongoDB and return structured state."""
    data = list(fake_coll.find({}, {"real_title": 1, "fake_title": 1, "_id": 0}))
    if len(data) < 3:
        # Not enough data to form the quiz
        return None

    random.shuffle(data)

    real = data[0]["real_title"]
    fake1 = data[1]["fake_title"]
    fake2 = data[2]["fake_title"]

    options = [real, fake1, fake2]
    random.shuffle(options)

    answer_map = {
        real: "real",
        fake1: "fake",
        fake2: "fake"
    }

    return {
        "real": real,
        "options": options,
        "answer_map": answer_map,
        "quiz_docs": data[:3],
        "answered": False,
        "result": None,
        "selected": None
    }


# Initialize session state safely (handles missing or None)
if "quiz_state" not in st.session_state or not st.session_state.get("quiz_state"):
    st.session_state.quiz_state = load_quiz()

# If load_quiz returned None (not enough data), inform and stop
if st.session_state.quiz_state is None:
    st.error("Not enough quiz data in `fake_news_dataset` collection to form a quiz (need ≥3 rows).")
    st.stop()

qs = st.session_state.quiz_state  # safe: now it's a dict

# -------------------- DISPLAY OPTIONS --------------------
st.subheader("Which one of these headlines is REAL?")

for idx, opt in enumerate(qs["options"], start=1):
    clean_opt = opt.split("|")[0].strip()
    # Use a compact button layout: each option as its own button
    if st.button(clean_opt, key=f"opt_{idx}"):
        if not qs["answered"]:
            qs["answered"] = True
            qs["selected"] = opt
            qs["result"] = (qs["answer_map"].get(opt) == "real")

st.write("---")

# -------------------- SHOW RESULT --------------------
if qs["answered"]:
    if qs["result"]:
        st.success("🎉 Correct! You found the real news!")
    else:
        st.error("❌ Nope… that was FAKE.")

    st.subheader("✔ The REAL headline was:")
    st.markdown(f"**{qs['real']}**")

    # lookup URL from final_dataset
    st.subheader("🔗 Source URL")
    real_doc = final_coll.find_one({"title": qs["real"]}, {"url": 1, "_id": 0})
    if real_doc and "url" in real_doc:
        st.markdown(f"[Click to read the article]({real_doc['url']})")
    else:
        st.warning("URL not found in final_dataset.")

    st.write("---")

    # -------------------- PLAY AGAIN --------------------
    if st.button("🔄 Play Again"):
        # Reinitialize quiz_state and rerun
        st.session_state.quiz_state = load_quiz()
        st.rerun()

# -------------------- BACK BUTTON (link-style) --------------------
st.write("\n")
# use a simple link that navigates to the app root (works reliably)
st.markdown("[⬅ Back to Dashboard](/)")

