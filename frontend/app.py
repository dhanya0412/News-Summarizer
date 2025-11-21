# frontend/app.py
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient

# Load .env from project root (optional: keeps same env as other pages)
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")  # string
COLLECTION = "final_dataset"

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
    return client[DB_NAME]

db = get_db()
collection = db[COLLECTION]

# ------------------ TOP SEARCH BAR ------------------
st.title("MongoDB + Streamlit Demo")

search_query = st.text_input("Search", placeholder="Ask a question...")

if st.button("Go"):
    if search_query.strip():
        st.query_params = {"q": search_query.strip()}
        # Navigate to summary.py
        st.switch_page("pages/summary.py")

# ------------------ TOP 5 HEADLINES ------------------
st.subheader("Top 5 Headlines")

docs = list(collection.find({}, {"title": 1}).sort("_id", -1).limit(5))

for d in docs:
    st.write("•", d.get("title", "(no title)"))

# ------------------ FOOTER BUTTONS ------------------
# Create a bottom row with left and right aligned buttons
st.markdown("<br><br>", unsafe_allow_html=True)

# We'll create three columns: left (Play Trivia), center (spacer), right (Real or Fake)
c_left, c_center, c_right = st.columns([12, 5, 12])

with c_left:
    if st.button("🎲 PLAY TRIVIA"):
        # Navigate to the trivia page in pages/
        st.switch_page("pages/play_trivia.py")

with c_center:
    # spacer or additional info
    st.markdown("")  # keep center empty to push buttons to edges

with c_right:
    if st.button("📰 REAL or FAKE"):
        # Navigate to the real-or-fake quiz page
        st.switch_page("pages/realorfake.py")
