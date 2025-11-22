import os
import sys
from pathlib import Path
from dotenv import load_dotenv

#env loaidng
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="News Summarizer", layout="wide", initial_sidebar_state="collapsed")

#project root to sys.path``
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#summarizer import
try:
    import summarizer as summ_mod
except Exception:
    st.error("Could not import summarizer.py from project root.")
    st.stop()

try:
    import search as search_mod
except Exception:
    st.error("Could not import search.py from project root.")
    st.stop()

for fn in ("generate_summary", "answer_followup"):
    if not hasattr(summ_mod, fn):
        st.error(f"summarizer.py missing required function: {fn}")
        st.stop()

#congfig to mongodb
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")
COLLECTION = "final_dataset"

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
    return client[DB_NAME]

db = get_db()
collection = db[COLLECTION]

st.markdown("""
<style>
    .main{
        background: linear-gradient(135deg, #F6F6F6 0%, #E8E8E8 100%);
    }
    
    .nav-bar{
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%);
        padding: 20px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-title{
        color: black;
        font-size: 2em;
        font-weight: 700;
        margin: 0;
    }
    
    .back-button{
        background: white !important;
        color: #DD795D !important;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .back-button:hover{
        background: #f0f0f0 !important;
        transform: translateX(-3px);
    }
    
    .search-section{
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    .stTextInput input{
        border: 2px solid #C9CECA !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 1.05em !important;
    }
    
    .stTextInput input:focus{
        border-color: #DD795D !important;
        box-shadow: 0 0 0 3px rgba(221, 121, 93, 0.2) !important;
    }
    
    .content-section{
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    
    .section-header{
        color: #0F1B2A;
        font-size: 1.8em;
        font-weight: 600;
        margin-bottom: 20px;
        border-left: 5px solid #DD795D;
        padding-left: 15px;
    }
    
    .query-box{
        background: linear-gradient(135deg, #FFF5F2 0%, #FFE8E0 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #DD795D;
        margin-bottom: 20px;
    }
    
    .query-text{
        color: #0F1B2A;
        font-size: 1.2em;
        font-weight: 600;
        margin: 0;
    }
    
    .stExpander{
        background: #F9F9F9;
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    .stExpander:hover{
        border-color: #DD795D;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .stMetric{
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    .stMetric label{
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2em !important;
    }
    
    .stButton button{
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(221, 121, 93, 0.4) !important;
    }
    
    .summary-box{
        background: linear-gradient(135deg, #F0F8FF 0%, #E6F2FF 100%);
        padding: 25px;
        border-radius: 10px;
        border-left: 4px solid #4A90E2;
        margin: 20px 0;
        line-height: 1.8;
        font-size: 1.05em;
    }
    
    .stSuccess{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stError{
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stInfo{
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #17a2b8;
        border-radius: 10px;
        padding: 15px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<div class="nav-title">News Summarizer</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<a href="/" class="back-button">← Home</a>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

raw_q = st.query_params.get("q", None)
if isinstance(raw_q, list) and raw_q:
    current_query = raw_q[0].strip()
else:
    current_query = (raw_q or "").strip()

if not current_query:
    qs_state = st.session_state.get("q", None)
    if isinstance(qs_state, list) and qs_state:
        current_query = qs_state[0].strip()
    elif isinstance(qs_state, str) and qs_state.strip():
        current_query = qs_state.strip()

#search section
st.markdown('<div class="search-section">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    new_search = st.text_input(
        "Search", 
        value=current_query or "", 
        placeholder="🔍 Ask a question...",
        label_visibility="collapsed"
    )
with col2:
    if st.button("Search", use_container_width=True):
        if (new_search or "").strip():
            st.query_params = {"q": new_search.strip()}
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

raw_q = st.query_params.get("q", None)
if isinstance(raw_q, list) and raw_q:
    query = raw_q[0].strip()
else:
    query = (raw_q or "").strip()

if not query:
    qs_state = st.session_state.get("q", None)
    if isinstance(qs_state, list) and qs_state:
        query = qs_state[0].strip()
    elif isinstance(qs_state, str) and qs_state.strip():
        query = qs_state.strip()

if not query:
    st.info("No query provided. Use the search box above or return to the dashboard.")
    st.stop()

# Display query
st.markdown('<div class="content-section">', unsafe_allow_html=True)
st.markdown('<div class="section-header"> Your Query</div>', unsafe_allow_html=True)
st.markdown(f'<div class="query-box"><p class="query-text">{query}</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Retrieval
st.markdown('<div class="content-section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Search Results</div>', unsafe_allow_html=True)

with st.spinner("Searching the database..."):
    try:
        results = search_mod.search(query, k=10, prune=True, debug=False)
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        st.stop()

if not results:
    st.warning("No results found. Try a different query.")
    st.stop()

#resultss
cols = st.columns([4, 1])
with cols[0]:
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)") if isinstance(r, dict) else getattr(r, "title", "(no title)")
        snippet = (r.get("content") or r.get("snippet") or "") if isinstance(r, dict) else (getattr(r, "content", "") or "")
        url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
        with st.expander(f" {i}. {title}", expanded=False):
            if snippet:
                st.write(snippet[:800] + ("..." if len(snippet) > 800 else ""))
            if url:
                st.markdown(f"[Read Full Article]({url})")

with cols[1]:
    st.metric("Documents", f"{len(results)}")
    if st.button("Refresh", use_container_width=True):
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

#cache the results
st.session_state["cached_docs"] = results
st.session_state["original_query"] = query

#get summary
st.markdown('<div class="content-section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">✨ AI Summary</div>', unsafe_allow_html=True)

with st.spinner("Generating summary..."):
    try:
        summary, cited = summ_mod.generate_summary(query, results)
    except Exception as e:
        st.error(f"Failed to generate summary :( : {str(e)}")
        st.stop()

st.markdown(f'<div class="summary-box">{summary or "No summary generated."}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="content-section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Ask a Follow Up question!!</div>', unsafe_allow_html=True)

followup = st.text_input("Type your follow-up question:", placeholder="Ask for more details...", key="followup_input")

if st.button("Get Answer", use_container_width=False):
    fu = (followup or "").strip()
    if not fu:
        st.warning("Please enter a follow-up question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                ans_text, ans_cited = summ_mod.answer_followup(fu, st.session_state["cached_docs"], st.session_state["original_query"])
            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")
                st.stop()

        st.markdown("**Answer:**")
        st.markdown(f'<div class="summary-box">{ans_text or "No answer generated."}</div>', unsafe_allow_html=True)

        if ans_cited:
            st.markdown("**Sources:**")
            for idx in ans_cited:
                if 1 <= idx <= len(st.session_state["cached_docs"]):
                    doc = st.session_state["cached_docs"][idx-1]
                    t = doc.get("title", "(no title)") if isinstance(doc, dict) else getattr(doc, "title", "(no title)")
                    st.markdown(f"• Document {idx}: {t}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)