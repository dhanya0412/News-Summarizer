# frontend/app.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient

# Load .env from project root
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()

# ADD THESE LINES TO FIX THE IMPORT
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")
COLLECTION = "final_dataset"

# ... rest of your code

st.set_page_config(
    page_title="News Intelligence Hub",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def get_db():
    client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
    return client[DB_NAME]

db = get_db()
collection = db[COLLECTION]

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #F6F6F6 0%, #E8E8E8 100%);
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%);
        padding: 60px 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        color: white;
        font-size: 3.5em;
        font-weight: 700;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        color: white;
        font-size: 1.3em;
        opacity: 0.95;
        margin-bottom: 30px;
    }
    
    /* Search bar container */
    .search-container {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 40px;
    }
    
    /* Search input styling */
    .stTextInput input {
        border: 2px solid #C9CECA !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #DD795D !important;
        box-shadow: 0 0 0 3px rgba(221, 121, 93, 0.2) !important;
    }
    
    /* Search button */
    .search-button button {
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 40px !important;
        border-radius: 10px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(221, 121, 93, 0.3) !important;
    }
    
    .search-button button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(221, 121, 93, 0.4) !important;
    }
    
    /* Headlines section */
    .headlines-section {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 40px;
    }
    
    .section-title {
        color: #0F1B2A;
        font-size: 1.8em;
        font-weight: 600;
        margin-bottom: 20px;
        border-left: 5px solid #DD795D;
        padding-left: 15px;
    }
    
    .headline-item {
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #F6F6F6 0%, #EFEFEF 100%);
        border-radius: 10px;
        border-left: 4px solid #DD795D;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .headline-item:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .headline-text {
        color: #0F1B2A;
        font-size: 1.05em;
        line-height: 1.6;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 40px 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 100%;
        border: 2px solid transparent;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border-color: #DD795D;
    }
    
    .feature-icon {
        font-size: 4em;
        margin-bottom: 20px;
    }
    
    .feature-title {
        color: #0F1B2A;
        font-size: 1.5em;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .feature-description {
        color: #666;
        font-size: 1em;
        line-height: 1.6;
        margin-bottom: 25px;
    }
    
    /* Feature buttons */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        border-radius: 10px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(221, 121, 93, 0.4) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #666;
        font-size: 0.9em;
        margin-top: 50px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">📰 News Intelligence Hub</div>
    <div class="hero-subtitle">Your AI-powered gateway to news analysis, trivia, and fact-checking</div>
</div>
""", unsafe_allow_html=True)

# Search Section
st.markdown('<div class="search-container">', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    search_query = st.text_input(
        "Search",
        placeholder="🔍 Ask a question or search for news...",
        label_visibility="collapsed"
    )
with col2:
    st.markdown('<div class="search-button">', unsafe_allow_html=True)
    search_btn = st.button("Search", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if search_btn and search_query.strip():
    st.query_params = {"q": search_query.strip()}
    st.switch_page("pages/summary.py")

st.markdown('</div>', unsafe_allow_html=True)

# Headlines Section
st.markdown('<div class="headlines-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔥 Top 5 Headlines</div>', unsafe_allow_html=True)

docs = list(collection.find({}, {"title": 1, "content": 1, "description": 1, "snippet": 1, "url": 1}).sort("_id", -1).limit(5))

if docs:
    # Import summarizer HERE (after sys.path is set)
    try:
        import summarizer as summ_mod
    except ImportError as e:
        st.error(f"Could not import summarizer: {e}")
        st.stop()
    
    # Prepare docs for summarization
    headline_docs = []
    for d in docs:
        headline_docs.append({
            'title': d.get('title', 'Untitled'),
            'content': d.get('content', d.get('description', d.get('snippet', ''))),
            'url': d.get('url', '')
        })
    
    # Generate all summaries in ONE API call
    with st.spinner("🤖 Generating summaries..."):
        try:
            summaries = summ_mod.generate_multiple_summaries(headline_docs)
        except Exception as e:
            st.error(f"Error generating summaries: {e}")
            summaries = ["Summary unavailable."] * len(docs)
    
    # Display each headline with its summary
    for i, (d, summary) in enumerate(zip(docs, summaries), start=1):
        title = d.get("title", "(no title)")
        
        with st.expander(f"**{i}.** {title}", expanded=False):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F0F8FF 0%, #E6F2FF 100%);
                       padding: 20px;
                       border-radius: 8px;
                       border-left: 4px solid #4A90E2;
                       line-height: 1.7;
                       font-size: 1.05em;">
                {summary}
            </div>
            """, unsafe_allow_html=True)
            
            url = d.get('url', '')
            if url:
                st.markdown(f"🔗 [Read Full Article]({url})")

st.markdown('</div>', unsafe_allow_html=True)
# Feature Cards Section
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎲</div>
        <div class="feature-title">Play Trivia</div>
        <div class="feature-description">
            Test your knowledge across multiple categories including Sports, Entertainment, 
            Technology, and more. Challenge yourself with engaging questions!
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🎮 START TRIVIA", key="trivia_btn"):
        st.switch_page("pages/play_trivia.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Real or Fake</div>
        <div class="feature-description">
            Put your news literacy to the test! Can you spot the difference between 
            real headlines and fake news? Sharpen your critical thinking skills.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🕵️ TEST YOUR SKILLS", key="fake_btn"):
        st.switch_page("pages/realorfake.py")

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by MongoDB & Streamlit | Built for News Intelligence</p>
</div>
""", unsafe_allow_html=True)