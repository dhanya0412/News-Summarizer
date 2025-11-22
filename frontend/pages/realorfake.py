import os
import random
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient

project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI) if MONGO_URI else MongoClient()
db = client[DB_NAME]

final_coll = db["final_dataset"]
fake_coll = db["fake_news_dataset"]

st.set_page_config(
    page_title="Real or Fake Quiz", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

#CSS for styling
st.markdown("""
<style>
    .main{
        background: linear-gradient(135deg, #F6F6F6 0%, #E8E8E8 100%);
    }
    
    .hero-section{
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%);
        padding: 50px 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title{
        color: white;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle{
        color: white;
        font-size: 1.2em;
        opacity: 0.95;
    }
    
    .back-link{
        display: inline-block;
        background: white;
        color: #DD795D;
        padding: 12px 25px;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .back-link:hover{
        background: #f0f0f0;
        transform: translateX(-3px);
    }
    
    .quiz-container{
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 30px;
    }
    
    .quiz-question{
        color: #0F1B2A;
        font-size: 1.5em;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 3px solid #DD795D;
    }
    
    .stButton button{
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%) !important;
        color: #0F1B2A !important;
        border: 3px solid #C9CECA !important;
        padding: 25px 20px !important;
        border-radius: 15px !important;
        font-size: 1.1em !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1) !important;
        margin: 10px 0 !important;
    }
    
    .stButton button:hover{
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%) !important;
        color: white !important;
        border-color: #DD795D !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(221, 121, 93, 0.3) !important;
    }
 
    .result-container{
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .result-title{
        font-size: 1.4em;
        font-weight: 600;
        color: #0F1B2A;
        margin-bottom: 15px;
        border-left: 5px solid #DD795D;
        padding-left: 15px;
    }
    
    .real-headline{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 15px 0;
        font-size: 1.1em;
        line-height: 1.6;
    }
    
    .source-link{
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 15px 0;
        text-align: center;
    }
    
    .source-link a{
        color: #1976D2;
        font-weight: 600;
        text-decoration: none;
        font-size: 1.1em;
    }
    
    .source-link a:hover{
        color: #0D47A1;
        text-decoration: underline;
    }

    .play-again-btn button{
        background: linear-gradient(135deg, #28a745 0%, #20923b 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 40px !important;
        font-size: 1.2em !important;
    }
    
    .play-again-btn button:hover{
        background: linear-gradient(135deg, #20923b 0%, #1e7e34 100%) !important;
        transform: scale(1.05) !important;
    }
    
    .stSuccess{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        font-size: 1.1em;
        font-weight: 600;
    }
    
    .stError{
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        font-size: 1.1em;
        font-weight: 600;
    }
    
    .stWarning{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🔍 REAL or FAKE</div>
    <div class="hero-subtitle">Can you spot the difference? Test your news literacy!</div>
</div>
""", unsafe_allow_html=True)

#return to home
st.markdown('<a href="/" class="back-link">← Back to Home</a>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

#loads
def load_quiz():
    """Fetch 1 real + 2 fake headlines and return structured state."""
    data = list(fake_coll.find({}, {"real_title": 1, "fake_title": 1, "_id": 0}))
    if len(data) < 3:
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

if "quiz_state" not in st.session_state or not st.session_state.get("quiz_state"):
    st.session_state.quiz_state = load_quiz()


if st.session_state.quiz_state is None:
    st.error("Not enough quiz data in `fake_news_dataset` collection (need ≥3 rows).")
    st.stop()

qs = st.session_state.quiz_state


st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
st.markdown('<div class="quiz-question">Which one of these headlines is REAL?</div>', unsafe_allow_html=True)

for idx, opt in enumerate(qs["options"], start=1):
    clean_opt = opt.split("|")[0].strip()
    if st.button(f"📰 {clean_opt}", key=f"opt_{idx}"):
        if not qs["answered"]:
            qs["answered"] = True
            qs["selected"] = opt
            qs["result"] = (qs["answer_map"].get(opt) == "real")
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

#display results
if qs["answered"]:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if qs["result"]:
        st.success("WUHU! You nailed it! Spotted the real news!")
    else:
        st.error("OH NO! That was a fake headline. I suggest you to go read more news articles!")
    
    #show the asneer
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">The REAL headline was:</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="real-headline">{qs["real"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    #source url
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">🔗 Verify the Source</div>', unsafe_allow_html=True)
    real_doc = final_coll.find_one({"title": qs["real"]}, {"url": 1, "_id": 0})
    if real_doc and "url" in real_doc:
        st.markdown(f'<div class="source-link"><a href="{real_doc["url"]}" target="_blank">📖 Read the Full Article</a></div>', unsafe_allow_html=True)
    else:
        st.warning("Source URL not found in database.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    #play again
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown('<div class="play-again-btn">', unsafe_allow_html=True)
        if st.button("Play Again", use_container_width=True):
            st.session_state.quiz_state = load_quiz()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)