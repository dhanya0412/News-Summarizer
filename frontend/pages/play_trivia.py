# frontend/pages/play_trivia.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import random
import re

# Load .env from project root
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="Play Trivia", layout="wide")

# Ensure project root is importable if needed
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

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #F6F6F6 0%, #E8E8E8 100%);
    }
    
    /* Page header styling */
    h1 {
        color: #DD795D !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Question cards with gradient backgrounds */
    .question-card {
        background: linear-gradient(145deg, #ffffff 0%, #f0f0f0 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 6px solid #DD795D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .question-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Question number badge */
    .question-card h3 {
        color: #DD795D !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
    }
    
    /* Question text styling */
    .question-card p strong {
        color: #0F1B2A !important;
        font-size: 1.1em !important;
        line-height: 1.6 !important;
    }
    
    /* Streamlit button overrides for MCQ options */
    .stButton button {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        border: 2px solid #C9CECA;
        border-radius: 10px;
        padding: 15px 20px;
        font-size: 1em;
        color: #0F1B2A;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%);
        color: white;
        border-color: #DD795D;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(221, 121, 93, 0.3);
    }
    
    /* Primary button (correct answer) */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #28a745 0%, #20923b 100%) !important;
        color: white !important;
        border-color: #28a745 !important;
        font-weight: 600 !important;
    }
    
    /* Secondary button (wrong answer) */
    .stButton button[kind="secondary"]:disabled {
        background: linear-gradient(135deg, #e0e0e0 0%, #d0d0d0 100%) !important;
        color: #666 !important;
        border-color: #c0c0c0 !important;
        opacity: 0.7;
    }
    
    /* Text input styling */
    .stTextInput input {
        border: 2px solid #C9CECA !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 1em !important;
        background-color: white !important;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #DD795D !important;
        box-shadow: 0 0 0 2px rgba(221, 121, 93, 0.2) !important;
    }
    
    /* Select box styling */
    .stSelectbox select {
        border: 2px solid #C9CECA !important;
        border-radius: 8px !important;
        padding: 10px !important;
        background-color: white !important;
        color: #0F1B2A !important;
        font-weight: 500 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        color: #155724;
        font-weight: 600;
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        color: #721c24;
        font-weight: 600;
    }
    
    /* Info message styling */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #DD795D;
        border-radius: 8px;
        padding: 15px;
        color: #0c5460;
    }
    
    /* Category header styling */
    h3 {
        color: #0F1B2A !important;
        font-weight: 600 !important;
    }
    
    /* Horizontal rule styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #DD795D, transparent);
        margin: 30px 0;
    }
    
    /* Caption styling */
    .caption {
        color: #666 !important;
        font-style: italic;
    }
    
    /* Play button special styling */
    div[data-testid="column"]:nth-child(2) .stButton button {
        background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        padding: 12px 24px !important;
    }
    
    div[data-testid="column"]:nth-child(2) .stButton button:hover {
        background: linear-gradient(135deg, #C96A4F 0%, #B55A3F 100%) !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

def parse_mcq_question(question_text):
    """Parse the question to extract main question and options"""
    # Split by newlines or common patterns
    parts = question_text.strip().split('\n')
    
    # First part is the main question (remove numbering if present)
    main_question = parts[0]
    # Remove leading numbering like "1. " or "Q1. "
    main_question = re.sub(r'^\d+\.\s*', '', main_question)
    
    # Extract options (a, b, c, d)
    options = {}
    for part in parts[1:]:
        part = part.strip()
        if part:
            # Match patterns like "a) text" or "a. text"
            match = re.match(r'^([a-d])[).]\s*(.+)', part, re.IGNORECASE)
            if match:
                option_letter = match.group(1).lower()
                option_text = match.group(2)
                options[option_letter] = option_text
    
    return main_question, options

st.title("🎲 Play Trivia")
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em; margin-bottom: 30px;'>Choose a topic and test your knowledge with multiple choice questions!</p>", unsafe_allow_html=True)

# Category mapping
categories = {
    "Sports": "sports",
    "Entertainment": "entertainment",
    "Health": "health",
    "Technology": "technology",
    "India": "india",
    "Business": "business"
}

# Initialize session state
if "trivia_answered" not in st.session_state:
    st.session_state.trivia_answered = set()

# Controls: select box and play button
col1, col2, col3 = st.columns([3, 2, 5])
with col1:
    selected_label = st.selectbox("Select category", list(categories.keys()))
    selected = categories[selected_label]
with col2:
    if st.button("🎮 Play", use_container_width=True):
        st.session_state["play_trivia_start"] = True
        st.session_state["trivia_category"] = selected
        st.session_state["trivia_answered"] = set()
        st.session_state.pop("trivia_questions", None)
        st.rerun()

# If user started playing
if st.session_state.get("play_trivia_start"):
    category = st.session_state.get("trivia_category", selected)
    coll_name = f"trivia_{category}"
    coll = db[coll_name]

    # Fetch data
    try:
        docs = list(coll.find())
    except Exception as e:
        st.error(f"Could not fetch trivia data from collection '{coll_name}': {e}")
        st.stop()

    if not docs:
        st.warning(f"No trivia entries found for category '{category}'.")
        st.stop()

    # Shuffle and pick first 5
    if "trivia_questions" not in st.session_state or st.session_state.get("trivia_category") != category:
        random.shuffle(docs)
        st.session_state["trivia_questions"] = docs[:5] if len(docs) >= 5 else docs

    questions = st.session_state["trivia_questions"]
    
    st.markdown(f"<div style='background: linear-gradient(135deg, #DD795D 0%, #C96A4F 100%); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 30px;'><h3 style='color: white !important; margin: 0;'>📚 Category: {selected_label}</h3><p style='color: white; margin: 5px 0 0 0;'>Answer all {len(questions)} questions below</p></div>", unsafe_allow_html=True)

    # Render questions
    for i, doc in enumerate(questions, start=1):
        qtext = doc.get("question") if isinstance(doc, dict) else doc["question"]
        correct_answer = doc.get("answer", "").strip().lower() if isinstance(doc, dict) else doc["answer"].strip().lower()
        
        # Parse question and options
        main_question, options = parse_mcq_question(qtext)
        
        st.markdown(f"<div class='question-card'>", unsafe_allow_html=True)
        st.markdown(f"### Question {i}")
        st.markdown(f"**{main_question}**")
        
        # Create a unique key for this question
        q_key = f"q_{category}_{i}"
        
        # Check if this question has been answered
        is_answered = q_key in st.session_state.trivia_answered
        
        # Display options as buttons if MCQ, otherwise show text input
        if options:
            cols = st.columns(2)
            for idx, (option_letter, option_text) in enumerate(sorted(options.items())):
                with cols[idx % 2]:
                    button_key = f"{q_key}_{option_letter}"
                    
                    # Determine button styling
                    if is_answered:
                        selected_answer = st.session_state.get(f"{q_key}_selected")
                        if option_letter == correct_answer:
                            button_label = f"✓ {option_letter.upper()}) {option_text}"
                            button_type = "primary"
                        elif option_letter == selected_answer and option_letter != correct_answer:
                            button_label = f"✗ {option_letter.upper()}) {option_text}"
                            button_type = "secondary"
                        else:
                            button_label = f"{option_letter.upper()}) {option_text}"
                            button_type = "secondary"
                        
                        st.button(button_label, key=button_key, disabled=True, use_container_width=True, type=button_type)
                    else:
                        button_label = f"{option_letter.upper()}) {option_text}"
                        if st.button(button_label, key=button_key, use_container_width=True):
                            # Record the answer
                            st.session_state.trivia_answered.add(q_key)
                            st.session_state[f"{q_key}_selected"] = option_letter
                            
                            # Show feedback
                            if option_letter == correct_answer:
                                st.session_state[f"{q_key}_correct"] = True
                            else:
                                st.session_state[f"{q_key}_correct"] = False
                            
                            st.rerun()
            
            # Show explanation if answered
            if is_answered:
                st.markdown("---")
                selected = st.session_state.get(f"{q_key}_selected")
                if selected == correct_answer:
                    st.success(f"✓ Correct! The answer is **{correct_answer.upper()}**")
                else:
                    st.error(f"✗ You selected **{selected.upper()}**. The correct answer is **{correct_answer.upper()}**")
        else:
            # Fill-in-the-blank type question
            if not is_answered:
                user_answer = st.text_input("Your answer:", key=f"{q_key}_input")
                if st.button("Submit Answer", key=f"{q_key}_submit"):
                    st.session_state.trivia_answered.add(q_key)
                    st.session_state[f"{q_key}_selected"] = user_answer.strip().lower()
                    st.rerun()
            else:
                user_answer = st.session_state.get(f"{q_key}_selected", "")
                st.text_input("Your answer:", value=user_answer, key=f"{q_key}_input_disabled", disabled=True)
                st.markdown("---")
                if user_answer == correct_answer:
                    st.success(f"✓ Correct! The answer is **{correct_answer}**")
                else:
                    st.error(f"✗ You answered **{user_answer}**. The correct answer is **{correct_answer}**")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

    # Play again button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Play Again", use_container_width=True):
            st.session_state["play_trivia_start"] = True
            st.session_state["trivia_answered"] = set()
            st.session_state.pop("trivia_questions", None)
            st.rerun()

else:
    st.info("👆 Pick a category and click Play to begin the trivia!")

st.caption("💡 Trivia data is pulled from MongoDB collections named `trivia_<category>`")