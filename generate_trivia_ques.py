import os
import spacy
import trafilatura
import sys
import re
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
from google.generativeai import GenerativeModel
import os
# ----------------------------------------------------
# LOAD ENV + SETUP API
# ----------------------------------------------------
load_dotenv()
gem_key = os.getenv("GEMINI_KEY")
genai.configure(api_key=gem_key)

sys.stdout.reconfigure(encoding='utf-8')

MONGO_URI = os.getenv("MONGO_URI")
DB = os.getenv("MONGO_DB")

model = GenerativeModel("gemini-2.5-flash")

client = MongoClient(MONGO_URI)
db = client[DB]
collection = db["raw_articles"]

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------------
# SUMMARIZATION UTILS
# ----------------------------------------------------
def get_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def compute_sentence_scores(sentences):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1)
    return [score.item() for score in scores]

def get_top_sentences(text, top_n=3):
    sentences = get_sentences(text)
    if not sentences:
        return []
    scores = compute_sentence_scores(sentences)
    ranked = list(zip(sentences, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]

# ----------------------------------------------------
# PARSE QUESTION BLOCKS PROPERLY
# ----------------------------------------------------
def parse_questions_block(text):
    lines = text.strip().split("\n")
    questions = []
    block = []

    for line in lines:
        if re.match(r"^\d+\.", line.strip()):
            if block:
                questions.append("\n".join(block))
                block = []
        block.append(line.strip())

    if block:
        questions.append("\n".join(block))

    return questions[:4]   # limit to 3-4

# ----------------------------------------------------
# GEMINI QUESTION GENERATOR
# ----------------------------------------------------
def generate_questions_from_text(title, text):
    prompt = f"""
You are a trivia question generator.

Using ONLY the title and article content below, create exactly **4** trivia questions.

Allowed formats:
2 ques of Multiple-choice (MCQ)
2 ques of Fill-in-the-blank

Rules:
- Every question must be factual and short.
- Every MCQ must contain 4 options: a), b), c), d)
- Each question must include an "ans:" line.
- 1. and 2. should be MCQ and 3. and 4. must be fill in the blanks.
- No explanations. No extra text.

STRICT OUTPUT FORMAT:

1. <question?>
   a) <option>
   b) <option>
   c) <option>
   d) <option>
   ans: <a/b/c/d>
2. <question?>
   a) <option>
   b) <option>
   c) <option>
   d) <option>
   ans: <a/b/c/d>
3. <fill-in-the-blank sentence ___ >
   ans: <correct word/phrase>
4. <fill-in-the-blank sentence ___ >
   ans: <correct word/phrase>

TITLE:
{title}

CONTENT:
{text}
"""

    response = model.generate_content(prompt)

    if not response or not response.text:
        return []

    return parse_questions_block(response.text)

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
categories = ["technology", "sports", "entertainment", "business", "health", "india","politics"]

for cat in categories:

    cursor = collection.find(
        {"category": cat},
        {"title": 1, "url": 1, "_id": 0}
    )

    for article in cursor:
        title = article.get("title")
        url = article.get("url")

        if not url:
            continue

        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=True
        )

        if not content or len(content) < 50:
            continue

        full_text = f"{title}. {content}"

        top_sentences = get_top_sentences(full_text, top_n=3)
        combined_text = " ".join(top_sentences)

        questions = generate_questions_from_text(title, combined_text)

        # Create collection: trivia_<category>
        trivia_collection = db[f"trivia_{cat}"]

        for q in questions:
            # Detect answer part
            ans_index = q.lower().rfind("ans:")
            if ans_index == -1:
                continue

            question_text = q[:ans_index].strip()
            answer = q[ans_index + 4:].strip()  # after "ans:"

            # Detect type
            if "a)" in q and "b)" in q:
                qtype = "mcq"
            else:
                qtype = "fill"

            # Insert into DB
            trivia_collection.insert_one({
                "title": title,
                "category": cat,
                "question": question_text,
                "answer": answer,
                "type": qtype,
                "source_url": url
            })

        print("=" * 90)
        time.sleep(7)
