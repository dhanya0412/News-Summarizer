import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import re
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import random

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB = os.getenv("MONGO_DB")
GROQ_KEY = os.getenv("GROQ_API_KEY")

client = MongoClient(MONGO_URI)
db = client[DB]
collection = db["final_dataset"]

fake_coll = db["fake_news_dataset"]

API_URL = "https://api.groq.com/openai/v1/chat/completions"


#prompt to generate fake headline
def make_prompt(title):
    return f"""
Generate ONE fake news headline based on this real headline.

Real: "{title}"

Rules:
- Must be false but sound believable
- Must match the topic
- Should look like a real news headline
- Must be short and natural
- No explanation
- Return ONLY the headline
- Must not be a rephrasing of the real headline
"""

def clean_fake_headline(text):
    if not text:
        return ""
    text = text.strip()

    return text.strip("'\"")

def generate_fake_headline(title):

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": make_prompt(title)}
        ],
        "temperature": 0.8
    }

    while True:  
        response = requests.post(API_URL, json=payload, headers=headers)

        try:
            data = response.json()
        except:
            print("Non-JSON response:", response.text)
            return ""


        if "choices" in data:
            raw_fake = data["choices"][0]["message"]["content"].strip()
            return clean_fake_headline(raw_fake)

        if "error" in data and "rate limit" in data["error"]["message"].lower():
            msg = data["error"]["message"]
            m = re.search(r"try again in (\d+)", msg)
            wait_time = int(m.group(1)) if m else 2
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        print("Unexpected error:", data)
        return ""


cursor = collection.find({}, {"title": 1, "_id": 0})
headlines = [d["title"] for d in cursor if d.get("title")]

print(f"Loaded {len(headlines)} headlines.\n")

for idx, title in enumerate(headlines, start=1):
    print(f"[{idx}/{len(headlines)}]")
    print("REAL:", title)

    fake = generate_fake_headline(title)
    print("FAKE:", fake)
    print("--------------------------------------------")

    fake_coll.insert_one({
        "real_title": title,
        "fake_title": fake
    })

    time.sleep(0.1)  #delay 


print("\nCompleted generating fake headlines.")
print(f"Saved ALL headlines to 'fake_news_dataset' collection.")
