import time
import os
import requests
from bs4 import BeautifulSoup
from readability import Document
from pymongo import MongoClient
from dotenv import load_dotenv
from langdetect import detect

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["raw_articles"]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/142.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def scrape_article(url):
    try:
        resp = requests.get(url, timeout=10, headers=headers)
        print("Status code:", resp.status_code, url)
        if resp.status_code != 200:
            return None

        doc = Document(resp.text)
        summary_html = doc.summary()
        if not summary_html.strip():
            print("Empty summary for:", url)
            return None

        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        return text
    except Exception as e:
        print("ERROR scraping:", url, e)
        return None


def process_unscraped():
    docs = list(coll.find({"scraped": False}))
    print("Found unscraped docs:", len(docs))


    for doc in docs:
        url = doc.get("url")
        print("Scraping:", url)

        text = scrape_article(url)

        if text and is_english(text):
            coll.update_one(
            {"_id": doc["_id"]},
            {"$set": {"scraped": True, "content_raw": text, "lang": "en"}}
    )
        else:
            print("Non-English — deleting:", url)
            coll.delete_one({"_id": doc["_id"]})



        time.sleep(1)

if __name__ == "__main__":
    process_unscraped()
