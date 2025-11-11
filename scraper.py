# backend/ingest/scraper.py
import time
import os
import requests
from bs4 import BeautifulSoup
from readability import Document
from pymongo import MongoClient
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from dateutil import parser as dateparser
from urllib.parse import urlparse

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["raw_articles"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/142.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def safe_detect_lang(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"

def extract_publish_date_from_html(soup):
    # look for common meta tags for published date
    selectors = [
        ('meta', {'property': 'article:published_time'}),
        ('meta', {'name': 'pubdate'}),
        ('meta', {'name': 'publishdate'}),
        ('meta', {'name': 'timestamp'}),
        ('meta', {'property': 'og:published_time'}),
        ('meta', {'name': 'date'}),
        ('meta', {'itemprop': 'datePublished'}),
        ('time', {}),
    ]
    for tag, attrs in selectors:
        try:
            if tag == 'time':
                t = soup.find('time')
                if t and t.get('datetime'):
                    return t.get('datetime')
                elif t:
                    return t.get_text().strip()
            else:
                m = soup.find(tag, attrs=attrs)
                if m:
                    val = m.get('content') or m.get('value') or m.get_text()
                    if val:
                        return val.strip()
        except Exception:
            pass
    return None

def parse_date_str(datestr):
    if not datestr:
        return None
    try:
        return dateparser.parse(datestr).isoformat()
    except Exception:
        return None

def scrape_article(url):
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        print("HTTP", resp.status_code, url)
        if resp.status_code != 200:
            return None, None

        doc = Document(resp.text)
        summary_html = doc.summary()
        if not summary_html or not summary_html.strip():
            # fallback: use whole page text
            soup_fallback = BeautifulSoup(resp.text, "html.parser")
            text = soup_fallback.get_text(separator="\n").strip()
        else:
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(separator="\n").strip()

        #parse published date from original page if possible
        soup_full = BeautifulSoup(resp.text, "html.parser")
        pub_raw = extract_publish_date_from_html(soup_full)
        pub_iso = parse_date_str(pub_raw)

        return text, pub_iso
    except Exception as e:
        print("Scrape error:", url, e)
        return None, None

def process_unscraped(batch_size=20, delete_non_en=True):
    while True:
        docs = list(coll.find({"scraped": False}).limit(batch_size))
        if not docs:
            print("No more unscraped docs.")
            break

        print("Found batch:", len(docs))
        for doc in docs:
            url = doc.get("url")
            if not url:
                coll.update_one({"_id": doc["_id"]}, {"$set": {"scraped": True}})
                continue

            print("Scraping:", url)
            text, pub_iso = scrape_article(url)

            if not text:
                print("Failed to extract text for:", url)
                coll.update_one({"_id": doc["_id"]}, {"$set": {"scraped": True, "content_raw": None}})
                time.sleep(1)
                continue

            lang = safe_detect_lang(text)
            if lang != "en":
                print("Non-English detected:", lang, "for", url)
                if delete_non_en:
                    coll.delete_one({"_id": doc["_id"]})
                    print("Deleted non-English doc:", url)
                else:
                    coll.update_one({"_id": doc["_id"]}, {"$set": {"scraped": True, "lang": lang, "content_raw": None}})
                time.sleep(1)
                continue

            # domain from URL if missing
            domain = doc.get("domain") or urlparse(url).netloc.lower()

            # published: prefer existing doc published field, else scraped pub_iso
            published = doc.get("published") or pub_iso

            coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "scraped": True,
                    "content_raw": text,
                    "lang": "en",
                    "domain": domain,
                    "published": published
                }}
            )
            print("OK saved:", url, "published:", published)
            time.sleep(1)  # polite crawl

if __name__ == "__main__":
    process_unscraped(batch_size=10, delete_non_en=True)
