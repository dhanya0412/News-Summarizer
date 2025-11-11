# backend/ingest/gdelt.py
import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["raw_articles"]

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

WIPE = False   # set True to reset DB

#gledt timestamp helper
def gdelt_timestamp(dt: datetime):
    return dt.strftime("%Y%m%d%H%M%S")
def get_last_month_range():
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    return gdelt_timestamp(start), gdelt_timestamp(end)

TOPICS = [
    "india",
    "world",
    "business",
    "technology",
    "health",
    "sports",
    "entertainment",
    "environment",
    "education",
    "crime",
    "politics",
]


def fetch_gdelt(query: str, maxrecords=10):
    start_ts, end_ts = get_last_month_range()

    params = {
        "query": f"{query} sourcelang:english",
        "mode": "artlist",
        "maxrecords": maxrecords,
        "format": "json",
        "startdatetime": start_ts,
        "enddatetime": end_ts,
    }

    for attempt in range(3):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("articles") or data.get("data") or []
        except Exception as e:
            print(f"Attempt {attempt+1}: error → {e}")
            time.sleep(2)

    return []


def __get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return None


#parse publishing date
def parse_pubdate(raw):
    """
    Convert GDELT timestamp → datetime
    Example format: "20250210153000"
    """
    if not raw:
        return None

    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S")
    except:
        return None


def save_metadata_to_db(articles, category: str):
    inserted = 0

    for a in articles:
        url = a.get("url") or a.get("documentidentifier")
        if not url:
            continue
        raw_pub = a.get("pubDate") or a.get("seendate")

        doc = {
            "title": a.get("title"),
            "url": url,
            "domain": a.get("sourceDomain") or __get_domain(url),
            "published": parse_pubdate(raw_pub),   
            "category": category,                  
            "scraped": False,
            "content_raw": None,
            "createdAt": datetime.utcnow(),
        }

        if coll.count_documents({"url": doc["url"]}) == 0:
            coll.insert_one(doc)
            inserted += 1
    return inserted


def main():
    if WIPE:
        print("WIPE mode: deleting existing raw_articles")
        coll.delete_many({})

    total_inserted = 0

    for topic in TOPICS:
        try:
            print(f"\nFetching topic: {topic}")

            articles = fetch_gdelt(topic)
            print("Fetched:", len(articles))

            inserted = save_metadata_to_db(articles, category=topic)
            total_inserted += inserted

            print(f"Inserted {inserted} new docs for '{topic}'")

            time.sleep(2)

        except Exception as e:
            print("Error fetching topic", topic, e)

    print("\nDone. Total new inserted:", total_inserted)


if __name__ == "__main__":
    main()
