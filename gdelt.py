import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt(query="india", maxrecords=50):
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "artlist",
        "maxrecords": maxrecords,
        "format": "json"
}

    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    data = r.json()

    articles = data.get("articles", [])
    return articles


def save_to_db(articles):
    coll = db["raw_articles"]

    for a in articles:
        doc = {
            "title": a.get("title"),
            "url": a.get("url"),
            "domain": a.get("sourceDomain"),
            "published": a.get("pubDate"),
            "scraped": False,     # we will fill later
            "content_raw": None,  # will fill later
            "createdAt": datetime.utcnow()
        }

        # insert if new
        if coll.count_documents({"url": doc["url"]}) == 0:
            coll.insert_one(doc)
            print("Inserted:", doc["title"])


def main():
    articles = fetch_gdelt(query="india", maxrecords=50)
    print("Fetched:", len(articles))
    save_to_db(articles)


if __name__ == "__main__":
    main()