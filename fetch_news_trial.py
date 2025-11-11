# multi_simple_ingest_chained_v2.py — fetch 20 articles per story

import os, requests
from pymongo import MongoClient, errors
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB = os.getenv("MONGO_DB")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

TIMEOUT = 20
MAX_ARTICLES_PER_STORY = 20  # 🔥 customize here

def extract_items(payload):
    """Extract list of articles from possibly nested response structures."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if isinstance(data, dict):
        for key in ["all_articles", "top_news", "top_headlines"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    def find_list_of_dicts(obj):
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                res = find_list_of_dicts(v)
                if res:
                    return res
        return []
    return find_list_of_dicts(data)

def ensure_doc(d):
    return d if isinstance(d, dict) else {"_value": d}

# ---------- Mongo & Headers ----------
client = MongoClient(MONGO_URI)
db = client[DB]

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST,
    "Accept": "application/json"
}

# ---------- STEP 1: Fetch Top Headlines ----------
TOP_HEADLINES_URL = f"https://{RAPIDAPI_HOST}/top-headlines"
params = {"country": "IN", "lang": "en"}

print("\n[INFO] Fetching top headlines...")
try:
    r = requests.get(TOP_HEADLINES_URL, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
except Exception as e:
    print(f"[ERROR] Failed to fetch headlines: {e}")
    client.close()
    exit(1)

headlines = extract_items(payload)
print(f"[INFO] Extracted {len(headlines)} headlines")

if not headlines:
    print("[WARNING] No headlines found — stopping.")
    client.close()
    exit(0)

coll_headlines = db["Top_Headline"]
coll_headlines.delete_many({})
coll_headlines.insert_many([ensure_doc(h) for h in headlines])
print("[INFO] Inserted top headlines into MongoDB")

# ---------- STEP 2: Collect Story IDs ----------
story_info = []
for h in headlines:
    sid = h.get("story_id") or h.get("storyId")
    if sid:
        story_info.append({"story_id": sid, "headline": h.get("title") or h.get("headline")})

story_info = {d["story_id"]: d for d in story_info}.values()  # dedup
print(f"[INFO] Found {len(story_info)} unique story IDs")

# ---------- STEP 3: Fetch Full Story Coverage ----------
FULL_STORY_URL = f"https://{RAPIDAPI_HOST}/full-story-coverage"
coll_full = db["Full_News"]
coll_full.delete_many({})

for i, info in enumerate(story_info, start=1):
    story_id = info["story_id"]
    print(f"\n[DEBUG] ({i}/{len(story_info)}) Fetching up to {MAX_ARTICLES_PER_STORY} articles for story {story_id}...")

    try:
        params = {"story": story_id, "sort": "DATE", "country": "IN", "lang": "en"}
        r = requests.get(FULL_STORY_URL, headers=headers, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        articles = extract_items(r.json())
        if not articles:
            print(f"[DEBUG] story {story_id}: no data returned")
            continue

        articles = articles[:MAX_ARTICLES_PER_STORY]
        doc = {
            "story_id": story_id,
            "headline": info.get("headline", "N/A"),
            "article_count": len(articles),
            "articles": articles
        }

        coll_full.replace_one({"story_id": story_id}, doc, upsert=True)
        print(f"[INFO] Inserted {len(articles)} articles for story {story_id}")

    except Exception as e:
        print(f"[ERROR] story {story_id}: {e}")
        continue

client.close()
print("\n[INFO] All done — headlines + full coverage inserted successfully.")
