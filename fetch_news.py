# multi_simple_ingest.py — minimal multi-endpoint ingester (insert all, no doc prints)
import os, requests
from pymongo import MongoClient, errors

# ---------- config (edit) ----------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://sharmaraashi21_db_user:Z4I3wRM2SJbEacvb@cluster0.mxuvera.mongodb.net/"
)
DB = os.getenv("MONGO_DB", "News")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "your api key here")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "real-time-news-data.p.rapidapi.com")

# map endpoint_path -> collection_name
ENDPOINTS = {
    "/top-headlines": "Top_Headline",
    "/full-story-coverage": "Full_News",
}

# ---------- params per endpoint ----------
# You can edit these per endpoint if needed
COMMON_PARAMS = {
    "/top-headlines": {
        "country": "US",
        "lang": "en"
    },
    "/full-story-coverage": {
        "story": "CAAqNggKIjBDQklTSGpvSmMzUnZjbmt0TXpZd1NoRUtEd2pzbFA3X0N4RjlDUlpVVnhudXBpZ0FQAQ",
        "sort": "RELEVANCE",
        "country": "US",
        "lang": "en"
    }
}

TIMEOUT = 20


def extract_items(payload):
    """Extract list of articles from possibly nested response structures."""
    if not isinstance(payload, dict):
        return []

    data = payload.get("data", payload)

    # Most endpoints: data contains all_articles / top_news / twitter_posts
    if isinstance(data, dict):
        # Prefer all_articles (your case)
        if "all_articles" in data and isinstance(data["all_articles"], list):
            print(f"[DEBUG] Found all_articles with {len(data['all_articles'])} items")
            return data["all_articles"]

        # Otherwise, check for top_news or top_headlines
        if "top_news" in data and isinstance(data["top_news"], list):
            print(f"[DEBUG] Found top_news with {len(data['top_news'])} items")
            return data["top_news"]

        if "top_headlines" in data and isinstance(data["top_headlines"], list):
            print(f"[DEBUG] Found top_headlines with {len(data['top_headlines'])} items")
            return data["top_headlines"]

    # fallback — recursively search for any list of dicts
    def find_list_of_dicts(obj):
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                res = find_list_of_dicts(v)
                if res:
                    return res
        return []

    result = find_list_of_dicts(data)
    if result:
        print(f"[DEBUG] Found nested list of dicts with {len(result)} items")
    else:
        print("[DEBUG] No suitable list found in payload")
    return result


def ensure_doc(d):
    """Ensure doc is a dict (wrap primitives)."""
    if isinstance(d, dict):
        return d
    return {"_value": d}

# ---------- run ----------
client = MongoClient(MONGO_URI)
db = client[DB]

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST,
    "Accept": "application/json"
}

for path, coll_name in ENDPOINTS.items():
    url = f"https://{RAPIDAPI_HOST}{path}"
    params = COMMON_PARAMS.get(path, {})  # ✅ add per-endpoint params
    print(f"\n[DEBUG] Processing {coll_name} from {path}")
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        
        # Debug output
        print(f"[DEBUG] Response status: {r.status_code}")
        print(f"[DEBUG] Response type: {type(payload)}")
        if isinstance(payload, dict):
            print(f"[DEBUG] Top-level keys: {list(payload.keys())}")
            if 'data' in payload:
                data = payload['data']
                print(f"[DEBUG] data type: {type(data)}")
                if isinstance(data, dict):
                    print(f"[DEBUG] data keys: {list(data.keys())}")
                elif isinstance(data, list):
                    print(f"[DEBUG] data is list with {len(data)} items")
        
    except Exception as e:
        print(f"[{coll_name}] fetch error: {e}")
        continue

    items = extract_items(payload)
    print(f"[DEBUG] Extracted {len(items)} items")
    
    if not items:
        print(f"[{coll_name}] WARNING: No items extracted, skipping")
        continue

    # sanitize items to dicts
    docs = [ensure_doc(d) for d in items]

    try:
        coll = db[coll_name]
        coll.delete_many({})
        print(f"[{coll_name}] old docs cleared")
        docs = docs[:50]
        coll.insert_many(docs, ordered=False)
        print(f"[{coll_name}] inserted {len(docs)} fresh documents")

    except errors.BulkWriteError as bwe:
        details = bwe.details or {}
        inserted = details.get("nInserted")
        if inserted is None:
            write_errors = details.get("writeErrors", [])
            inserted = max(0, len(docs) - len(write_errors))
        print(f"[{coll_name}] bulk write: inserted {inserted} / {len(docs)} (some duplicates/errors)")
    except Exception as e:
        print(f"[{coll_name}] insert error: {e}")

client.close()
