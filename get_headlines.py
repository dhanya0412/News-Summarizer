import os, requests, random
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

TIMEOUT = 15
HEADLINE_LIMIT = 10

# ---------- Helper Functions ----------
def extract_items(payload):
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

url = f"https://{RAPIDAPI_HOST}/top-headlines"
headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST,
    "Accept": "application/json"
}

all_items = []
try:
    page = 1
    country_articles = 0
    while country_articles < 5:
        params = {
            "country": "IN",
            "lang": "en",
            "limit": 10,
            "page": page
        }

        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            new_items = extract_items(payload)
        except Exception as inner_e:
            break

        if not new_items:
            break

        for item in new_items:
            item["country"] = "IN"

        all_items.extend(new_items)
        country_articles += len(new_items)
        if len(new_items) < 10:
            break
        page += 1
        if len(all_items) >= HEADLINE_LIMIT:
            break
except Exception as e:
    exit(1)

items = all_items[:HEADLINE_LIMIT]
if not items:
    exit(0)
random.shuffle(items)
sample = items[:5]
print("----- HEADLINES -----")
for idx, doc in enumerate(sample, start=1):
    title = doc.get("title") or doc.get("headline") or "[No title]"
    link = doc.get("link") or doc.get("url") or "[No link]"
    date = doc.get("published_datetime_utc") or "[No date]"
    country = doc.get("country", "N/A")
    data=doc.get("snippet", "N/A")
    print(f"{idx}. {title}\n   {data}...\n   Link: {link}\n   Date: {date[:10]}\n   Time: {date[11:19]}")