import requests
from datetime import datetime, timedelta
from pymongo import MongoClient, UpdateOne
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
API_KEY = "abc3148664614edb84278e0896d8f257"  # replace with your API key
MONGO_URI = "mongodb+srv://manyagoel2014_db_user:qKm4963Q6Lg4a5vd@cluster0.ete9n0t.mongodb.net/"
DB_NAME = "News"
COLLECTION_NAME = "articles"

DOMAINS = [
    "timesofindia.indiatimes.com",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "deccanherald.com",
    "livemint.com",
    "business-standard.com",
    "thewire.in",
    "telegraphindia.com",
    "thequint.com",
    "dnaindia.com",
    "newindianexpress.com",
    "asianage.com",
    "punemirror.indiatimes.com",
    "bangaloremirror.indiatimes.com",
    "mid-day.com",
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "financialexpress.com",
    "business-standard.com",
    "gadgets.ndtv.com",
    "firstpost.com/tech",
    "yourstory.com",
    "tech2.in.com",
    "theprint.in",
    "scroll.in",
    "outlookindia.com",
    "indiatoday.in",
    "dnaindia.com/political",
    "newslaundry.com",
    "news18.com",
    "abplive.com",
    "zeenews.india.com",
    "republicworld.com",
    "mynation.com",
    "freepressjournal.in",
    "dailyhunt.in/news",
    "latestly.com",
    "india.com",
    "oneindia.com"
]

# Convert to a single comma-separated string
domains_string = ", ".join(DOMAINS)
print(domains_string)


PAGE_SIZE = 100  # max articles per request (depends on API)
MAX_PAGES = 5    # adjust depending on API limits
LANGUAGE = "en"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def fetch_articles(from_date, to_date, domains, page=1):
    """Fetch articles from NewsAPI (or any similar API)"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "from": from_date,
        "to": to_date,
        "domains": ",".join(domains),
        "language": LANGUAGE,
        "pageSize": PAGE_SIZE,
        "page": page
    }
    headers = {"Authorization": API_KEY}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        print(f"Error fetching page {page}: {response.status_code}")
        return []

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # compute last 30 days
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=30)
    from_date = thirty_days_ago.strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    all_articles = []
    for page in range(1, MAX_PAGES + 1):
        articles = fetch_articles(from_date, to_date, DOMAINS, page)
        if not articles:
            break
        all_articles.extend(articles)
        time.sleep(1)  # avoid hitting API rate limits

    print(f"Fetched {len(all_articles)} articles")

    # Prepare bulk operations for MongoDB
    operations = []
    for article in all_articles:
        operations.append(UpdateOne(
            {"url": article.get("url")},
            {"$set": {
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name"),
                "publishedAt": article.get("publishedAt"),
                "description": article.get("description"),
                "full_text": None,  # placeholder for later scraping
                "scraped": False
            }},
            upsert=True
        ))

    if operations:
        result = collection.bulk_write(operations)
        print(f"Inserted/Updated {result.upserted_count + result.modified_count} articles")

    print("Corpus ready in MongoDB. Full-text scraping can be done next.")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()