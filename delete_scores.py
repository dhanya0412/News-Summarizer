from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["raw_articles"]

# Remove content_clean from all docs
result = coll.update_many({}, {"$unset": {"content_clean": ""}})
print(f"Removed 'content_clean' from {result.modified_count} documents.")
res = coll.update_many({"vector": {"$exists": True}}, {"$unset": {"vector": ""}})
print(f"Removed 'vector' from {res.modified_count} documents.")
cred = coll.update_many({"cred_breakdown": {"$exists": True}}, {"$unset": {"cred_breakdown": ""}})
print(f"Removed 'cred_breakdown' from {cred.modified_count} documents.")
credibility = coll.update_many({"credibility": {"$exists": True}}, {"$unset": {"credibility": ""}})
print(f"Removed 'credility' from {credibility.modified_count} documents.")