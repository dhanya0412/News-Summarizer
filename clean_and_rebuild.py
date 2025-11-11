# backend/ingest/cleanup_and_rebuild.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import subprocess

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["raw_articles"]

coll.delete_many({})
print("Deleted all raw_articles.")

subprocess.run(["python", "gdelt.py"])
subprocess.run(["python", "scraper.py"])
