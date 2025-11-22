import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import re
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import random

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB = os.getenv("MONGO_DB")
GROQ_KEY = os.getenv("GROQ_API_KEY")

client = MongoClient(MONGO_URI)
db = client[DB]

final_coll = db["final_dataset"]
fake_coll = db["fake_news_dataset"]

data = list(fake_coll.find({}, {"real_title": 1, "fake_title": 1, "_id": 0}))

random.shuffle(data)

real = data[0]["real_title"]
fake1 = data[1]["fake_title"]
fake2 = data[2]["fake_title"]

answer_map = {
    real: "real",
    fake1: "fake",
    fake2: "fake"
}


options = [real, fake1, fake2]
random.shuffle(options)


print("Guess the REAL headline:\n")
for i, opt in enumerate(options, start=1):
    opt=opt.split("|")[0].strip()
    print(f"{i}. {opt}")

choice = int(input("\nEnter your choice (1-3): "))

selected = options[choice - 1]

if answer_map[selected] == "real":
    print("\nYOU'RE CORRECT!\n")
else:
    print("\nOH NO! THAT WAS FAKE.\n")

print("Correct answer was:", real)

print("\nURL of the news:\n")
for i in range(3):
    doc = final_coll.find_one({"title": data[i]["real_title"]}, {"url": 1, "_id": 0})
    if doc and "url" in doc:
        print("TITLE:",data[i]["real_title"])
        print(doc["url"])
    else:
        print("Could not find URL of the real headline in final_dataset.")
    print("\n")
