# File: backend/ingest/preprocess_bigrams.py
# Local path: backend/ingest/preprocess_bigrams.py
"""
Document preprocessing pipeline (LNC-only).
Everything is stored in final_dataset (no external vocab_terms / preproc_index).
- content_clean: spaCy-lemmatized unigram tokens (space-joined) stored per-document.
- vector: LNC (1 + log10(tf), L2-normalized) stored per-document for content terms.
- term_lnc: per-doc raw lnc values (optional; stored only for terms present in the doc).
- title_bigrams: adjacent lemma bigrams stored per-document when requested.
- title_bigram_weights: LNC weights for title bigrams stored per-document when requested.
"""
import os
import re
import math
from datetime import datetime, timezone
from collections import Counter

from pymongo import MongoClient
import pymongo
from dotenv import load_dotenv
import spacy
import ftfy
from tqdm import tqdm

# ---------------------------------------------------------------------
# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["final_dataset"]

# ---------------------------------------------------------------------
# Load spaCy English model (disable unused components for speed)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ---------------------------------------------------------------------
# Compiled regex patterns for efficiency
RE_URL = re.compile(r"http\S+", flags=re.IGNORECASE)
RE_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
RE_USA = re.compile(r"\bU\.?S\.?\b", flags=re.IGNORECASE)
RE_NON_WORD = re.compile(r"[^\w\s%]")
RE_SPACES = re.compile(r"\s+")
RE_POSSESSIVE = re.compile(r"â€™s\b|\'s\b")

# ---------------------------------------------------------------------
# Named Entity Recognition for Common Phrases
COMMON_PHRASES = {
    'united states', 'new york', 'new delhi', 'saudi arabia',
    'world cup', 'supreme court', 'high court', 'lok sabha',
    'rajya sabha', 'prime minister', 'chief minister',
    'european union', 'united kingdom', 'united nations',
    'white house', 'red fort', 'india gate'
}

# ---------------------------------------------------------------------
# Text cleaning and tokenization

def clean_text(text):
    """Clean and normalize text with encoding fixes and standardization."""
    if not text:
        return ""

    text = ftfy.fix_text(text)

    # Remove publish/update metadata lines
    text = re.sub(
        r"^(updated|published)[^a-zA-Z0-9]+.*?\n",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Remove city + date headers
    text = re.sub(
        r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[^\n]*\n",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Remove timezone words
    text = re.sub(r"\b(ist|gmt|am|pm)\b", " ", text, flags=re.IGNORECASE)

    # Original cleaning rules
    text = RE_POSSESSIVE.sub("", text)
    text = RE_URL.sub(" ", text)
    text = RE_PCT.sub(r"\1 percent", text)
    text = RE_USA.sub("united states", text)
    text = RE_NON_WORD.sub(" ", text)
    text = RE_SPACES.sub(" ", text).strip()

    return text


def preprocess_text_to_tokens(text: str, keep_numbers: bool = True,
                               min_lemma_len: int = 1, include_bigrams: bool = False):
    """
    Tokenize and lemmatize text, removing stopwords.
    Optionally append adjacent bigrams to the returned list (internal helper).
    """
    text = clean_text(text)
    if not text:
        return []

    protected_text = text.lower()
    phrase_map = {}
    for phrase in COMMON_PHRASES:
        if phrase in protected_text:
            placeholder = phrase.replace(' ', '_')
            protected_text = protected_text.replace(phrase, placeholder)
            phrase_map[placeholder] = phrase

    doc = nlp(protected_text)
    tokens = []

    for token in doc:
        lemma = token.lemma_.lower().strip()

        if not lemma:
            continue

        # protected phrase placeholder
        if lemma in phrase_map:
            tokens.append(lemma)
            continue

        if token.is_stop:
            continue
        if not keep_numbers and not token.is_alpha:
            continue
        if len(lemma) < min_lemma_len:
            continue

        tokens.append(lemma)

    if include_bigrams and len(tokens) >= 2:
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        tokens.extend(bigrams)

    return tokens


def preprocess_text(text: str, include_bigrams: bool = False, **kwargs):
    """Convert text to space-separated string of processed tokens (unigrams by default)."""
    return " ".join(preprocess_text_to_tokens(text, include_bigrams=include_bigrams, **kwargs))


# ---------------------------------------------------------------------
# Title bigram PMI filtering (optional) - computes significant bigrams across titles
# This only filters per-document stored title_bigrams; it does not write to any other collection.

def compute_bigram_scores(min_count=5, min_pmi=3.0):
    """
    Compute PMI for title bigrams across titles and filter stored per-document title_bigrams.
    No external collections are used.
    """
    print("Computing significant title bigrams (PMI)...")

    unigram_counts = Counter()
    bigram_counts = Counter()
    total_tokens = 0

    cursor = coll.find({"title": {"$exists": True, "$ne": ""}}, {"title": 1})
    for doc in tqdm(cursor, desc="Scanning titles for PMI"):
        title = doc.get("title", "") or ""
        tokens = preprocess_text_to_tokens(title, include_bigrams=False)
        total_tokens += len(tokens)
        for t in tokens:
            unigram_counts[t] += 1
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i+1])] += 1

    significant_bigrams = set()
    if total_tokens == 0:
        print("No title tokens found; skipping PMI.")
        return significant_bigrams

    for (w1, w2), count in bigram_counts.items():
        if count < min_count:
            continue
        p_bigram = count / total_tokens
        p_w1 = unigram_counts[w1] / total_tokens
        p_w2 = unigram_counts[w2] / total_tokens
        if p_w1 > 0 and p_w2 > 0 and p_bigram > 0:
            pmi = math.log2(p_bigram / (p_w1 * p_w2))
            if pmi >= min_pmi:
                significant_bigrams.add(f"{w1}_{w2}")

    print(f"Found {len(significant_bigrams)} significant title bigrams (PMI >= {min_pmi})")

    # Filter per-document stored title_bigrams (if any)
    if significant_bigrams:
        cursor2 = coll.find({"title_bigrams": {"$exists": True}}, {"title_bigrams": 1})
        bulk_ops = []
        for doc in tqdm(cursor2, desc="Filtering per-doc title_bigrams"):
            tb = doc.get("title_bigrams", []) or []
            filtered = [b for b in tb if b in significant_bigrams]
            if filtered:
                bulk_ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$set": {"title_bigrams": filtered}}))
            else:
                bulk_ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$unset": {"title_bigrams": ""}}))

            if len(bulk_ops) >= 500:
                coll.bulk_write(bulk_ops, ordered=False)
                bulk_ops = []
        if bulk_ops:
            coll.bulk_write(bulk_ops, ordered=False)

    print("Title bigram filtering complete (stored in final_dataset only).")
    return significant_bigrams


# ---------------------------------------------------------------------
# LNC builders: content and title bigrams

def tf_weight_raw(tf):
    """Raw L value: 1 + log10(tf) for tf>0, else 0"""
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


def build_and_store_doc_vectors_lnc():
    """
    Compute per-document LNC (1 + log10(tf), then L2-normalize) for content terms.
    Stores per-document:
      - vector: {term: normalized_lnc}
      - term_lnc: {term: raw_lnc}    (only for terms in that document)
    """
    print("Building and storing per-document LNC vectors (content)...")
    cursor = coll.find({"content_clean": {"$exists": True, "$ne": ""}}, {"content_clean": 1})
    bulk_ops = []
    for doc in tqdm(cursor, desc="Computing LNC vectors"):
        content = doc.get("content_clean", "") or ""
        if not content:
            bulk_ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$unset": {"vector": "", "term_lnc": ""}}))
            if len(bulk_ops) >= 500:
                coll.bulk_write(bulk_ops, ordered=False)
                bulk_ops = []
            continue

        terms = content.split()
        tf = Counter(terms)

        # raw lnc values
        lnc_raw = {t: tf_weight_raw(f) for t, f in tf.items()}

        # L2 normalization
        norm = math.sqrt(sum(v * v for v in lnc_raw.values()))
        if norm > 0:
            vec = {t: (v / norm) for t, v in lnc_raw.items()}
        else:
            vec = {}

        update = {"vector": vec, "term_lnc": lnc_raw}
        bulk_ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$set": update}))

        if len(bulk_ops) >= 500:
            coll.bulk_write(bulk_ops, ordered=False)
            bulk_ops = []

    if bulk_ops:
        coll.bulk_write(bulk_ops, ordered=False)
    print("Content LNC vectors stored per-document.")


def build_and_store_title_bigram_lnc(include_bigrams=False):
    """
    For each document, compute adjacent title bigrams (from spaCy-lemmatized title tokens)
    and store LNC weights for those bigrams.

    Stores per-document:
      - title_bigrams: [bigrams]            (if include_bigrams True)
      - title_bigram_weights: {bigram: normalized_lnc}
      - title_bigram_lnc: {bigram: raw_lnc}

    If include_bigrams is False, removes any existing title_bigram fields.
    """
    if not include_bigrams:
        # remove any existing fields safely
        cursor = coll.find({"$or": [{"title_bigrams": {"$exists": True}}, {"title_bigram_weights": {"$exists": True}}, {"title_bigram_lnc": {"$exists": True}}]}, {"_id": 1})
        bulk = []
        for doc in tqdm(cursor, desc="Removing title bigram fields"):
            bulk.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$unset": {"title_bigrams": "", "title_bigram_weights": "", "title_bigram_lnc": ""}}))
            if len(bulk) >= 500:
                coll.bulk_write(bulk, ordered=False)
                bulk = []
        if bulk:
            coll.bulk_write(bulk, ordered=False)
        print("Removed title bigram fields where present.")
        return

    print("Computing & storing per-document title bigram LNC weights...")
    cursor = coll.find({}, {"title": 1})
    bulk_ops = []
    for doc in tqdm(cursor, desc="Title bigram LNC"):
        title = doc.get("title", "") or ""
        tokens = preprocess_text_to_tokens(title, include_bigrams=False)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)] if len(tokens) >= 2 else []
        if not bigrams:
            # remove fields if present
            bulk_ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$unset": {"title_bigrams": "", "title_bigram_weights": "", "title_bigram_lnc": ""}}))
            if len(bulk_ops) >= 500:
                coll.bulk_write(bulk_ops, ordered=False)
                bulk_ops = []
            continue

        tf = Counter(bigrams)
        lnc_raw = {b: tf_weight_raw(f) for b, f in tf.items()}
        norm = math.sqrt(sum(v * v for v in lnc_raw.values()))
        if norm > 0:
            vec = {b: (v / norm) for b, v in lnc_raw.items()}
        else:
            vec = {}

        update = {"title_bigrams": bigrams, "title_bigram_weights": vec, "title_bigram_lnc": lnc_raw}
        bulk_ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$set": update}))

        if len(bulk_ops) >= 500:
            coll.bulk_write(bulk_ops, ordered=False)
            bulk_ops = []

    if bulk_ops:
        coll.bulk_write(bulk_ops, ordered=False)
    print("Per-document title bigram LNC weights stored.")


# ---------------------------------------------------------------------
# Main preprocessing loop

def process_unsanitized(batch_size=50, include_bigrams=False):
    """
    - content_clean: cleaned + tokenized + lemmatized unigram string (stored per-document).
    - title_bigrams: adjacent bigrams stored per-document only if include_bigrams True.
    """
    processed_count = 0
    while True:
        docs = list(coll.find({"content": {"$ne": None}, "content_clean": {"$exists": False}}).limit(batch_size))
        if not docs:
            print("✓ No more documents to preprocess.")
            break

        print(f"📝 Processing batch of {len(docs)} documents...")
        ops = []
        for doc in docs:
            raw = doc.get("content", "") or ""
            tokens = preprocess_text_to_tokens(raw, keep_numbers=False, min_lemma_len=1, include_bigrams=False)
            content_clean = " ".join(tokens)
            update = {"content_clean": content_clean}

            if include_bigrams:
                title = doc.get("title", "") or ""
                title_tokens = preprocess_text_to_tokens(title, keep_numbers=False, min_lemma_len=1, include_bigrams=False)
                if len(title_tokens) >= 2:
                    title_bigrams = [f"{title_tokens[i]}_{title_tokens[i+1]}" for i in range(len(title_tokens)-1)]
                    update["title_bigrams"] = title_bigrams

            ops.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$set": update}))
            if len(ops) >= 500:
                coll.bulk_write(ops, ordered=False)
                ops = []
            processed_count += 1

        if ops:
            coll.bulk_write(ops, ordered=False)

    print(f"✓ Total processed: {processed_count}")


# ---------------------------------------------------------------------
# Entrypoint orchestration (LNC-only; no flags added for LNC)

if __name__ == "__main__":
    import sys
    use_bigrams = '--bigrams' in sys.argv or '-b' in sys.argv

    print("=" * 60)
    print("DOCUMENT PREPROCESSING PIPELINE — LNC ONLY")
    print("All outputs are stored in final_dataset only.")
    print("=" * 60)

    print("\n[1/3] Preprocessing raw documents (content_clean and optional raw title_bigrams)...")
    process_unsanitized(batch_size=50, include_bigrams=use_bigrams)

    print("\n[2/3] Building & storing per-document content LNC vectors (vector, term_lnc)...")
    build_and_store_doc_vectors_lnc()

    if use_bigrams:
        print("\n[3/3] Building & storing per-document title bigram LNC weights...")
        # Optionally you can run compute_bigram_scores(...) before this to filter bigrams globally.
        build_and_store_title_bigram_lnc(include_bigrams=True)
    else:
        print("\n[3/3] Skipping title bigram weight computation (use --bigrams to enable).")
        build_and_store_title_bigram_lnc(include_bigrams=False)

    print("\nPREPROCESSING COMPLETE")
    print("=" * 60)