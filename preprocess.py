# backend/ingest/preprocess.py
import os
import re
import csv
import math
from datetime import datetime
from urllib.parse import urlparse

from pymongo import MongoClient
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
coll = db["raw_articles"]

# ---------------------------------------------------------------------
# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ---------------------------------------------------------------------
# Text cleaning / tokenization helpers
RE_URL = re.compile(r"http\S+", flags=re.IGNORECASE)
RE_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
RE_USA = re.compile(r"\bU\.?S\.?\b", flags=re.IGNORECASE)
RE_NON_WORD = re.compile(r"[^\w\s%]")
RE_SPACES = re.compile(r"\s+")

def clean_text(text):
    if not text:
        return ""
    #text = ftfy.fix_text(text)                       # fix mojibake
    text = re.sub(r"’s\b|\'s\b", "", text)           # remove possessives
    text = text.encode("ascii", "ignore").decode()   # strip non-ascii (optional)
    text = RE_URL.sub(" ", text)
    text = RE_PCT.sub(r"\1 percent", text)
    text = RE_USA.sub("united states", text)
    text = RE_NON_WORD.sub(" ", text)
    text = RE_SPACES.sub(" ", text).strip()
    # lemma + stopword removal
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)

def preprocess_text_to_tokens(text: str, keep_numbers: bool = False, min_lemma_len: int = 1):
    text = clean_text(text)
    if not text:
        return []
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if not keep_numbers and not token.is_alpha:
            continue
        lemma = token.lemma_.lower().strip()
        if not lemma or len(lemma) < min_lemma_len:
            continue
        tokens.append(lemma)
    return tokens

def preprocess_text(text: str, **kwargs):
    return " ".join(preprocess_text_to_tokens(text, **kwargs))
    


from datetime import datetime, timezone

'''def compute_recency_score(published_date, decay_days=30):
    """
    Returns a score [0..1] for recency.
    - Newest articles ~1
    - Older than `decay_days` ~0
    """
    if not published_date:
        return 0.5  # unknown -> neutral
    if isinstance(published_date, str):
        try:
            published_date = datetime.fromisoformat(published_date)
        except:
            return 0.5

    now = datetime.now(timezone.utc)
    if published_date.tzinfo is None:
        published_date = published_date.replace(tzinfo=timezone.utc)

    delta_days = (now - published_date).days
    score = math.exp(-delta_days / decay_days)
    return max(0.0, min(1.0, score))'''

# ---------------------------------------------------------------------
# Credibility loader & scoring helpers

CRED_CSV_PATH = "data/credibility.csv"  # change path if needed

def domain_from_url(url):
    try:
        h = urlparse(url).netloc.lower()
    except:
        return ""
    if h.startswith("www."):
        h = h[4:]
    return h

def load_cred_map(path=CRED_CSV_PATH):
    """
    Load CSV with columns: domain,outlet, ... mapped_score
    Returns dict domain -> mapped_score (float 0..1)
    """
    cred = {}
    if not os.path.exists(path):
        print(f"[cred] CSV not found at {path}. Domain-level credibility will default to 0.5")
        return cred
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dom = (r.get("domain") or "").strip().lower()
            if not dom:
                continue
            try:
                mapped = float(r.get("mapped_score") or 0.0)
            except:
                mapped = 0.0
            cred[dom] = {
                "outlet": r.get("outlet", "").strip(),
                "mapped_score": max(0.0, min(1.0, mapped)),
                "source": r.get("source", "").strip(),
                "last_updated": r.get("last_updated", "")
            }
    print(f"[cred] Loaded {len(cred)} domain credibility entries from {path}")
    return cred

# load once at import
CRED_MAP = load_cred_map()

# Content quality heuristic (very lightweight)
def compute_content_score(raw_text: str):
    """
    Basic content heuristics returning score in [0,1].
    Uses:
      - caps ratio (penalize shouting)
      - doc length
      - presence of numbers/dates
    """
    if not raw_text:
        return 0.5

    text = raw_text.strip()
    words = re.findall(r"\w+", text)
    n_words = max(1, len(words))

    # uppercase words ratio (len>1)
    uppercase_words = [w for w in re.findall(r"\b\w+\b", text) if w.isupper() and len(w) > 1]
    caps_ratio = len(uppercase_words) / n_words

    # numeric tokens
    nums = re.findall(r"\b\d{1,}\b", text)
    has_numbers = len(nums) > 0

    # base
    score = 0.5

    # length heuristic
    if n_words >= 300:
        score += 0.10
    elif n_words < 50:
        score -= 0.05

    # caps penalty
    if caps_ratio > 0.05:
        score -= 0.10
    elif caps_ratio > 0.02:
        score -= 0.05

    # numeric presence is weak positive (often contains facts)
    if has_numbers:
        score += 0.05

    # clamp
    score = max(0.0, min(1.0, score))
    return score

def compute_author_score(author_name):
    """
    If you have an author reputation system, plug it here.
    For now: basic defaults:
      - if author_name is empty -> 0.5
      - else -> 0.6 (slightly above unknown)
    """
    if not author_name:
        return 0.5
    # placeholder; you can maintain an author reputation dict
    return 0.6

def combine_credibility(domain_score, content_score, author_score, w_domain=0.80, w_content=0.15, w_author=0.05):
    final = w_domain * domain_score + w_content * content_score + w_author * author_score
    return max(0.0, min(1.0, final))

# assign credibility to a single doc (reads doc fields)
def compute_doc_credibility(doc):
    # domain-level
    url = doc.get("url", "") or ""
    domain = domain_from_url(url)
    domain_entry = CRED_MAP.get(domain)
    domain_score = domain_entry["mapped_score"] if domain_entry else 0.5

    # content score: prefer raw text if available
    raw_text = doc.get("content_raw") or ""
    content_score = compute_content_score(raw_text)

    # author score if present
    author = doc.get("author") or ""
    author_score = compute_author_score(author)

    cred = combine_credibility(domain_score, content_score, author_score)
    return cred, {"domain_score": domain_score, "content_score": content_score, "author_score": author_score}

# convenience: update all docs missing credibility
def update_credibility_for_unscored(batch_size=100):
    print("[cred] Scanning for docs missing credibility...")
    cursor = coll.find({"credibility": {"$exists": False}}, {"content_raw":1, "url":1, "author":1}).limit(batch_size)
    updated = 0
    for doc in tqdm(list(cursor)):
        cred, breakdown = compute_doc_credibility(doc)
        coll.update_one({"_id": doc["_id"]}, {"$set": {"credibility": cred, "cred_breakdown": breakdown}})
        updated += 1
    print(f"[cred] Updated {updated} documents (batch).")

# ---------------------------------------------------------------------
# TF-IDF / LNC.LTN index code (unchanged except usage below)
index_coll = db["preproc_index"]

def build_df_map(min_df=1):
    print("Building DF map...")
    df = {}
    total_docs = 0

    cursor = coll.find({"content_clean": {"$exists": True, "$ne": ""}}, {"content_clean": 1})
    for doc in tqdm(cursor):
        total_docs += 1
        content = doc.get("content_clean", "")  # expected string of lemmas
        if isinstance(content, str):
            terms = content.split()
        else:
            terms = content
        unique_terms = set(terms)
        for t in unique_terms:
            df[t] = df.get(t, 0) + 1

    if min_df > 1:
        df = {t: c for t, c in df.items() if c >= min_df}

    meta = {"name": "vocab_df", "N": total_docs, "df": df, "updated_at": datetime.utcnow()}
    index_coll.replace_one({"name": "vocab_df"}, meta, upsert=True)
    print(f"Built DF for {len(df)} terms across {total_docs} documents.")
    return total_docs, df

def compute_idf_map(N, df_map):
    print("Computing IDF map...")
    idf = {}
    for t, df in df_map.items():
        if df <= 0:
            continue
        idf[t] = math.log10(N / df)
    index_coll.update_one({"name": "vocab_df"}, {"$set": {"idf": idf, "idf_updated_at": datetime.utcnow()}}, upsert=True)
    print("IDF map computed.")
    return idf

def tf_weight(tf):
    return 1.0 + math.log10(tf) if tf > 0 else 0.0

def doc_lnc_vector_from_terms(terms, vocab=None):
    tf = {}
    for t in terms:
        if vocab is not None and t not in vocab:
            continue
        tf[t] = tf.get(t, 0) + 1

    vec = {}
    for t, f in tf.items():
        if f <= 0:
            continue
        w = tf_weight(f)
        vec[t] = w

    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm > 0:
        for t in list(vec.keys()):
            vec[t] = vec[t] / norm
    return vec

def build_and_store_doc_vectors(min_df=1, force_rebuild=False):
    meta = index_coll.find_one({"name": "vocab_df"})
    if not meta or force_rebuild:
        N, df_map = build_df_map(min_df=min_df)
        idf_map = compute_idf_map(N, df_map)
    else:
        N = meta["N"]
        df_map = meta["df"]
        idf_map = meta.get("idf")
        if idf_map is None:
            idf_map = compute_idf_map(N, df_map)

    vocab = set(df_map.keys())

    print("Building and storing document vectors (lnc)...")
    cursor = coll.find({"content_clean": {"$exists": True, "$ne": ""}}, {"content_clean": 1})
    for doc in tqdm(cursor):
        content = doc.get("content_clean", "")
        terms = content.split() if isinstance(content, str) else content
        vec = doc_lnc_vector_from_terms(terms, vocab=vocab)
        coll.update_one({"_id": doc["_id"]}, {"$set": {"vector": vec}})
    index_coll.update_one({"name":"vocab_df"}, {"$set":{"index_built_at": datetime.utcnow()}}, upsert=True)
    print("Document vectors (lnc) stored.")

def query_to_ltn_vector(query_text, idf_map, vocab=None):
    tokens = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    if not tokens:
        return {}

    tf_q = {}
    for t in tokens:
        if vocab is not None and t not in vocab:
            continue
        tf_q[t] = tf_q.get(t, 0) + 1

    vec_q = {}
    for t, f in tf_q.items():
        if t not in idf_map:
            continue
        l = tf_weight(f)
        vec_q[t] = l * idf_map[t]
    return vec_q

def score_docs_for_query(vec_q, top_k=10):
    if not vec_q:
        return []

    cursor = coll.find({"vector": {"$exists": True}}, {"title":1, "url":1, "vector":1, "published":1, "credibility":1, "content_clean":1})
    results = []
    for doc in cursor:
        doc_vec = doc.get("vector", {})
        relevance = 0.0
        for t, q_w in vec_q.items():
            d_w = doc_vec.get(t)
            if d_w:
                relevance += d_w * q_w
        if relevance > 0:
            results.append({"doc": doc, "relevance": relevance})
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]


def build_index_if_needed(force=False, min_df=1):
    meta = index_coll.find_one({"name":"vocab_df"})
    if not meta or force:
        N, df_map = build_df_map(min_df=min_df)
        idf_map = compute_idf_map(N, df_map)
        build_and_store_doc_vectors(min_df=min_df, force_rebuild=True)
        return N, df_map, idf_map
    else:
        N = meta["N"]
        df_map = meta["df"]
        idf_map = meta.get("idf") or compute_idf_map(N, df_map)
        return N, df_map, idf_map

# ---------------- stricter hybrid prune + debug ----------------
def count_common_terms(query_terms, doc_terms):
    qset = set(query_terms)
    dset = set(doc_terms.split() if isinstance(doc_terms, str) else doc_terms)
    return len(qset & dset)

def hybrid_prune_dynamic(hits, query_text, idf_map, alpha=0.5, gap_ratio_threshold=2.0, min_docs=1, debug=False):
    if not hits:
        return []

    hits = sorted(hits, key=lambda x: x['relevance'], reverse=True)

    # get query vector (TF-IDF weighted)
    vec_q = query_to_ltn_vector(query_text, idf_map)
    if not vec_q:
        return hits[:min_docs]

    max_q_weight = sum(vec_q.values())

    kept = []
    debug_rows = []

    for h in hits:
        doc_terms = h['doc'].get("content_clean", "").split()
        # weighted overlap with query
        overlap_weight = sum(vec_q.get(t, 0) for t in doc_terms)

        # decide if it passes
        passes = overlap_weight >= alpha * max_q_weight

        if passes:
            kept.append(h)

        if debug:
            debug_rows.append({
                "title": h['doc'].get("title"),
                "overlap_weight": overlap_weight,
                "passes": passes
            })

    # debug print
    if debug:
        print("DYNAMIC PRUNE DEBUG:")
        for r in debug_rows[:20]:
            print(r)

    # handle gaps
    if len(kept) > 1:
        scores = [h['relevance'] for h in kept]
        ratios = [(scores[i] / scores[i+1]) if scores[i+1] != 0 else float('inf') for i in range(len(scores)-1)]
        max_ratio = max(ratios)
        if max_ratio >= gap_ratio_threshold:
            idx = ratios.index(max_ratio)
            kept = kept[:idx+1]

    if len(kept) < min_docs:
        return hits[:min_docs]

    return kept


# ------------------ updated search pipeline ------------------
def search(query_text, k=None, prune=True, gap_ratio_threshold=3.0,
           rel_weight=0.85, cred_weight=0.1, recency_weight=0.05):
    """
    Combines normalized relevance + credibility + recency for ranking.
    - k: maximum number of results (None -> return all passing documents)
    """
    N, df_map, idf_map = build_index_if_needed(force=False)
    vocab = set(df_map.keys())
    vec_q = query_to_ltn_vector(query_text, idf_map, vocab=vocab)
    if not vec_q:
        return []

    hits = score_docs_for_query(vec_q, top_k=100)  # get a larger candidate pool
    if not hits:
        return []

    max_rel = max(h['relevance'] for h in hits) if hits else 1.0

    for h in hits:
        rel_norm = h['relevance'] / max_rel if max_rel > 0 else 0.0
        doc = h['doc']

        # credibility
        cred = doc.get("credibility")
        if cred is None:
            cred, breakdown = compute_doc_credibility(doc)
            coll.update_one({"_id": doc["_id"]}, {"$set": {"credibility": cred, "cred_breakdown": breakdown}})
            h['doc'] = coll.find_one({"_id": doc["_id"]})
        
        # recency
        #recency = compute_recency_score(doc.get("published"))

        # combined score
        combined = (rel_weight * rel_norm) + (cred_weight * cred) + (0)
        h['relevance'] = combined

    hits.sort(key=lambda x: x['relevance'], reverse=True)

    if prune:
        pruned_hits = hybrid_prune_dynamic(hits, query_text, idf_map, alpha=0.5, min_docs=1)
    else:
        pruned_hits = hits

    # dynamic k: if k is None, return all pruned hits
    if k is not None:
        pruned_hits = pruned_hits[:k]

    out = []
    for h in pruned_hits:
        doc = h["doc"]
        out.append({
            "title": doc.get("title"),
            "url": doc.get("url"),
            "relevance": h["relevance"],
            "credibility": doc.get("credibility"),
            "snippet": (doc.get("content_clean") or "")[:300]
        })
    return out



# ---------------------------------------------------------------------
# Existing processing pipeline — now we call credibility update during preprocessing
def process_unsanitized(batch_size=20):
    """
    Process docs where content_raw exists but content_clean is missing.
    Also computes & stores a credibility score for each processed doc.
    """
    while True:
        docs = list(coll.find(
            {"content_raw": {"$ne": None}, "content_clean": {"$exists": False}}
        ).limit(batch_size))

        if not docs:
            print("No more docs to preprocess.")
            break

        print(f"🔹 Found batch of {len(docs)} docs")

        for doc in tqdm(docs):
            raw_text = doc.get("content_raw", "")
            clean_version = preprocess_text(raw_text)

            # compute credibility and breakdown
            cred, breakdown = compute_doc_credibility(doc)

            coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"content_clean": clean_version, "credibility": cred, "cred_breakdown": breakdown}}
            )

# ---------------- Example CLI usage ----------------
if __name__ == "__main__":
    # 1) Preprocess (fills content_clean & credibility)
    process_unsanitized(batch_size=10)

    # 2) Build index (df/idf + doc vectors)
    build_index_if_needed(force=True, min_df=1)

    # 3) Query loop
    while True:
        q = input("query> ").strip()
        if not q:
            break
        hits = search(q, k=10)
        for i, h in enumerate(hits, 1):
            print(f"{i}. ({h['relevance']:.6f}) [cred={h.get('credibility'):.2f}] {h['title']} -- {h['url']}")
            print("    ", h["snippet"][:200], "...")
