# backend/ingest/preprocess.py
import os
import re
import math
from datetime import datetime

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
coll = db["final_dataset"]
index_coll = db["preproc_index"]

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
# Text cleaning and tokenization

def clean_text(text):
    if not text:
        return ""

    text = ftfy.fix_text(text)

    # ------------------------------------------------
    # REMOVE PUBLISH/UPDATE METADATA LINES
    # ------------------------------------------------
    text = re.sub(
        r"^(updated|published)[^a-zA-Z0-9]+.*?\n",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # ------------------------------------------------
    # REMOVE CITY + DATE HEADERS ("New Delhi, Nov 5 —")
    # ------------------------------------------------
    text = re.sub(
        r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[^\n]*\n",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Remove timezone words (pm, am, IST)
    text = re.sub(r"\b(ist|gmt|am|pm)\b", " ", text, flags=re.IGNORECASE)

    # ------------------------------------------------
    # ORIGINAL CLEANING RULES
    # ------------------------------------------------
    text = RE_POSSESSIVE.sub("", text)
    text = RE_URL.sub(" ", text)
    text = RE_PCT.sub(r"\1 percent", text)
    text = RE_USA.sub("united states", text)
    text = RE_NON_WORD.sub(" ", text)
    text = RE_SPACES.sub(" ", text).strip()

    return text


def preprocess_text_to_tokens(text: str, keep_numbers: bool = False, min_lemma_len: int = 1):
    """
    Tokenize and lemmatize text, removing stopwords.
    
    Args:
        text: Input text string
        keep_numbers: Whether to keep numeric tokens
        min_lemma_len: Minimum length for lemmas to keep
    
    Returns:
        List of processed tokens
    """
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
        if lemma and len(lemma) >= min_lemma_len:
            tokens.append(lemma)
    
    return tokens


def preprocess_text(text: str, **kwargs):
    """Convert text to space-separated string of processed tokens."""
    return " ".join(preprocess_text_to_tokens(text, **kwargs))


# ---------------------------------------------------------------------
# TF-IDF Index Building

def build_df_map(min_df=1):
    """
    Build document frequency map for all terms in corpus.
    
    Args:
        min_df: Minimum document frequency threshold
    
    Returns:
        Tuple of (total_docs, df_map)
    """
    print("Building DF map...")
    df = {}
    total_docs = 0

    cursor = coll.find(
        {"content_clean": {"$exists": True, "$ne": ""}}, 
        {"content_clean": 1}
    )
    
    for doc in tqdm(cursor):
        total_docs += 1
        content = doc.get("content_clean", "")
        
        if isinstance(content, str):
            terms = content.split()
        else:
            terms = content
        
        # Count unique terms only
        unique_terms = set(terms)
        for t in unique_terms:
            df[t] = df.get(t, 0) + 1

    # Filter by minimum document frequency
    if min_df > 1:
        df = {t: c for t, c in df.items() if c >= min_df}

    # Store in database
    meta = {
        "name": "vocab_df",
        "N": total_docs,
        "df": df,
        "updated_at": datetime.utcnow()
    }
    index_coll.replace_one({"name": "vocab_df"}, meta, upsert=True)
    
    print(f"Built DF for {len(df)} terms across {total_docs} documents.")
    return total_docs, df


def compute_idf_map(N, df_map):
    """
    Compute IDF (Inverse Document Frequency) for all terms.
    
    Args:
        N: Total number of documents
        df_map: Document frequency map
    
    Returns:
        IDF map dictionary
    """
    print("Computing IDF map...")
    idf = {}
    
    for t, df in df_map.items():
        if df > 0:
            idf[t] = math.log10(N / df)
    
    # Update database with IDF values
    index_coll.update_one(
        {"name": "vocab_df"},
        {"$set": {"idf": idf, "idf_updated_at": datetime.utcnow()}},
        upsert=True
    )
    
    print("IDF map computed.")
    return idf


def tf_weight(tf):
    """Compute logarithmic TF weight."""
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


def doc_lnc_vector_from_terms(terms, vocab=None):
    """
    Build LNC (log-normalized-cosine) document vector.
    
    Args:
        terms: List of document terms
        vocab: Optional vocabulary set to filter terms
    
    Returns:
        Dictionary mapping terms to normalized weights
    """
    # Compute term frequencies
    tf = {}
    for t in terms:
        if vocab is not None and t not in vocab:
            continue
        tf[t] = tf.get(t, 0) + 1

    # Apply log weighting
    vec = {}
    for t, f in tf.items():
        if f > 0:
            vec[t] = tf_weight(f)

    # Cosine normalization
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm > 0:
        for t in vec:
            vec[t] = vec[t] / norm
    
    return vec


def build_and_store_doc_vectors(min_df=1, force_rebuild=False):
    """
    Build and store document vectors for all articles.
    
    Args:
        min_df: Minimum document frequency
        force_rebuild: Force rebuilding DF/IDF maps
    """
    # Load or build DF/IDF maps
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

    print("Building and storing document vectors (LNC)...")
    cursor = coll.find(
        {"content_clean": {"$exists": True, "$ne": ""}},
        {"content_clean": 1}
    )
    
    for doc in tqdm(cursor):
        content = doc.get("content_clean", "")
        terms = content.split() if isinstance(content, str) else content
        vec = doc_lnc_vector_from_terms(terms, vocab=vocab)
        coll.update_one({"_id": doc["_id"]}, {"$set": {"vector": vec}})
    
    # Mark index as built
    index_coll.update_one(
        {"name": "vocab_df"},
        {"$set": {"index_built_at": datetime.utcnow()}},
        upsert=True
    )
    
    print("Document vectors (LNC) stored.")


def build_index_if_needed(force=False, min_df=1):
    """
    Build index if it doesn't exist or if forced.
    
    Args:
        force: Force rebuild even if index exists
        min_df: Minimum document frequency
    
    Returns:
        Tuple of (N, df_map, idf_map)
    """
    meta = index_coll.find_one({"name": "vocab_df"})
    
    if not meta or force:
        N, df_map = build_df_map(min_df=min_df)
        idf_map = compute_idf_map(N, df_map)
        build_and_store_doc_vectors(min_df=min_df, force_rebuild=True)
        return N, df_map, idf_map
    else:
        N = meta["N"]
        df_map = meta["df"]
        idf_map = meta.get("idf")
        if idf_map is None:
            idf_map = compute_idf_map(N, df_map)
        return N, df_map, idf_map


# ---------------------------------------------------------------------
# Document Processing Pipeline

def process_unsanitized(batch_size=50):
    """
    Process documents where content exists but content_clean is missing.
    
    Args:
        batch_size: Number of documents to process per batch
    """
    processed_count = 0
    
    while True:
        docs = list(coll.find(
            {"content": {"$ne": None}, "content_clean": {"$exists": False}}
        ).limit(batch_size))

        if not docs:
            print("✓ No more documents to preprocess.")
            break

        print(f"📝 Processing batch of {len(docs)} documents...")

        for doc in tqdm(docs):
            raw_text = doc.get("content", "")
            clean_version = preprocess_text(raw_text)

            coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"content_clean": clean_version}}
            )
            processed_count += 1

    print(f"✓ Total processed: {processed_count} documents")


# ---------------------------------------------------------------------
# Main execution

if __name__ == "__main__":
    print("=" * 60)
    print("DOCUMENT PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Preprocess documents
    print("\n[1/2] Preprocessing documents...")
    process_unsanitized(batch_size=50)
    
    # Step 2: Build search index
    print("\n[2/2] Building search index...")
    build_index_if_needed(force=True, min_df=1)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)