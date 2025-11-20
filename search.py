# backend/ingest/search.py
import os
import math
from datetime import datetime

from pymongo import MongoClient
from dotenv import load_dotenv

# Import your preprocessing helpers: adjust path if needed
from preprocess_new import preprocess_text_to_tokens

# ---------------------------------------------------------------------
# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db["final_dataset"]

# ---------------------------------------------------------------------
# Query Processing / Synonyms / Spelling

SYNONYM_MAP = {
    "us": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "pm": "prime minister",
    "pres": "president",
    "u.n": "united nations",
    "un": "united nations",
    "ai": "artificial intelligence",
}

def expand_synonyms_query(tokens):
    expanded = []
    for t in tokens:
        if t.lower() in SYNONYM_MAP:
            expanded.extend(SYNONYM_MAP[t.lower()].split())
        else:
            expanded.append(t)
    return expanded

# ---------------------------------------------------------------------
# TF / Query vector helpers

def tf_weight(tf):
    """1 + log10(tf) LNC raw weight"""
    return 1.0 + math.log10(tf) if tf > 0 else 0.0

def make_query_bigrams(tokens):
    """Return bigrams as 'w1_w2' strings"""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]

def query_to_ltn_vector(query_text, idf_map, vocab=None):
    """Convert query to unigram L * idf mapping (unnormalized). No fuzzy correction."""
    tokens = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    tokens = expand_synonyms_query(tokens)
    if not tokens:
        return {}

    tf_q = {}
    for t in tokens:
        # do NOT filter by vocab or run fuzzy correction
        tf_q[t] = tf_q.get(t, 0) + 1

    vec_q = {}
    for t, f in tf_q.items():
        if idf_map and t in idf_map:
            vec_q[t] = tf_weight(f) * idf_map[t]
        elif idf_map:
            # if term missing in idf_map, skip (unknown term)
            continue
        else:
            # if no idf_map, use lnc only
            vec_q[t] = tf_weight(f)
    return vec_q

def query_to_bigram_vector(query_text, idf_map=None, vocab=None):
    """Convert query to bigram vector. Uses idf_map when available for bigrams. No fuzzy correction."""
    tokens = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    tokens = expand_synonyms_query(tokens)
    bigrams = make_query_bigrams(tokens)
    if not bigrams:
        return {}

    tf_b = {}
    for b in bigrams:
        # do NOT fuzzy-correct or filter by vocab
        tf_b[b] = tf_b.get(b, 0) + 1

    vec_b = {}
    for b, f in tf_b.items():
        l = tf_weight(f)
        idf_b = None
        if idf_map:
            # try underscore form first, then space, then hyphen
            if b in idf_map:
                idf_b = idf_map[b]
            else:
                space = b.replace("_", " ")
                if space in idf_map:
                    idf_b = idf_map[space]
                else:
                    hy = b.replace("_", "-")
                    if hy in idf_map:
                        idf_b = idf_map[hy]
        vec_b[b] = l * (idf_b if idf_b is not None else 1.0)
    return vec_b

# ---------------------------------------------------------------------
# Index builder helper (reads/writes single meta doc in final_dataset)

def get_or_build_index_from_collection(collection, use_sampling=False, sample_size=1000, bigram_field_names=None, debug=False):
    """
    Returns (N, df_map, idf_map).
    Tries to read single meta doc with id="index_meta_".
    If not present, scans final_dataset and builds df_map from content_clean unigrams
    and any bigram fields present (keys in title_bigram_lnc/title_bigram_weights or title_bigrams array).
    """
    bigram_field_names = bigram_field_names or ["title_bigram_lnc", "title_bigram_weights"]

    # Try meta doc
    try:
        meta = collection.find_one({"id": "index_meta_"})
        if meta and "N" in meta and "df_map" in meta and "idf_map" in meta:
            if debug:
                print("Loaded index metadata from final_dataset._index_meta_")
            return meta["N"], meta["df_map"], meta["idf_map"]
    except Exception:
        pass

    df_map = {}
    N = 0

    if use_sampling:
        cursor = collection.aggregate([
            {"$sample": {"size": sample_size}},
            {"$project": {"content_clean": 1, "title_bigrams": 1, **{f: 1 for f in bigram_field_names}}}
        ])
    else:
        cursor = collection.find({}, {"content_clean": 1, "title_bigrams": 1, **{f: 1 for f in bigram_field_names}})

    for doc in cursor:
        N += 1
        content = doc.get("content_clean", "")
        if content:
            for t in set(content.split()):
                df_map[t] = df_map.get(t, 0) + 1

        # title_bigrams array (list of bigram strings) - count presence once per doc
        tb_arr = doc.get("title_bigrams")
        if isinstance(tb_arr, list):
            for b in set(tb_arr):
                df_map[b] = df_map.get(b, 0) + 1

        # bigram LNC objects (title_bigram_lnc or title_bigram_weights) whose keys are bigram strings
        for bf in bigram_field_names:
            bmap = doc.get(bf)
            if isinstance(bmap, dict):
                for b in bmap.keys():
                    df_map[b] = df_map.get(b, 0) + 1

    if N == 0:
        return 0, {}, {}

    idf_map = {}
    for term, df in df_map.items():
        # smoothed idf (avoid division by zero); N / (1 + df)
        idf_map[term] = math.log10(N / (1.0 + df)) if df > 0 else 0.0

    # try to persist meta for faster loads (best for production)
    try:
        collection.update_one(
            {"id": "index_meta_"},
            {"$set": {"N": N, "df_map": df_map, "idf_map": idf_map, "built_at": datetime.utcnow()}},
            upsert=True
        )
    except Exception:
        # ignore write errors
        pass

    return N, df_map, idf_map

# ---------------------------------------------------------------------
# Scoring: hybrid content + title-bigram

def score_docs_for_query(vec_q, vec_q_bigram=None, top_k=10, content_weight=0.6, bigram_weight=0.4):
    """
    For each document:
      - content_score_raw = sum(q_w * doc.vector[t])  (doc.vector is L2-normalized unigram vector)
      - bigram_score_raw = sum(qb_w * doc_bigram_lnc.get(b,0)) (doc bigram LNC expected unnormalized)
    Normalize each by sum of query-side weights and combine with weights (0.7,0.3).
    """
    if not vec_q and not vec_q_bigram:
        return []

    cursor = coll.find(
        {"vector": {"$exists": True}},
        {"title": 1, "url": 1, "vector": 1, "published": 1, "content_clean": 1,
         "title_bigram_lnc": 1, "title_bigram_weights": 1, "term_lnc": 1, "title_bigrams": 1}
    )

    max_q_weight = sum(vec_q.values()) if vec_q else 0.0
    max_q_bigram_weight = sum(vec_q_bigram.values()) if vec_q_bigram else 0.0

    results = []
    for doc in cursor:
        doc_vec = doc.get("vector", {})

        # content score
        content_score_raw = 0.0
        if vec_q:
            # doc_vec contains L2-normalized lnc weights; vec_q contains l*idf (unnormalized)
            content_score_raw = sum(q_w * doc_vec.get(t, 0.0) for t, q_w in vec_q.items())

        # bigram source: prefer title_bigram_lnc, then title_bigram_weights, then title_bigrams array (presence only)
        doc_bigram_field = doc.get("title_bigram_weights") 
        bigram_score_raw = 0.0
        if vec_q_bigram:
            if isinstance(doc_bigram_field, dict) and doc_bigram_field:
                # doc stores bigram -> lnc (unnormalized) -> do direct dot
                bigram_score_raw = sum(qb_w * doc_bigram_field.get(b, 0.0) for b, qb_w in vec_q_bigram.items())
            else:
                # fallback: title_bigrams array -> treat match as tf=1 (presence)
                tb_arr = doc.get("title_bigrams") or []
                tb_set = set(tb_arr) if tb_arr else set()
                if tb_set:
                    for b, qb_w in vec_q_bigram.items():
                        if b in tb_set or b.replace("_", " ") in tb_set:
                            bigram_score_raw += qb_w  # presence-based

        # Normalize by query-side totals so scores are comparable
        content_score_norm = content_score_raw / max_q_weight if max_q_weight > 0 else 0.0
        bigram_score_norm = bigram_score_raw / max_q_bigram_weight if max_q_bigram_weight > 0 else 0.0


        final_relevance = (content_weight * content_score_norm) + (bigram_weight * bigram_score_norm)

        if final_relevance > 0:
            results.append({
                "doc": doc,
                "relevance": final_relevance,
                "content_score": content_score_norm,
                "bigram_score": bigram_score_norm
            })

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]

# ---------------------------------------------------------------------
# Pruning, snippet, and search interface (mostly unchanged behavior)

def hybrid_prune_dynamic(hits, query_text, idf_map, alpha=0.25, gap_ratio_threshold=2.0, min_docs=1, debug=True):
    """
    Keep your existing overlap-based dynamic prune but operate on combined hits.
    This function currently uses unigram overlap (as before) to determine pruning.
    """
    if not hits:
        return []

    hits = sorted(hits, key=lambda x: x['relevance'], reverse=True)
    vec_q = query_to_ltn_vector(query_text, idf_map)
    if not vec_q:
        return hits[:min_docs]

    max_q_weight = sum(vec_q.values())
    kept = []
    debug_rows = []
    for h in hits:
        doc_terms = h['doc'].get("content_clean", "").split()
        overlap_weight = sum(vec_q.get(t, 0) for t in doc_terms)
        passes = overlap_weight >= alpha * max_q_weight
        if passes:
            kept.append(h)
        if debug:
            debug_rows.append({"title": h['doc'].get("title"), "overlap_weight": overlap_weight, "passes": passes})

    if debug:
        print("\n=== DYNAMIC PRUNE DEBUG ===")
        for r in debug_rows[:20]:
            print(f"  {r['title'][:50]:50s} | overlap={r['overlap_weight']:.4f} | passes={r['passes']}")

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

def get_relevant_snippet(content_clean, query_terms, snippet_length=300, window_size=50):
    if not content_clean or not query_terms:
        return content_clean[:snippet_length] if content_clean else ""
    words = content_clean.split()
    query_set = set(query_terms)
    if len(words) <= window_size:
        snippet = " ".join(words)
        return snippet[:snippet_length]
    best_score = 0
    best_start = 0
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        score = sum(1 for w in window if w in query_set)
        if score > best_score:
            best_score = score
            best_start = i
    snippet_words = words[best_start:best_start + window_size]
    snippet = " ".join(snippet_words)
    if len(snippet) > snippet_length:
        snippet = snippet[:snippet_length].rsplit(' ', 1)[0] + "..."
    return snippet

def search(query_text, k=10, prune=True, gap_ratio_threshold=3.0, alpha=0.5, debug=False,
           content_weight=0.6, bigram_weight=0.4, use_sampling_index=False):
    """
    Top-level search entrypoint. content_weight and bigram_weight default to 0.7 and 0.3.
    """
    # Build or read index from final_dataset
    N, df_map, idf_map = get_or_build_index_from_collection(
        coll,
        use_sampling=use_sampling_index,
        sample_size=1000,
        bigram_field_names=["title_bigram_lnc", "title_bigram_weights"],
        debug=debug
    )
    vocab = set(df_map.keys()) if df_map else set()

    # Query vectors (no fuzzy correction)
    vec_q = query_to_ltn_vector(query_text, idf_map, vocab=None)
    vec_q_bigram = query_to_bigram_vector(query_text, idf_map=idf_map, vocab=None)

    # token list for snippet matching
    query_tokens_raw = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    query_terms = expand_synonyms_query(query_tokens_raw)
    # NO fuzzy correction applied to query_terms

    if not vec_q and not vec_q_bigram:
        if debug:
            print("⚠ Query produced no valid terms.")
        return []

    hits = score_docs_for_query(vec_q, vec_q_bigram=vec_q_bigram, top_k=100,
                                content_weight=content_weight, bigram_weight=bigram_weight)
    if not hits:
        if debug:
            print("⚠ No matching documents found.")
        return []

    if prune:
        pruned_hits = hybrid_prune_dynamic(
            hits, query_text, idf_map,
            alpha=alpha,
            gap_ratio_threshold=gap_ratio_threshold,
            min_docs=1,
            debug=debug
        )
    else:
        pruned_hits = hits

    if k is not None:
        pruned_hits = pruned_hits[:k]

    results = []
    for h in pruned_hits:
        doc = h["doc"]
        content_clean = doc.get("content_clean", "")
        snippet = get_relevant_snippet(content_clean, query_terms, snippet_length=300, window_size=50)
        results.append({
            "title": doc.get("title", "Untitled"),
            "url": doc.get("url", ""),
            "relevance": h["relevance"],
            "snippet": snippet,
            "published": doc.get("published"),
            "content_score": h.get("content_score", 0.0),
            "bigram_score": h.get("bigram_score", 0.0)
        })
    return results

# ---------------------------------------------------------------------
# Interactive search (optional)
def interactive_search():
    print("\n" + "=" * 60)
    print("SEARCH ENGINE - Interactive Mode")
    print("=" * 60)
    print("Type your query and press Enter. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    while True:
        try:
            query = input("\n🔍 Query> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            print(f"\n⚡ Searching for: '{query}'")
            hits = search(query, k=10, prune=True, debug=False)
            if not hits:
                print("❌ No results found.\n")
                continue
            print(f"\n✓ Found {len(hits)} results:\n")
            for i, h in enumerate(hits, 1):
                print(f"{i}. [{h['relevance']:.4f}] {h['title']}")
                print(f"   🔗 {h['url']}")
                print(f"   📝 {h['snippet'][:150]}...")
                if h.get('published'):
                    print(f"   📅 {h['published']}")
                print()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    interactive_search()