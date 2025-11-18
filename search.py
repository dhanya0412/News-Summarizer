# backend/ingest/search.py
import os
import math

from pymongo import MongoClient
from dotenv import load_dotenv
from preprocess_new import preprocess_text_to_tokens, build_index_if_needed

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
# Query Processing

def tf_weight(tf):
    """Compute logarithmic TF weight."""
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


def query_to_ltn_vector(query_text, idf_map, vocab=None):
    """
    Convert query to LTN (log-TF-IDF-normalized) vector.
    
    Args:
        query_text: Query string
        idf_map: IDF values for vocabulary
        vocab: Optional vocabulary set to filter terms
    
    Returns:
        Query vector dictionary
    """
    tokens = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    if not tokens:
        return {}

    # Compute term frequencies in query
    tf_q = {}
    for t in tokens:
        if vocab is not None and t not in vocab:
            continue
        tf_q[t] = tf_q.get(t, 0) + 1

    # Apply log-TF and IDF weighting
    vec_q = {}
    for t, f in tf_q.items():
        if t in idf_map:
            l = tf_weight(f)
            vec_q[t] = l * idf_map[t]
    
    return vec_q


def score_docs_for_query(vec_q, top_k=100):
    """
    Score all documents against query vector using cosine similarity.
    
    Args:
        vec_q: Query vector
        top_k: Number of top results to return
    
    Returns:
        List of dictionaries with 'doc' and 'relevance' keys
    """
    if not vec_q:
        return []

    cursor = coll.find(
        {"vector": {"$exists": True}},
        {"title": 1, "url": 1, "vector": 1, "published": 1, "content_clean": 1, "description": 1}
    )
    
    results = []
    for doc in cursor:
        doc_vec = doc.get("vector", {})
        
        # Compute cosine similarity
        relevance = sum(q_w * doc_vec.get(t, 0) for t, q_w in vec_q.items())
        
        if relevance > 0:
            results.append({"doc": doc, "relevance": relevance})
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]


def hybrid_prune_dynamic(hits, query_text, idf_map, alpha=0.5, gap_ratio_threshold=2.0, min_docs=1, debug=False):
    """
    Prune search results based on overlap with query terms and score gaps.
    
    Args:
        hits: List of search results
        query_text: Original query string
        idf_map: IDF values
        alpha: Minimum overlap threshold (0-1)
        gap_ratio_threshold: Maximum ratio between consecutive scores
        min_docs: Minimum number of documents to return
        debug: Print debug information
    
    Returns:
        Pruned list of hits
    """
    if not hits:
        return []

    hits = sorted(hits, key=lambda x: x['relevance'], reverse=True)

    # Get query vector with IDF weights
    vec_q = query_to_ltn_vector(query_text, idf_map)
    if not vec_q:
        return hits[:min_docs]

    max_q_weight = sum(vec_q.values())

    kept = []
    debug_rows = []

    # Filter by weighted overlap
    for h in hits:
        doc_terms = h['doc'].get("content_clean", "").split()
        
        # Calculate weighted overlap with query
        overlap_weight = sum(vec_q.get(t, 0) for t in doc_terms)
        
        # Check if document passes threshold
        passes = overlap_weight >= alpha * max_q_weight

        if passes:
            kept.append(h)

        if debug:
            debug_rows.append({
                "title": h['doc'].get("title"),
                "overlap_weight": overlap_weight,
                "passes": passes
            })

    # Debug output
    if debug:
        print("\n=== DYNAMIC PRUNE DEBUG ===")
        for r in debug_rows[:20]:
            print(f"  {r['title'][:50]:50s} | overlap={r['overlap_weight']:.4f} | passes={r['passes']}")

    # Check for large gaps in relevance scores
    if len(kept) > 1:
        scores = [h['relevance'] for h in kept]
        ratios = [
            (scores[i] / scores[i+1]) if scores[i+1] != 0 else float('inf')
            for i in range(len(scores)-1)
        ]
        max_ratio = max(ratios)
        
        # Cut off after large gap
        if max_ratio >= gap_ratio_threshold:
            idx = ratios.index(max_ratio)
            kept = kept[:idx+1]

    # Ensure minimum number of results
    if len(kept) < min_docs:
        return hits[:min_docs]

    return kept


# ---------------------------------------------------------------------
# Search Interface

def get_relevant_snippet(content_clean, query_terms, snippet_length=300, window_size=50):
    """
    Extract snippet containing the most query terms.
    
    Args:
        content_clean: Preprocessed document content
        query_terms: List of query terms to look for
        snippet_length: Maximum character length of snippet
        window_size: Size of sliding window in words
    
    Returns:
        String snippet containing relevant content
    """
    if not content_clean or not query_terms:
        return content_clean[:snippet_length] if content_clean else ""
    
    words = content_clean.split()
    query_set = set(query_terms)
    
    # Handle short documents
    if len(words) <= window_size:
        snippet = " ".join(words)
        return snippet[:snippet_length]
    
    # Find window with most query term matches
    best_score = 0
    best_start = 0
    
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        score = sum(1 for w in window if w in query_set)
        
        if score > best_score:
            best_score = score
            best_start = i
    
    # Extract snippet from best window
    snippet_words = words[best_start:best_start + window_size]
    snippet = " ".join(snippet_words)
    
    # Truncate to character limit
    if len(snippet) > snippet_length:
        snippet = snippet[:snippet_length].rsplit(' ', 1)[0] + "..."
    
    return snippet


def search(query_text, k=10, prune=True, gap_ratio_threshold=3.0, alpha=0.5, debug=False):
    """
    Search for documents matching the query.
    
    Args:
        query_text: Search query string
        k: Maximum number of results (None = return all passing documents)
        prune: Whether to apply dynamic pruning
        gap_ratio_threshold: Threshold for detecting score gaps
        alpha: Minimum overlap threshold for pruning
        debug: Print debug information
    
    Returns:
        List of result dictionaries with title, url, relevance, and snippet
    """
    # Ensure index is built
    N, df_map, idf_map = build_index_if_needed(force=False)
    vocab = set(df_map.keys())
    
    # Convert query to vector and get query terms
    vec_q = query_to_ltn_vector(query_text, idf_map, vocab=vocab)
    query_terms = preprocess_text_to_tokens(query_text, keep_numbers=False, min_lemma_len=1)
    
    if not vec_q:
        print("⚠ Query produced no valid terms.")
        return []

    # Score all documents
    hits = score_docs_for_query(vec_q, top_k=100)
    if not hits:
        print("⚠ No matching documents found.")
        return []

    # Apply pruning if enabled
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

    # Limit to k results if specified
    if k is not None:
        pruned_hits = pruned_hits[:k]

    # Format output
    results = []
    for h in pruned_hits:
        doc = h["doc"]
        
        # Use content_clean for snippet with context-aware extraction
        content_clean = doc.get("content_clean", "")
        snippet = get_relevant_snippet(content_clean, query_terms, snippet_length=300, window_size=50)
        
        results.append({
            "title": doc.get("title", "Untitled"),
            "url": doc.get("url", ""),
            "relevance": h["relevance"],
            "snippet": snippet,
            "published": doc.get("published")
        })
    
    return results


# ---------------------------------------------------------------------
# Interactive Query Interface

def interactive_search():
    """Run interactive search loop."""
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
            print("-" * 60)
            
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


# ---------------------------------------------------------------------
# Main execution

if __name__ == "__main__":
    interactive_search()