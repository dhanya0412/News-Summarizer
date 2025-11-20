# backend/ingest/summarizer.py
import os
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import your search module
from search import search

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️  Warning: GEMINI_API_KEY not found in .env file")
    print("Please get your API key from: https://aistudio.google.com/app/apikey")
    print("Add it to .env as: GEMINI_API_KEY=your_key_here")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.5-flash')


def generate_summary(query: str, docs: List[Dict], max_sentences: int = 5) -> Tuple[str, List[int]]:
    """
    Generate a concise summary from retrieved documents using Gemini API.
    
    Args:
        query: The user's search query
        docs: List of document dicts from search.py (with 'title', 'content', 'url', etc.)
        max_sentences: Maximum sentences in summary (default 5)
    
    Returns:
        Tuple of (summary_text, list_of_cited_doc_indices)
    """
    if not docs:
        return "No documents found to summarize.", []
    
    # Prepare documents for prompt
    doc_context = ""
    for i, doc in enumerate(docs, 1):
        title = doc.get('title', 'Untitled')
        content = doc.get('content', doc.get('snippet', ''))[:2000]  # Limit to 2000 chars per doc
        doc_context += f"[Doc {i}] Title: {title}\nContent: {content}\n\n"
    
    # Craft strict prompt to prevent hallucinations
    prompt = f"""You are a news summarizer. Your task is to provide a concise, accurate summary based ONLY on the provided documents.

CRITICAL RULES:
1. Use ONLY information from the documents below
2. Do NOT add any external knowledge or assumptions
3. Keep summary to {max_sentences} sentences or less
4. Be specific and answer the user's query directly
5. At the end, list which documents you used as: "Sources: Doc X, Doc Y"

Documents:
{doc_context}

User Query: {query}

Provide a concise summary answering the query using ONLY these documents:"""

    try:
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Extract cited doc numbers
        cited_docs = []
        doc_pattern = r'Doc\s+(\d+)'
        matches = re.findall(doc_pattern, summary)
        cited_docs = sorted(set(int(m) for m in matches if int(m) <= len(docs)))
        
        return summary, cited_docs
    
    except Exception as e:
        return f"Error generating summary: {str(e)}", []


def extract_relevant_passages(query: str, docs: List[Dict], top_k: int = 5, 
                              min_similarity: float = 0.08) -> Tuple[List[str], List[int]]:
    """
    Use TF-IDF to find most relevant passages from cached documents.
    
    Args:
        query: Follow-up query text
        docs: Cached documents from initial search
        top_k: Number of top passages to extract
        min_similarity: Minimum similarity threshold (below this = no info found)
    
    Returns:
        Tuple of (list_of_relevant_passages, list_of_doc_indices)
    """
    if not docs:
        return [], []
    
    # Extract content from docs and split into sentences
    all_sentences = []
    sentence_to_doc = []  # Track which doc each sentence came from
    
    for doc_idx, doc in enumerate(docs):
        content = doc.get('content', doc.get('snippet', ''))
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Min 20 chars
        
        for sent in sentences:
            all_sentences.append(sent)
            sentence_to_doc.append(doc_idx)
    
    if not all_sentences:
        return [], []
    
    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    try:
        # Fit on all sentences + query
        corpus = all_sentences + [query]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Query vector is the last one
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        
        # Check if any similarity above threshold
        if similarities.max() < min_similarity:
            return [], []
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Extract passages and their doc sources
        relevant_passages = []
        source_docs = []
        
        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                relevant_passages.append(all_sentences[idx])
                source_docs.append(sentence_to_doc[idx])
        
        return relevant_passages, source_docs
    
    except Exception as e:
        print(f"⚠️  Error in TF-IDF extraction: {e}")
        return [], []


def answer_followup(followup_query: str, cached_docs: List[Dict], 
                   original_query: str) -> Tuple[str, List[int]]:
    """
    Answer a follow-up query using only the cached documents.
    
    Args:
        followup_query: The follow-up question
        cached_docs: Documents from original search
        original_query: Original search query (for context)
    
    Returns:
        Tuple of (answer_text, list_of_cited_doc_indices)
    """
    if not cached_docs:
        return "No documents available to answer from.", []
    
    # Find relevant passages using TF-IDF
    passages, source_doc_indices = extract_relevant_passages(
        followup_query, cached_docs, top_k=5, min_similarity=0.1
    )
    
    if not passages:
        return "❌ No information about this topic found in the retrieved documents.", []
    
    # Build context from relevant passages
    passage_context = ""
    for i, passage in enumerate(passages, 1):
        doc_idx = source_doc_indices[i-1]
        doc_title = cached_docs[doc_idx].get('title', 'Untitled')
        passage_context += f"[Doc {doc_idx+1}] {doc_title}\nPassage: {passage}\n\n"
    
    # Craft prompt with original query for context
    prompt = f"""You are answering a follow-up question based ONLY on the provided document passages.

CRITICAL RULES:
1. Use ONLY information from the passages below
2. Do NOT add external knowledge
3. Keep answer concise (2-4 sentences)
4. If passages don't contain the answer, say so
5. End with "Sources: Doc X, Doc Y"

Original Query Context: {original_query}

Relevant Passages:
{passage_context}

Follow-up Question: {followup_query}

Answer using ONLY the passages above:"""

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Extract cited doc numbers
        cited_docs = []
        doc_pattern = r'Doc\s+(\d+)'
        matches = re.findall(doc_pattern, answer)
        cited_docs = sorted(set(int(m) for m in matches if int(m) <= len(cached_docs)))
        
        return answer, cited_docs
    
    except Exception as e:
        return f"Error generating answer: {str(e)}", []


def display_results(query: str, docs: List[Dict], summary: str):
    """Display search results in a nice CLI format."""
    print("\n" + "=" * 80)
    print(f"🔍 Query: {query}")
    print("=" * 80)
    
    print(f"\n✅ Found {len(docs)} relevant document(s)")
    
    print("\n" + "─" * 80)
    print("📝 SUMMARY:")
    print("─" * 80)
    print(summary)
    
    # Display top min(3, num_docs) links
    num_links = min(3, len(docs))
    print("\n" + "─" * 80)
    print(f"🔗 Check these links for more information:")
    print("─" * 80)
    for i in range(num_links):
        doc = docs[i]
        print(f"{i+1}. {doc.get('title', 'Untitled')}")
        print(f"   🌐 {doc.get('url', 'No URL')}")
        if doc.get('published'):
            print(f"   📅 {doc.get('published')}")
        print()


def cli_interface():
    """Main CLI interface for testing the summarizer."""
    print("\n" + "=" * 80)
    print("📰 NEWS SUMMARIZER - CLI Testing Interface")
    print("=" * 80)
    print("Commands:")
    print("  - Enter a query to search and get a summary")
    print("  - After summary, ask follow-up questions")
    print("  - Type 'new' to start a new search")
    print("  - Type 'quit' or 'exit' to stop")
    print("=" * 80 + "\n")
    
    cached_docs = []
    original_query = ""
    
    while True:
        try:
            if not cached_docs:
                # New search
                query = input("🔎 Enter your search query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Search using search.py
                print("\n⚡ Searching...")
                results = search(query, k=10, prune=True, debug=False)
                
                if not results:
                    print("❌ No results found. Try a different query.\n")
                    continue
                
                # Cache results for follow-ups
                cached_docs = results
                original_query = query
                
                # Generate summary
                print("🤖 Generating summary...")
                summary, cited = generate_summary(query, results)
                
                # Display results
                display_results(query, results, summary)
            
            else:
                # Follow-up mode
                followup = input("\n💬 Ask a follow-up question (or 'new' for new search, 'quit' to exit): ").strip()
                
                if not followup:
                    continue
                
                if followup.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if followup.lower() == 'new':
                    cached_docs = []
                    original_query = ""
                    print("\n" + "=" * 80 + "\n")
                    continue
                
                # Answer follow-up
                print("\n⚡ Analyzing cached documents...")
                answer, cited = answer_followup(followup, cached_docs, original_query)
                
                print("\n" + "─" * 80)
                print("📝 ANSWER:")
                print("─" * 80)
                print(answer)
                print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    # Check if API key is set
    if not GEMINI_API_KEY:
        print("\n❌ Cannot start: GEMINI_API_KEY not found!")
        print("\nSetup instructions:")
        print("1. Go to: https://aistudio.google.com/app/apikey")
        print("2. Click 'Create API Key'")
        print("3. Copy the key")
        print("4. Add to your .env file: GEMINI_API_KEY=your_key_here")
        print("5. Run this script again\n")
    else:
        cli_interface()