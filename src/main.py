import argparse
import os
import sys

from preprocessing import Preprocessor
from index import InvertedIndex
from tfidf import TFIDFScorer
from ranking import Ranker
from bm25 import BM25Scorer
from semantic_search import SemanticSearcher
from hybrid_search import HybridSearcher
from query_expansion import QueryExpander
from explanation import Explainer

def print_debug_info(idx, scorer, preprocessor, query, query_tokens, query_vec, top_k, bm25_scorer=None):
    print("\n" + "="*50)
    print("INFORMATION RETRIEVAL DEBUG MODE")
    print("="*50)
    print(f"Vocabulary Size: {len(idx.index)}")
    print(f"Total Documents: {idx.total_docs}")
    print(f"Original Query: '{query}'")
    print(f"Tokens: {query_tokens}")
    print("-" * 50)
    
    for token in query_tokens:
        df = idx.get_document_frequency(token)
        idf = scorer.idf_map.get(token, 0)
        weight = query_vec.get(token, 0) if query_vec else 0.0
        bm_idf = bm25_scorer.compute_idf(token) if bm25_scorer else 0.0
        print(f"Term: '{token}'")
        print(f"  DF: {df}")
        print(f"  TF-IDF IDF: {idf:.4f}")
        if bm25_scorer:
            print(f"  BM25 RSJ-IDF: {bm_idf:.4f}")
        print(f"  Query Vector Weight: {weight:.4f}")
        idx.print_debug_postings(token)
        print("-" * 50)
        
    print("Top retrieved documents and their scores:")
    for score, doc_id in top_k[:5]:
        print(f"  {doc_id} -> {score:.4f}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Academic Search Engine (CSD358 Information Retrieval)")
    parser.add_argument("--data", type=str, default="../data/arxiv_corpus.json", help="Path to corpus JSON")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--algo", type=str, choices=['hybrid', 'bm25', 'semantic', 'tfidf', 'smart'], default='hybrid', help="Retrieval algorithm")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight on BM25 (1 - alpha for Semantic)")
    parser.add_argument("--expand", action="store_true", help="Enable automatic Query Expansion")
    parser.add_argument("--debug", action="store_true", help="Enable IR DEBUG mode to show intermediate outputs")
    parser.add_argument("--smart", action="store_true", help="Use SMART lnc.ltc weighting instead of standard TF-IDF (backward compatibility)")
    
    args = parser.parse_args()
    
    corpus_path = os.path.join(os.path.dirname(__file__), args.data)
    if not os.path.exists(corpus_path):
        print(f"Error: Dataset not found at {corpus_path}")
        print("Please run fetch_dataset.py first.")
        sys.exit(1)
        
    print("Initializing Advanced Information Retrieval Pipeline...")
    # 1. Preprocessor & Inverted Index
    preprocessor = Preprocessor()
    idx = InvertedIndex(preprocessor)
    idx.build_from_corpus(corpus_path)
    
    # 2. Classical TF-IDF Scorers
    tfidf_ltc = TFIDFScorer(idx, smart_weighting="ltc.ltc")
    tfidf_ltc.build_vectors()
    
    tfidf_lnc = TFIDFScorer(idx, smart_weighting="lnc.ltc")
    tfidf_lnc.build_vectors()
    
    # 3. BM25 Scorer
    bm25 = BM25Scorer(idx, k1=1.5, b=0.75, title_weight=0.6, body_weight=0.4)
    
    # 4. Semantic Searcher
    semantic = SemanticSearcher()
    semantic.build_or_load_embeddings(corpus_path)
    
    # 5. Hybrid Searcher
    hybrid = HybridSearcher(bm25, semantic, default_alpha=args.alpha)
    
    # 6. Query Expander
    expander = QueryExpander(preprocessor=preprocessor, inverted_index=idx)
    
    selected_algo = 'smart' if args.smart else args.algo
    print(f"\nSystem ready! (Active Algorithm: {selected_algo.upper()}) Type your query or 'quit' to exit.")
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
                
            # Query Expansion
            active_query_text = query
            query_tokens = preprocessor.process(query)
            if not query_tokens:
                print("Query contained only stop words or punctuation. Please try again.")
                continue
                
            if args.expand:
                ranker_ref = bm25 if selected_algo == 'bm25' else Ranker(tfidf_ltc)
                active_query_text, query_tokens, exp_info = expander.expand_query(query, method="hybrid", max_terms=3, ranker=ranker_ref)
                print(f"[Query Expansion Active]: '{query}' -> '{active_query_text}'")
                print(f"  Added Concepts: {', '.join(exp_info.get('added_terms', []))}")
                
            # Retrieval execution
            if selected_algo == 'hybrid':
                top_k, explanations = hybrid.get_top_k(query_tokens, active_query_text, k=args.k, alpha=args.alpha)
            elif selected_algo == 'bm25':
                top_k, explanations = bm25.get_top_k(query_tokens, k=args.k)
            elif selected_algo == 'semantic':
                top_k, explanations = semantic.get_top_k(active_query_text, k=args.k)
            elif selected_algo == 'smart':
                ranker = Ranker(tfidf_lnc, title_weight=0.7, body_weight=0.3)
                top_k, explanations = ranker.get_top_k(query_tokens, k=args.k)
            else: # Standard tfidf
                ranker = Ranker(tfidf_ltc, title_weight=0.7, body_weight=0.3)
                top_k, explanations = ranker.get_top_k(query_tokens, k=args.k)
                
            # Debug Mode
            if args.debug:
                active_scorer = tfidf_lnc if selected_algo == 'smart' else tfidf_ltc
                query_vec, _ = active_scorer.get_query_vector(query_tokens)
                print_debug_info(idx, active_scorer, preprocessor, active_query_text, query_tokens, query_vec, top_k, bm25_scorer=bm25)
                
            # Result Presentation & Attribution
            print(f"\nFound {len(top_k)} results for '{query}' [{selected_algo.upper()}]:")
            print("=" * 60)
            
            for rank, (score, doc_id) in enumerate(top_k, 1):
                metadata = idx.doc_metadata[doc_id]
                print(f"Rank {rank}:")
                if selected_algo == 'hybrid':
                    print(Explainer.format_hybrid_explanation(doc_id, metadata, explanations[doc_id]))
                elif selected_algo == 'bm25':
                    print(Explainer.format_bm25_explanation(doc_id, metadata, score, explanations[doc_id]))
                elif selected_algo == 'semantic':
                    print(Explainer.format_semantic_explanation(doc_id, metadata, score, explanations[doc_id]))
                else:
                    print(Explainer.format_explanation(doc_id, metadata, score, explanations[doc_id]))
                print("-" * 60)
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
