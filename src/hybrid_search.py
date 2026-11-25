import heapq

class HybridSearcher:
    """
    Hybrid Search Engine fusing Lexical Retrieval (BM25 or TF-IDF)
    and Dense Semantic Retrieval (Sentence Transformers / LSA)
    using convex score combination with robust min-max normalization:
    
    HybridScore(d, q) = alpha * S_lexical_norm(d, q) + (1 - alpha) * S_semantic_norm(d, q)
    
    Also supports Reciprocal Rank Fusion (RRF) as an alternative ranking strategy.
    """
    def __init__(self, lexical_scorer, semantic_searcher, default_alpha=0.5, norm_strategy="min_max"):
        self.lexical_scorer = lexical_scorer # BM25Scorer or TF-IDF Ranker
        self.semantic_searcher = semantic_searcher
        self.default_alpha = float(default_alpha)
        self.norm_strategy = norm_strategy
        
    def _normalize_scores(self, score_dict):
        """Applies Min-Max normalization to a dictionary of {doc_id: raw_score}."""
        if not score_dict:
            return {}
        scores = list(score_dict.values())
        min_s = min(scores)
        max_s = max(scores)
        
        if max_s - min_s < 1e-9:
            # All scores identical
            return {doc_id: 1.0 for doc_id in score_dict}
            
        return {
            doc_id: (s - min_s) / (max_s - min_s)
            for doc_id, s in score_dict.items()
        }

    def get_top_k(self, query_tokens, query_text, k=10, alpha=None, candidate_pool_size=50):
        """
        Retrieves Top-K documents by combining lexical and semantic rankings.
        
        Args:
            query_tokens: list of preprocessed stems for lexical search.
            query_text: raw or expanded string for semantic search.
            k: number of final results to return.
            alpha: weight for lexical score (1 - alpha for semantic score). Range [0.0, 1.0].
            candidate_pool_size: candidate depth retrieved from each underlying system.
        """
        if alpha is None:
            alpha = self.default_alpha
        alpha = float(max(0.0, min(1.0, alpha)))
        
        # 1. Retrieve Lexical Candidates
        lexical_top, lexical_contribs = self.lexical_scorer.get_top_k(query_tokens, k=candidate_pool_size)
        raw_lexical_scores = {doc_id: score for score, doc_id in lexical_top}
        
        # 2. Retrieve Semantic Candidates in a single vector dot product
        all_semantic_scores = self.semantic_searcher.get_all_similarities(query_text)
        
        # Candidate set: Top lexical candidates + top semantic candidates
        semantic_top = heapq.nlargest(candidate_pool_size, all_semantic_scores.items(), key=lambda x: x[1])
        raw_semantic_scores = {doc_id: score for doc_id, score in semantic_top}
        
        # 3. Union of all candidate documents
        candidate_ids = set(raw_lexical_scores.keys()).union(set(raw_semantic_scores.keys()))
        if not candidate_ids:
            return [], {}
            
        # Fast score alignment for union
        semantic_exps = {}
        for doc_id in candidate_ids:
            if doc_id not in raw_lexical_scores:
                score, contrib = self.lexical_scorer.score_document(query_tokens, doc_id)
                raw_lexical_scores[doc_id] = score
                if contrib:
                    lexical_contribs[doc_id] = contrib
            # For semantic score, look up directly from precomputed all_semantic_scores
            s_score = all_semantic_scores.get(doc_id, 0.0)
            raw_semantic_scores[doc_id] = s_score
            band = "Strong Semantic Alignment" if s_score >= 0.65 else ("Moderate Semantic Alignment" if s_score >= 0.45 else "Thematic Overlap")
            semantic_exps[doc_id] = {
                'semantic_score': s_score,
                'alignment_band': band,
                'model_name': self.semantic_searcher.model_name,
                'dimension': self.semantic_searcher.embedding_dim
            }
                    
        # 4. Normalize Scores
        norm_lexical = self._normalize_scores(raw_lexical_scores)
        norm_semantic = self._normalize_scores(raw_semantic_scores)
        
        # 5. Convex Score Fusion
        hybrid_scores = []
        detailed_explanations = {}
        
        for doc_id in candidate_ids:
            s_lex_norm = norm_lexical.get(doc_id, 0.0)
            s_sem_norm = norm_semantic.get(doc_id, 0.0)
            
            lex_part = alpha * s_lex_norm
            sem_part = (1.0 - alpha) * s_sem_norm
            final_score = lex_part + sem_part
            
            # Determine dominant ranking driver
            if lex_part > sem_part * 1.2:
                driver = "Lexical Keyword Match (Exact Term Saturation)"
            elif sem_part > lex_part * 1.2:
                driver = "Semantic Context (Dense Embedding Alignment)"
            else:
                driver = "Balanced Lexical-Semantic Synergy"
                
            detailed_explanations[doc_id] = {
                'hybrid_score': final_score,
                'alpha': alpha,
                'raw_lexical': raw_lexical_scores.get(doc_id, 0.0),
                'norm_lexical': s_lex_norm,
                'lexical_contribution': lex_part,
                'raw_semantic': raw_semantic_scores.get(doc_id, 0.0),
                'norm_semantic': s_sem_norm,
                'semantic_contribution': sem_part,
                'ranking_driver': driver,
                'lexical_terms': lexical_contribs.get(doc_id, {}),
                'semantic_details': semantic_exps.get(doc_id, {})
            }
            
            hybrid_scores.append((final_score, doc_id))
            
        top_k = heapq.nlargest(k, hybrid_scores, key=lambda x: x[0])
        return top_k, detailed_explanations

    def get_top_k_rrf(self, query_tokens, query_text, k=10, rrf_k=60, candidate_pool_size=50):
        """
        Reciprocal Rank Fusion (RRF) alternative ranking:
        RRF_Score(d) = 1 / (k + rank_lex(d)) + 1 / (k + rank_sem(d))
        """
        lexical_top, _ = self.lexical_scorer.get_top_k(query_tokens, k=candidate_pool_size)
        semantic_top, _ = self.semantic_searcher.get_top_k(query_text, k=candidate_pool_size)
        
        lex_ranks = {doc_id: rank for rank, (_, doc_id) in enumerate(lexical_top, 1)}
        sem_ranks = {doc_id: rank for rank, (_, doc_id) in enumerate(semantic_top, 1)}
        
        all_docs = set(lex_ranks.keys()).union(set(sem_ranks.keys()))
        rrf_scores = []
        explanations = {}
        
        for doc_id in all_docs:
            r_lex = lex_ranks.get(doc_id, candidate_pool_size + 10)
            r_sem = sem_ranks.get(doc_id, candidate_pool_size + 10)
            
            score_lex = 1.0 / (rrf_k + r_lex)
            score_sem = 1.0 / (rrf_k + r_sem)
            final_rrf = score_lex + score_sem
            
            rrf_scores.append((final_rrf, doc_id))
            explanations[doc_id] = {
                'rrf_score': final_rrf,
                'lexical_rank': r_lex,
                'semantic_rank': r_sem,
                'score_lex_part': score_lex,
                'score_sem_part': score_sem
            }
            
        top_k = heapq.nlargest(k, rrf_scores, key=lambda x: x[0])
        return top_k, explanations
