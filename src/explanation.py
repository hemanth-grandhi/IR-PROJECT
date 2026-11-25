class Explainer:
    """
    Transparent Information Retrieval Explanation Engine.
    Provides mathematically grounded attribution for:
    - Classical TF-IDF / Cosine Similarity (Mid-term compatibility)
    - Probabilistic Okapi BM25 term saturation and length normalization
    - Dense Semantic Vector Cosine Similarity
    - Convex Hybrid Linear Fusion (alpha * lexical + (1-alpha) * semantic)
    - Query Expansion term contribution and provenance
    """
    
    @staticmethod
    def format_explanation(doc_id, metadata, score, contributions, max_terms=3):
        """
        Original Mid-term explanation formatter (Preserved for 100% backward compatibility).
        Formats an explainable IR output for a single document under TF-IDF/Cosine.
        """
        title = metadata.get('title', 'Unknown Title')
        
        output = [
            f"Document: {title} (ID: {doc_id})",
            f"Score: {score:.4f}",
            "Why this ranked highly:"
        ]
        
        if not contributions:
            output.append("  (No direct lexical terms recorded)")
            return "\n".join(output)
            
        # Sort terms by their contribution to the final score
        sorted_terms = sorted(contributions.items(), key=lambda x: x[1].get('total', 0.0), reverse=True)
        
        for term, scores in sorted_terms[:max_terms]:
            total = scores.get('total', 0.0)
            t_score = scores.get('title_score', 0.0)
            b_score = scores.get('body_score', 0.0)
            
            if total > 0.3:
                impact = "high"
            elif total > 0.1:
                impact = "medium"
            else:
                impact = "low"
                
            zone_info = []
            if t_score > 0: zone_info.append("title")
            if b_score > 0: zone_info.append("body")
            zone_str = f" (matched in: {', '.join(zone_info)})" if zone_info else ""
            
            output.append(f"  - '{term}' -> {impact} contribution ({total:.4f}){zone_str}")
            
        return "\n".join(output)

    @staticmethod
    def format_bm25_explanation(doc_id, metadata, score, contributions, max_terms=5):
        """Formats detailed attribution for Okapi BM25 retrieval."""
        title = metadata.get('title', 'Unknown Title')
        output = [
            f"Document: {title} (ID: {doc_id})",
            f"BM25 Ranking Score: {score:.4f}",
            "Term-Level Attribution & Saturation Breakdown:"
        ]
        if not contributions:
            output.append("  (No direct term matches)")
            return "\n".join(output)
            
        sorted_terms = sorted(contributions.items(), key=lambda x: x[1].get('total', 0.0), reverse=True)
        for term, d in sorted_terms[:max_terms]:
            tot = d.get('total', 0.0)
            idf = d.get('idf', 0.0)
            t_tf = d.get('title_tf', 0)
            b_tf = d.get('body_tf', 0)
            t_score = d.get('title_score', 0.0)
            b_score = d.get('body_score', 0.0)
            
            zones = []
            if t_tf > 0: zones.append(f"Title (tf={t_tf}, pts={t_score:.3f})")
            if b_tf > 0: zones.append(f"Abstract/Body (tf={b_tf}, pts={b_score:.3f})")
            zone_desc = " & ".join(zones) if zones else "No direct occurrences"
            
            output.append(f"  * '{term}' -> Total: {tot:.4f} [IDF: {idf:.3f}] | Matched in: {zone_desc}")
            
        return "\n".join(output)

    @staticmethod
    def format_semantic_explanation(doc_id, metadata, score, semantic_info):
        """Formats attribution for dense semantic vector retrieval."""
        title = metadata.get('title', 'Unknown Title')
        band = semantic_info.get('alignment_band', 'Semantic Match')
        model = semantic_info.get('model_name', 'Dense Embeddings')
        dim = semantic_info.get('dimension', 384)
        
        output = [
            f"Document: {title} (ID: {doc_id})",
            f"Semantic Cosine Similarity: {score:.4f}",
            f"Model: {model} ({dim}-dimensional latent dense space)",
            f"Alignment Level: {band}",
            "Explanation: The document abstract was placed in close vector proximity to the query embedding in dense semantic space, capturing conceptual relatedness even in the absence of exact literal keyword overlap."
        ]
        return "\n".join(output)

    @staticmethod
    def format_hybrid_explanation(doc_id, metadata, hybrid_details, max_terms=4):
        """
        Formats a comprehensive hybrid explanation showing:
        - Lexical vs Semantic score decomposition
        - Alpha weighting calculation
        - Primary ranking driver
        - Term matches and zone provenance
        - Dense embedding similarity
        """
        title = metadata.get('title', 'Unknown Title')
        h_score = hybrid_details.get('hybrid_score', 0.0)
        alpha = hybrid_details.get('alpha', 0.5)
        raw_lex = hybrid_details.get('raw_lexical', 0.0)
        norm_lex = hybrid_details.get('norm_lexical', 0.0)
        lex_contrib = hybrid_details.get('lexical_contribution', 0.0)
        raw_sem = hybrid_details.get('raw_semantic', 0.0)
        norm_sem = hybrid_details.get('norm_semantic', 0.0)
        sem_contrib = hybrid_details.get('semantic_contribution', 0.0)
        driver = hybrid_details.get('ranking_driver', 'Combined Evidence')
        
        output = [
            f"Document: {title} (ID: {doc_id})",
            f"Final Hybrid Score: {h_score:.4f}",
            f"Ranking Fusion Formula: {alpha:.2f} * LexicalNorm({norm_lex:.4f}) + {(1-alpha):.2f} * SemanticNorm({norm_sem:.4f})",
            f"  - Lexical Component: {lex_contrib:.4f} (Raw Score: {raw_lex:.4f})",
            f"  - Semantic Component: {sem_contrib:.4f} (Raw Cosine Sim: {raw_sem:.4f})",
            f"Primary Ranking Driver: {driver}",
            "Evidence Breakdown:"
        ]
        
        # Add Lexical Term Details
        lex_terms = hybrid_details.get('lexical_terms', {})
        if lex_terms:
            sorted_terms = sorted(lex_terms.items(), key=lambda x: x[1].get('total', 0.0), reverse=True)
            matched_terms_str = []
            for t, td in sorted_terms[:max_terms]:
                t_score = td.get('title_score', 0.0)
                b_score = td.get('body_score', 0.0)
                zones = []
                if t_score > 0: zones.append("Title")
                if b_score > 0: zones.append("Abstract")
                zone_str = f" in {', '.join(zones)}" if zones else ""
                matched_terms_str.append(f"'{t}'{zone_str} (score={td.get('total', 0.0):.3f})")
            output.append(f"  * Lexical Keywords: {', '.join(matched_terms_str)}")
        else:
            output.append("  * Lexical Keywords: None (Paper ranked purely through semantic association)")

        # Add Semantic Context Details
        sem_details = hybrid_details.get('semantic_details', {})
        if sem_details:
            band = sem_details.get('alignment_band', 'Contextual')
            model = sem_details.get('model_name', 'Transformer')
            output.append(f"  * Semantic Context: {band} using {model}")
            
        return "\n".join(output)

    @staticmethod
    def format_expansion_impact(doc_id, metadata, before_rank, after_rank, expansion_terms):
        """Explains the empirical impact of query expansion on a specific retrieved paper."""
        title = metadata.get('title', 'Unknown Title')
        diff = before_rank - after_rank if (before_rank is not None and after_rank is not None) else None
        
        if before_rank is None and after_rank is not None:
            impact_str = f"Newly discovered paper at Rank #{after_rank} (previously unretrieved without expansion)."
        elif diff is not None and diff > 0:
            impact_str = f"Promoted by +{diff} positions (from Rank #{before_rank} -> Rank #{after_rank})."
        elif diff is not None and diff < 0:
            impact_str = f"Demoted by {abs(diff)} positions (from Rank #{before_rank} -> Rank #{after_rank})."
        else:
            impact_str = f"Maintained stable rank at #{after_rank}."
            
        return f"Expansion Impact for '{title}': {impact_str} Enabled by expansion terms: {', '.join(expansion_terms)}"
