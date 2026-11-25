from collections import Counter
import re

class QueryExpander:
    """
    Academic Query Expansion module supporting:
    1. WordNet synset synonym expansion
    2. Curated Academic Computer Science domain term/abbreviation expansion
    3. Pseudo-Relevance Feedback (PRF / Rocchio-inspired term mining)
    """
    def __init__(self, preprocessor=None, inverted_index=None):
        self.preprocessor = preprocessor
        self.index = inverted_index
        
        # Curated Academic CS domain abbreviations and synonyms
        self.domain_synonyms = {
            "ml": ["machine", "learning"],
            "nlp": ["natural", "language", "processing"],
            "ir": ["information", "retrieval"],
            "llm": ["large", "language", "models"],
            "llms": ["large", "language", "models"],
            "ai": ["artificial", "intelligence"],
            "cv": ["computer", "vision"],
            "rl": ["reinforcement", "learning"],
            "gnn": ["graph", "neural", "networks"],
            "cnn": ["convolutional", "neural", "networks"],
            "rnn": ["recurrent", "neural", "networks"],
            "ann": ["approximate", "nearest", "neighbor"],
            "gpu": ["graphics", "processing", "unit", "acceleration"],
            "kg": ["knowledge", "graph"],
            "gan": ["generative", "adversarial", "networks"],
            "transformer": ["attention", "self-attention", "bert"],
            "retrieval": ["search", "ranking", "indexing"],
            "distributed": ["parallel", "cluster", "decentralized"]
        }

    def _get_wordnet_synonyms(self, word, max_synonyms=2):
        """Retrieves synset lemmas from NLTK WordNet."""
        try:
            from nltk.corpus import wordnet as wn
            synonyms = set()
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    clean_name = lemma.name().lower().replace('_', ' ')
                    # Avoid identical word or punctuation
                    if clean_name != word.lower() and len(clean_name) > 2 and ' ' not in clean_name:
                        synonyms.add(clean_name)
                        if len(synonyms) >= max_synonyms:
                            return list(synonyms)
            return list(synonyms)
        except Exception:
            return []

    def expand_query(self, query_text, method="hybrid", max_terms=3, ranker=None):
        """
        Expands query_text using specified strategy:
        - 'wordnet': linguistic WordNet synset lemmas
        - 'domain': academic CS abbreviation & concept expansion
        - 'prf': pseudo-relevance feedback from top documents
        - 'hybrid': combines domain synonyms with WordNet
        
        Returns:
            expanded_text (str): query string augmented with expansion terms
            expanded_tokens (list): list of preprocessed stems for lexical search
            expansion_details (dict): metadata describing original vs added terms
        """
        if not query_text:
            return query_text, [], {}
            
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', query_text.lower())
        added_terms = []
        term_provenance = {}
        
        # 1. Domain-specific academic expansion
        if method in ["domain", "hybrid"]:
            for w in words:
                if w in self.domain_synonyms:
                    syns = self.domain_synonyms[w]
                    for s in syns:
                        if s not in words and s not in added_terms:
                            added_terms.append(s)
                            term_provenance[s] = f"Domain abbreviation/concept of '{w}'"
                            if len(added_terms) >= max_terms:
                                break
                if len(added_terms) >= max_terms:
                    break

        # 2. WordNet expansion
        if method in ["wordnet", "hybrid"] and len(added_terms) < max_terms:
            for w in words:
                # Skip words that are very short or numbers
                if len(w) <= 2 or w.isdigit():
                    continue
                wn_syns = self._get_wordnet_synonyms(w, max_synonyms=1)
                for s in wn_syns:
                    if s not in words and s not in added_terms:
                        added_terms.append(s)
                        term_provenance[s] = f"WordNet synset of '{w}'"
                        if len(added_terms) >= max_terms:
                            break
                if len(added_terms) >= max_terms:
                    break

        # 3. Pseudo-Relevance Feedback (PRF)
        if method == "prf" and ranker is not None and self.index is not None and self.preprocessor is not None:
            initial_tokens = self.preprocessor.process(query_text)
            top_docs, _ = ranker.get_top_k(initial_tokens, k=3)
            
            # Mine prominent terms from top document abstracts
            candidate_counts = Counter()
            for _, doc_id in top_docs:
                meta = self.index.doc_metadata.get(doc_id, {})
                doc_text = f"{meta.get('title', '')} {meta.get('body', '')}"
                doc_tokens = self.preprocessor.process(doc_text)
                for t in doc_tokens:
                    if t not in initial_tokens:
                        candidate_counts[t] += 1
                        
            # Select top feedback terms
            for term, count in candidate_counts.most_common(max_terms):
                if term not in added_terms:
                    added_terms.append(term)
                    term_provenance[term] = f"PRF top document co-occurrence (freq={count})"

        # Assemble expanded text
        if added_terms:
            expanded_text = f"{query_text} {' '.join(added_terms)}"
        else:
            expanded_text = query_text

        # Process tokens
        if self.preprocessor:
            expanded_tokens = self.preprocessor.process(expanded_text)
        else:
            expanded_tokens = expanded_text.split()

        expansion_details = {
            'original_query': query_text,
            'expanded_query': expanded_text,
            'added_terms': added_terms,
            'provenance': term_provenance,
            'method': method,
            'term_count_increase': len(added_terms)
        }
        
        return expanded_text, expanded_tokens, expansion_details
