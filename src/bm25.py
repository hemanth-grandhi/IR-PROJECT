import math
import heapq

class BM25Scorer:
    """
    Okapi BM25 ranking algorithm with term-frequency saturation,
    document-length normalization, and optional zone weighting.
    """
    def __init__(self, inverted_index, k1=1.5, b=0.75, title_weight=0.6, body_weight=0.4):
        self.index = inverted_index
        self.k1 = float(k1)
        self.b = float(b)
        self.title_weight = float(title_weight)
        self.body_weight = float(body_weight)
        
        self.N = self.index.total_docs
        self.idf_map = {}
        self.avg_doc_len = 0.0
        self.avg_title_len = 0.0
        self.avg_body_len = 0.0
        
        self._compute_corpus_statistics()
        
    def _compute_corpus_statistics(self):
        """Precomputes average document lengths and Robertson-Spärck Jones IDF for all terms."""
        if self.N == 0:
            return
            
        total_tokens = 0
        total_title_tokens = 0
        total_body_tokens = 0
        
        for doc_id, length_info in self.index.doc_lengths.items():
            if isinstance(length_info, dict):
                total_title_tokens += length_info.get('title_len', 0)
                total_body_tokens += length_info.get('body_len', 0)
                total_tokens += length_info.get('total_len', 0)
            else:
                total_tokens += length_info
                
        self.avg_doc_len = total_tokens / self.N if self.N > 0 else 1.0
        self.avg_title_len = total_title_tokens / self.N if self.N > 0 else 1.0
        self.avg_body_len = total_body_tokens / self.N if self.N > 0 else 1.0
        
        # Lucene/Robertson-Spärck Jones IDF with non-negative smoothing
        for term, postings in self.index.index.items():
            df = len(postings)
            # Standard smoothed RSJ formula: ln(1 + (N - df + 0.5) / (df + 0.5))
            self.idf_map[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            
    def compute_idf(self, term):
        """Returns the BM25 IDF for a specific term."""
        if term in self.idf_map:
            return self.idf_map[term]
        # For unseen query terms
        df = len(self.index.get_postings(term))
        idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5)) if self.N > 0 else 0.0
        return max(0.0, idf)

    def _term_bm25_tf_weight(self, tf, doc_len, avg_len):
        """Computes BM25 saturated TF weight: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))"""
        if tf <= 0:
            return 0.0
        length_factor = 1.0 - self.b + self.b * (doc_len / avg_len if avg_len > 0 else 1.0)
        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * length_factor
        return numerator / denominator if denominator > 0 else 0.0

    def score_document(self, query_tokens, doc_id):
        """
        Computes BM25 score for a document given query tokens.
        Decomposes term-level contributions and zone contributions for explainability.
        """
        doc_metadata = self.index.doc_metadata.get(doc_id)
        if not doc_metadata:
            return 0.0, {}
            
        len_info = self.index.doc_lengths.get(doc_id, {})
        title_len = len_info.get('title_len', self.avg_title_len) if isinstance(len_info, dict) else self.avg_title_len
        body_len = len_info.get('body_len', self.avg_body_len) if isinstance(len_info, dict) else self.avg_body_len
        
        total_score = 0.0
        term_contributions = {}
        
        # Count frequency of terms in query (support repeated query terms)
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
            
        for term, q_weight in query_tf.items():
            postings = self.index.index.get(term, {})
            if doc_id not in postings:
                continue
                
            tfs = postings[doc_id]
            title_tf = tfs.get('title_tf', 0)
            body_tf = tfs.get('body_tf', 0)
            
            idf = self.compute_idf(term)
            if idf <= 0:
                continue
                
            title_tf_norm = self._term_bm25_tf_weight(title_tf, title_len, self.avg_title_len)
            body_tf_norm = self._term_bm25_tf_weight(body_tf, body_len, self.avg_body_len)
            
            term_title_score = idf * title_tf_norm
            term_body_score = idf * body_tf_norm
            
            term_total = self.title_weight * term_title_score + self.body_weight * term_body_score
            term_total *= q_weight # Multiplied by query term repetition
            
            if term_total > 0:
                total_score += term_total
                term_contributions[term] = {
                    'title_score': self.title_weight * term_title_score,
                    'body_score': self.body_weight * term_body_score,
                    'total': term_total,
                    'idf': idf,
                    'title_tf': title_tf,
                    'body_tf': body_tf,
                    'title_tf_norm': title_tf_norm,
                    'body_tf_norm': body_tf_norm
                }
                
        return total_score, term_contributions

    def get_top_k(self, query_tokens, k=10):
        """
        Retrieves Top-K documents using BM25 and Min-Heap selection (O(N log K)).
        """
        if not query_tokens:
            return [], {}
            
        # Identify candidate documents with at least one matching term
        candidate_docs = set()
        for token in query_tokens:
            if token in self.index.index:
                candidate_docs.update(self.index.index[token].keys())
                
        scores = []
        all_contributions = {}
        
        for doc_id in candidate_docs:
            score, contributions = self.score_document(query_tokens, doc_id)
            if score > 0:
                scores.append((score, doc_id))
                all_contributions[doc_id] = contributions
                
        top_k = heapq.nlargest(k, scores, key=lambda x: x[0])
        return top_k, all_contributions
