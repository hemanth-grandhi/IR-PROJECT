import unittest
import math
import numpy as np

from src.preprocessing import Preprocessor
from src.index import InvertedIndex
from src.tfidf import TFIDFScorer
from src.ranking import Ranker
from src.jaccard import JaccardBaseline
from src.bm25 import BM25Scorer
from src.semantic_search import SemanticSearcher
from src.hybrid_search import HybridSearcher
from src.query_expansion import QueryExpander
from src.explanation import Explainer

class TestIRPipeline(unittest.TestCase):
    def setUp(self):
        self.p = Preprocessor()
        self.idx = InvertedIndex(self.p)
        
        # Add tiny known corpus
        self.idx.add_document("doc1", "apple banana", "apple apple")
        self.idx.add_document("doc2", "banana", "banana cherry")
        self.idx.add_document("doc3", "cherry date", "date date date")
        
        self.scorer = TFIDFScorer(self.idx, smart_weighting="ltc.ltc")
        self.scorer.build_vectors()
        
        self.ranker = Ranker(self.scorer, title_weight=0.5, body_weight=0.5)
        self.bm25 = BM25Scorer(self.idx, k1=1.5, b=0.75, title_weight=0.5, body_weight=0.5)

    # ---------------------------------------------------------
    # Baseline Tests (Preserved from Mid-term for 100% Compatibility)
    # ---------------------------------------------------------
    def test_tf_and_log_tf(self):
        postings_apple = self.idx.get_postings("apple")
        self.assertEqual(postings_apple["doc1"]["title_tf"], 1)
        self.assertEqual(postings_apple["doc1"]["body_tf"], 2)
        
        expected_log_tf = 1 + math.log10(2)
        actual_log_tf = self.scorer._compute_tf(2)
        self.assertAlmostEqual(actual_log_tf, expected_log_tf)
        
    def test_df_and_idf(self):
        df_banana = self.idx.get_document_frequency("banana")
        self.assertEqual(df_banana, 2)
        
        expected_idf = math.log10(3/2)
        actual_idf = self.scorer._compute_idf(2)
        self.assertAlmostEqual(actual_idf, expected_idf)
        
    def test_tfidf_vectors(self):
        apple_idf = math.log10(3/1)
        expected_w = (1 + math.log10(2)) * apple_idf
        
        doc1_vec = self.scorer.doc_vectors["doc1"]["body_vec"]
        apple_token = self.p.process("apple")[0]
        self.assertAlmostEqual(doc1_vec[apple_token], expected_w)
        
    def test_vector_norm(self):
        doc2_vec = self.scorer.doc_vectors["doc2"]
        banana_idf = math.log10(3/2)
        expected_norm = math.sqrt((1 * banana_idf)**2)
        self.assertAlmostEqual(doc2_vec["title_norm"], expected_norm)
        
    def test_cosine_similarity(self):
        tokens = self.p.process("banana")
        q_vec, q_norm = self.scorer.get_query_vector(tokens)
        score, _ = self.ranker.score_document(q_vec, q_norm, "doc2")
        self.assertGreater(score, 0)
        
    def test_jaccard(self):
        jaccard = JaccardBaseline(self.idx, self.p)
        tokens = self.p.process("banana apple")
        top_k, _ = jaccard.get_top_k(tokens, k=1)
        self.assertEqual(top_k[0][1], "doc1")
        self.assertAlmostEqual(top_k[0][0], 1.0)

    # ---------------------------------------------------------
    # Advanced Upgraded System Tests
    # ---------------------------------------------------------
    def test_bm25_term_saturation(self):
        """Verifies that BM25 term frequency saturation exhibits diminishing marginal gains."""
        k1 = 1.5
        b = 0.75
        avg_len = 10.0
        doc_len = 10.0
        
        # Gain from tf=1 to tf=2
        w1 = self.bm25._term_bm25_tf_weight(1, doc_len, avg_len)
        w2 = self.bm25._term_bm25_tf_weight(2, doc_len, avg_len)
        gain_1_to_2 = w2 - w1
        
        # Gain from tf=10 to tf=20
        w10 = self.bm25._term_bm25_tf_weight(10, doc_len, avg_len)
        w20 = self.bm25._term_bm25_tf_weight(20, doc_len, avg_len)
        gain_10_to_20 = w20 - w10
        
        # Because of saturation, gain from 10 to 20 MUST be significantly smaller than 1 to 2
        self.assertGreater(gain_1_to_2, gain_10_to_20)
        # As tf -> infinity, weight approaches (k1 + 1) = 2.5
        self.assertLess(w20, k1 + 1.0)

    def test_bm25_length_normalization(self):
        """Verifies that shorter documents receive higher relative weights than longer documents with identical raw TF."""
        tf = 2
        avg_len = 50.0
        short_doc_len = 20.0
        long_doc_len = 100.0
        
        w_short = self.bm25._term_bm25_tf_weight(tf, short_doc_len, avg_len)
        w_long = self.bm25._term_bm25_tf_weight(tf, long_doc_len, avg_len)
        
        self.assertGreater(w_short, w_long)

    def test_bm25_score_document_and_attribution(self):
        """Verifies BM25 score calculation and term attribution structure."""
        tokens = self.p.process("apple banana")
        score, contrib = self.bm25.score_document(tokens, "doc1")
        self.assertGreater(score, 0.0)
        self.assertIn("appl", contrib)
        self.assertIn("banana", contrib)
        self.assertGreater(contrib["appl"]["total"], 0.0)

    def test_query_expansion_domain_synonyms(self):
        """Verifies Academic CS domain acronym expansion."""
        expander = QueryExpander(preprocessor=self.p, inverted_index=self.idx)
        text, tokens, info = expander.expand_query("ml research", method="domain")
        self.assertIn("machine", info['added_terms'])
        self.assertIn("learning", info['added_terms'])
        self.assertIn("machin", tokens)

    def test_query_expansion_wordnet(self):
        """Verifies WordNet synset synonym expansion."""
        expander = QueryExpander(preprocessor=self.p, inverted_index=self.idx)
        text, tokens, info = expander.expand_query("retrieval", method="wordnet")
        # WordNet contains synset lemmas for retrieval (e.g. recovery, search)
        self.assertGreater(len(info['added_terms']), 0)

    def test_semantic_fallback_lsa(self):
        """Verifies Dense LSA Semantic Fallback when running without neural model."""
        sem_fallback = SemanticSearcher(use_fallback_only=True)
        # Fake mini corpus
        mini_corpus = [
            {"document_id": "d1", "title": "Information Retrieval", "body": "Search engine indexing and ranking."},
            {"document_id": "d2", "title": "Computer Vision", "body": "Image classification and object detection."},
            {"document_id": "d3", "title": "Quantum Computing", "body": "Superposition and quantum gates algorithm."}
        ]
        vecs = sem_fallback._train_lsa_fallback(mini_corpus)
        sem_fallback.embeddings = vecs
        sem_fallback.doc_ids = ["d1", "d2", "d3"]
        sem_fallback.doc_id_to_idx = {"d1": 0, "d2": 1, "d3": 2}
        sem_fallback.embedding_dim = vecs.shape[1]
        
        top_k, _ = sem_fallback.get_top_k("information retrieval search", k=1)
        self.assertEqual(top_k[0][1], "d1")

    def test_hybrid_convex_score_fusion(self):
        """Verifies convex hybrid score fusion: S = alpha * Lexical + (1 - alpha) * Semantic."""
        # Create mock semantic searcher
        sem_mock = SemanticSearcher(use_fallback_only=True)
        sem_mock.embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5]
        ], dtype=np.float32)
        sem_mock.doc_ids = ["doc1", "doc2", "doc3"]
        sem_mock.doc_id_to_idx = {"doc1": 0, "doc2": 1, "doc3": 2}
        sem_mock.embedding_dim = 2
        sem_mock.encode_text = lambda text: np.array([1.0, 0.0], dtype=np.float32)
        
        hybrid = HybridSearcher(self.bm25, sem_mock, default_alpha=0.6)
        
        # Test alpha=1.0 (pure lexical)
        q_tokens = self.p.process("apple")
        top_lex, exp_lex = hybrid.get_top_k(q_tokens, "apple", k=1, alpha=1.0)
        self.assertEqual(top_lex[0][1], "doc1")
        self.assertAlmostEqual(exp_lex[top_lex[0][1]]['alpha'], 1.0)

    def test_explainability_formatter(self):
        """Verifies that Explainer produces well-formatted text for all modes."""
        meta = {"title": "Distributed Systems"}
        contribs = {"distribut": {"total": 0.45, "title_score": 0.25, "body_score": 0.20}}
        
        # Test legacy format
        legacy_txt = Explainer.format_explanation("doc1", meta, 0.85, contribs)
        self.assertIn("Why this ranked highly:", legacy_txt)
        self.assertIn("distribut", legacy_txt)
        
        # Test hybrid format
        hybrid_info = {
            'hybrid_score': 0.88,
            'alpha': 0.5,
            'raw_lexical': 4.5,
            'norm_lexical': 0.9,
            'lexical_contribution': 0.45,
            'raw_semantic': 0.82,
            'norm_semantic': 0.86,
            'semantic_contribution': 0.43,
            'ranking_driver': 'Balanced Synergy',
            'lexical_terms': contribs,
            'semantic_details': {'alignment_band': 'Strong Semantic Alignment', 'model_name': 'all-MiniLM-L6-v2'}
        }
        hybrid_txt = Explainer.format_hybrid_explanation("doc1", meta, hybrid_info)
        self.assertIn("Final Hybrid Score: 0.8800", hybrid_txt)
        self.assertIn("Balanced Synergy", hybrid_txt)

if __name__ == "__main__":
    unittest.main()
