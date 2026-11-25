import os
import json
import heapq
import numpy as np

class SemanticSearcher:
    """
    Dense Semantic Search Engine utilizing Sentence Transformers
    (all-MiniLM-L6-v2) with persistent on-disk embedding storage,
    L2-normalized cosine similarity, and an automatic fallback
    to Latent Semantic Analysis (TruncatedSVD) when offline or resource-constrained.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", embeddings_dir=None, use_fallback_only=False):
        self.model_name = model_name
        self.use_fallback_only = use_fallback_only
        self.embeddings_dir = embeddings_dir or os.path.join(os.path.dirname(__file__), '..', 'embeddings')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.model = None
        self.is_neural = False
        self.lsa_model = None
        self.lsa_vectorizer = None
        
        self.doc_ids = []
        self.doc_id_to_idx = {}
        self.embeddings = None # Numpy array (N, D)
        self.embedding_dim = 0
        
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load SentenceTransformer, or falls back gracefully to Scikit-learn LSA."""
        if not self.use_fallback_only:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading SentenceTransformer model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name)
                self.is_neural = True
                print("Neural SentenceTransformer model loaded successfully.")
                return
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer ({e}). Falling back to Dense LSA semantic engine.")
                
        self.is_neural = False
        self.model_name = "dense-lsa-128"

    def _train_lsa_fallback(self, documents):
        """Trains a 128-dimensional TruncatedSVD Latent Semantic Analysis model on corpus text."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        corpus_texts = [f"{doc.get('title', '')} {doc.get('body', '')}" for doc in documents]
        
        n_components = min(128, max(16, len(documents) // 3))
        self.lsa_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        tfidf_mat = self.lsa_vectorizer.fit_transform(corpus_texts)
        
        self.lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
        dense_vecs = self.lsa_model.fit_transform(tfidf_mat)
        
        # L2 normalize
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense_vecs / norms).astype(np.float32)

    def encode_text(self, text):
        """Encodes a single string into a normalized dense embedding vector."""
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        if self.is_neural and self.model is not None:
            vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32)
        elif self.lsa_vectorizer is not None and self.lsa_model is not None:
            tfidf = self.lsa_vectorizer.transform([text])
            vec = self.lsa_model.transform(tfidf)[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32)
        else:
            dim = self.embedding_dim if self.embedding_dim > 0 else 128
            return np.zeros(dim, dtype=np.float32)

    def build_or_load_embeddings(self, corpus_path):
        """
        Loads precomputed embeddings from disk if available and valid;
        otherwise computes them and persists to embeddings/ directory.
        """
        with open(corpus_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            
        current_doc_ids = [doc['document_id'] for doc in documents]
        N = len(documents)
        
        emb_file = os.path.join(self.embeddings_dir, f"corpus_embeddings_{self.model_name.replace('/', '_')}.npy")
        ids_file = os.path.join(self.embeddings_dir, f"doc_ids_{self.model_name.replace('/', '_')}.json")
        meta_file = os.path.join(self.embeddings_dir, f"metadata_{self.model_name.replace('/', '_')}.json")
        
        # Check if cached files exist and match current corpus
        if os.path.exists(emb_file) and os.path.exists(ids_file):
            try:
                with open(ids_file, 'r', encoding='utf-8') as f:
                    cached_ids = json.load(f)
                if cached_ids == current_doc_ids:
                    print(f"Loading cached semantic embeddings from {emb_file}...")
                    self.embeddings = np.load(emb_file)
                    self.doc_ids = cached_ids
                    self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
                    self.embedding_dim = self.embeddings.shape[1]
                    print(f"Semantic embeddings loaded! Count: {len(self.doc_ids)}, Dimension: {self.embedding_dim}")
                    return
            except Exception as e:
                print(f"Notice: Failed to load cached embeddings ({e}), recomputing...")

        print(f"Computing semantic embeddings for {N} documents using '{self.model_name}'...")
        corpus_texts = []
        for doc in documents:
            title = doc.get('title', '')
            body = doc.get('body', '')
            # Weight title heavily in semantic text representation
            combined = f"Title: {title}. Abstract: {body}"
            corpus_texts.append(combined)

        if self.is_neural and self.model is not None:
            # Batch encode with progress bar
            raw_embeddings = self.model.encode(corpus_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = (raw_embeddings / norms).astype(np.float32)
        else:
            self.embeddings = self._train_lsa_fallback(documents)
            
        self.doc_ids = current_doc_ids
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.embedding_dim = self.embeddings.shape[1]
        
        # Save to disk for instant reuse
        np.save(emb_file, self.embeddings)
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f, indent=2)
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': self.model_name,
                'is_neural': self.is_neural,
                'num_docs': N,
                'embedding_dim': self.embedding_dim
            }, f, indent=2)
            
        print(f"Saved semantic embeddings to {emb_file} ({N} documents, dim={self.embedding_dim}).")

    def get_all_similarities(self, query_text):
        """Computes dense cosine similarity against ALL documents in one matrix operation."""
        if not query_text or self.embeddings is None:
            return {}
        q_vec = self.encode_text(query_text)
        sims = np.dot(self.embeddings, q_vec)
        sims = np.clip(sims, 0.0, 1.0)
        return {doc_id: float(sims[idx]) for idx, doc_id in enumerate(self.doc_ids)}

    def score_document(self, query_text, doc_id, q_vec=None):
        """Computes semantic cosine similarity for a single document."""
        if doc_id not in self.doc_id_to_idx or self.embeddings is None:
            return 0.0, {}
        idx = self.doc_id_to_idx[doc_id]
        doc_vec = self.embeddings[idx]
        if q_vec is None:
            q_vec = self.encode_text(query_text)
        
        sim = float(np.dot(doc_vec, q_vec))
        score = max(0.0, sim)
        
        band = "Strong Semantic Alignment" if score >= 0.65 else ("Moderate Semantic Alignment" if score >= 0.45 else "Thematic Overlap")
        explanation = {
            'semantic_score': score,
            'alignment_band': band,
            'model_name': self.model_name,
            'dimension': self.embedding_dim
        }
        return score, explanation

    def get_top_k(self, query_text, k=10):
        """
        Retrieves Top-K documents using vector dot product over pre-normalized matrix.
        Time complexity: O(N * D + N log K).
        """
        if not query_text or self.embeddings is None or len(self.doc_ids) == 0:
            return [], {}
            
        q_vec = self.encode_text(query_text)
        
        # Matrix-vector dot product for all documents simultaneously
        # Shape: (N, D) @ (D,) -> (N,)
        similarities = np.dot(self.embeddings, q_vec)
        
        # Non-negative normalization for ranking
        similarities = np.clip(similarities, 0.0, 1.0)
        
        # Efficient Top-K using partition / argpartition
        k = min(k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        # Sort top K descending
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        top_k = []
        all_explanations = {}
        for idx in top_indices:
            score = float(similarities[idx])
            doc_id = self.doc_ids[idx]
            band = "Strong Semantic Alignment" if score >= 0.65 else ("Moderate Semantic Alignment" if score >= 0.45 else "Thematic Overlap")
            top_k.append((score, doc_id))
            all_explanations[doc_id] = {
                'semantic_score': score,
                'alignment_band': band,
                'model_name': self.model_name,
                'dimension': self.embedding_dim
            }
            
        return top_k, all_explanations
