# Changelog: Evolution from Mid-Term to Final Semester IR Research Engine

All notable technical upgrades, theoretical IR extensions, architectural expansions, and algorithmic milestones for **CSD358: Information Retrieval** are documented herein.

---

## [Version 2.0.0] - Final Semester Research Release (November 19, 2026)

### Added
- **Probabilistic BM25 Retrieval Engine (`src/bm25.py`):**
  - Robertson-Spärck Jones non-negative smoothed IDF formulation ($IDF(t) = \ln(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5})$).
  - Asymptotic term frequency saturation with tunable $k_1 \in [0.5, 3.0]$ (default $k_1 = 1.5$).
  - Document-length normalization with tunable parameter $b \in [0.0, 1.0]$ (default $b = 0.75$) using exact token lengths from `InvertedIndex.doc_lengths`.
  - Zone weighting support ($w_{title} = 0.6, w_{body} = 0.4$) for field-specific term saturation.
  - Term-level score attribution decomposing each query keyword into IDF, title/body TF, and normalized score.

- **Dense Semantic Retrieval System (`src/semantic_search.py`):**
  - Integration of Pretrained Neural Sentence Transformer (`all-MiniLM-L6-v2`, 384 dimensions) for dense conceptual embedding generation.
  - Offline vector persistence in `embeddings/corpus_embeddings_all-MiniLM-L6-v2.npy` and `embeddings/doc_ids_all-MiniLM-L6-v2.json` to eliminate redundant embedding computation during querying.
  - High-performance matrix-vector dot product ($O(N \cdot D)$) yielding full-corpus cosine similarity in $<10$ ms.
  - Automatic zero-dependency Latent Semantic Analysis (TruncatedSVD, 128 dimensions) fallback for resource-constrained or offline execution.

- **Convex Hybrid Retrieval Engine (`src/hybrid_search.py`):**
  - Convex score combination: $S_{hybrid}(d, q) = \alpha \cdot S_{lexical}^{norm}(d, q) + (1 - \alpha) \cdot S_{semantic}^{norm}(d, q)$ with tunable $\alpha \in [0.0, 1.0]$.
  - Dynamic Min-Max score normalization ensuring commensurate scale across disparate scoring distributions.
  - Reciprocal Rank Fusion (RRF) implementation ($RRF(d) = \frac{1}{60 + r_{lex}} + \frac{1}{60 + r_{sem}}$) as an alternative ranking strategy.
  - Unified explanation attribution tracking lexical vs. semantic component dominance and ranking drivers.

- **Multi-Strategy Query Expansion (`src/query_expansion.py`):**
  - WordNet synset synonym expansion using NLTK lemmas.
  - Curated academic Computer Science domain abbreviation and acronym expansion (e.g., "ML" $\to$ "machine learning", "NLP" $\to$ "natural language processing", "GNN" $\to$ "graph neural networks").
  - Pseudo-Relevance Feedback (PRF / Rocchio-style term mining) extracting salient co-occurring terms from top retrieved documents.

- **Dataset Scale & Enrichment (`src/fetch_dataset.py`, `data/arxiv_corpus.json`):**
  - Scaled dataset by 500% from 300 to **1,500 real academic research papers** harvested from the arXiv API.
  - Enriched metadata schema with `categories` (e.g., `cs.IR`, `cs.AI`, `cs.LG`, `cs.CV`, `cs.DC`) and canonical `abstract_url` links.
  - Automated XML parsing, title/abstract validation, missing-value sanitization, and strict deduplication.

- **Modern University Demonstration Web App (`src/app.py`):**
  - Upgraded Streamlit interface with 4 dedicated research tabs:
    1. *Academic Search & Explainability*: Interactive query search with visual component bar charts, LaTeX formulations, and metadata chips.
    2. *Multi-Algorithm Comparison*: Side-by-side comparative retrieval across TF-IDF, BM25, Semantic, and Hybrid.
    3. *Evaluation & Empirical Dashboard*: Live leaderboard, interactive plots, alpha sensitivity curve, and ablation tables.
    4. *Search History & Session Analytics*: Query audit log.
  - Facet filtering by arXiv category and publication year range (2010–2026).
  - Hyperparameter controls for BM25 ($k_1, b$) and Hybrid alpha ($\alpha$).

- **Comprehensive Evaluation & Benchmarking Suite (`src/evaluation.py`):**
  - Benchmarked 10 systems: Boolean AND, Boolean OR, Jaccard, TF-IDF + Cosine, TF-IDF + Zone, SMART lnc.ltc, BM25, Semantic, Hybrid, Hybrid + Query Expansion.
  - Evaluated on 19 validated queries against multi-judge relevance sets across P@5, P@10, R@5, R@10, F1@5, F1@10, MAP, MRR, and NDCG@10.
  - Empirical alpha sensitivity optimization sweep from $\alpha=0.0$ to $\alpha=1.0$.
  - BM25 ablation study verifying the mathematical necessity of saturation ($k_1$) and length normalization ($b$).
  - Exported raw CSV, JSON, and high-resolution publication figures to `results/`.

- **Comprehensive Test Suite (`tests/test_pipeline.py`):**
  - Expanded test suite to 14 unit and integration tests (100% pass rate).

---

## [Version 1.0.0] - Mid-Term Hackathon Prototype (October 2026)

- Initial prototype implementing Tokenization, Porter Stemmer, and Stopword removal (`src/preprocessing.py`).
- Inverted Index mapping terms to document title and body frequencies (`src/index.py`).
- Classical Log-TF, Document Frequency, and IDF Vector Space Model (`src/tfidf.py`).
- Cosine similarity with Title/Body Zone Weighting (`src/ranking.py`).
- Set-based Jaccard similarity baseline (`src/jaccard.py`).
- Term contribution explainability for cosine scores (`src/explanation.py`).
- Basic 300-paper arXiv harvester (`src/fetch_dataset.py`).
- Prototype evaluation script generating `results/p_at_5_comparison.png`.
- Baseline Streamlit app (`src/app.py`).
