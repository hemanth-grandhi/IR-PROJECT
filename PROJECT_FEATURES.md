# Comprehensive Project Features: Explainable Hybrid Academic Search Engine

**Author:** Hemanth Grandhi (Roll No.: 2310110121)  
**Course:** CSD358: Information Retrieval  
**System Architecture:** Explainable Hybrid Scientific Retrieval Engine  

---

## 1. Feature Overview & Architecture

The upgraded engine transforms a classical TF-IDF mid-term hackathon project into an advanced, university-grade research retrieval system combining **Classical Lexical Search**, **Probabilistic BM25**, **Dense Semantic Retrieval**, **Convex Hybrid Fusion**, and **Multi-Strategy Query Expansion**.

```text
                                  +---------------------------------------+
                                  |            User Query (q)             |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     Query Expansion Module            |
                                  |  - Academic CS Domain Synonyms        |
                                  |  - NLTK WordNet Synsets               |
                                  |  - Pseudo-Relevance Feedback (PRF)    |
                                  +---------------------------------------+
                                         /                         \
                                        /                           \
                                       v                             v
                       +-------------------------------+   +-------------------------------+
                       |      Lexical Retrieval        |   |       Semantic Retrieval      |
                       | - Inverted Index & Postings   |   | - Sentence-Transformer        |
                       | - TF-IDF / SMART lnc.ltc      |   |   (all-MiniLM-L6-v2, 384d)    |
                       | - Okapi BM25 (k1, b)          |   | - Dense LSA TruncatedSVD      |
                       | - Title & Abstract Zones      |   |   (128d Fallback Engine)      |
                       +-------------------------------+   +-------------------------------+
                                       \                             /
                                        \                           /
                                         v                         v
                                  +---------------------------------------+
                                  |    Convex Hybrid Ranking Fusion       |
                                  | S_hybrid = a*S_lex^norm + (1-a)*S_sem |
                                  |    (Min-Max Dynamic Normalization)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Unified Explainability Layer        |
                                  | - Keyword Saturation Attribution      |
                                  | - Title vs. Body Zone Provenance      |
                                  | - Dense Embedding Alignment Band      |
                                  | - Ranking Driver Diagnosis            |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Modern Streamlit Web UI / Python CLI  |
                                  +---------------------------------------+
```

---

## 2. Detailed Technical Feature Breakdown

### Feature 1: Probabilistic BM25 Retrieval Engine (`src/bm25.py`)
- **Robertson-Spärck Jones IDF:** Non-negative smoothed formulation ($IDF(t) = \ln(1 + \frac{N - df(t) + 0.5}{df(t) + 0.5})$) preventing negative weights on high-frequency terms.
- **Term Frequency Saturation:** Tunable $k_1 \in [0.5, 3.0]$ (default $k_1=1.5$). As raw TF increases, marginal score gains decrease asymptotically, preventing term spamming.
- **Document-Length Normalization:** Tunable parameter $b \in [0.0, 1.0]$ (default $b=0.75$) penalizing excessively verbose documents using real document token counts recorded during indexing.
- **Field Zone Weighting:** Separate BM25 scoring for title and abstract zones ($w_{title} = 0.6, w_{body} = 0.4$).
- **Term-Level Attribution:** Decomposes final document score into individual keyword contributions, raw frequencies, and IDF values.

### Feature 2: Dense Semantic Search (`src/semantic_search.py`)
- **Pretrained Transformer Representation:** Uses `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
- **Offline Persistence:** Embeddings are generated once and persisted in `embeddings/corpus_embeddings_all-MiniLM-L6-v2.npy` and `embeddings/doc_ids_all-MiniLM-L6-v2.json`. On subsequent queries or system restarts, embeddings load in $<10$ ms.
- **Fast Vector Dot Product:** Since embeddings and query vectors are L2-normalized ($\|v\|_2 = 1.0$), cosine similarity is computed simultaneously across all 1,500 documents via a single matrix-vector product ($O(N \cdot D)$) in $<1$ ms.
- **Dual Architecture Fallback:** If PyTorch or internet access is unavailable, the engine automatically falls back to a 128-dimensional TruncatedSVD Latent Semantic Analysis (LSA) dense representation with zero external model dependencies.

### Feature 3: Convex Hybrid Retrieval (`src/hybrid_search.py`)
- **Linear Convex Fusion:**
  $$\text{HybridScore}(d, q) = \alpha \cdot S_{BM25}^{\text{norm}}(d, q) + (1 - \alpha) \cdot S_{\text{semantic}}^{\text{norm}}(d, q)$$
  where $\alpha \in [0.0, 1.0]$ is dynamically adjustable by the user.
- **Min-Max Score Normalization:** Re-scales BM25 scores (typically $0 - 25+$) and semantic cosine similarities ($0 - 1$) onto a unified, commensurate interval $[0.0, 1.0]$:
  $$S^{\text{norm}}(d) = \frac{S(d) - \min_D S}{\max_D S - \min_D S + \epsilon}$$
- **Reciprocal Rank Fusion (RRF):** Implements an alternative rank-based fusion algorithm:
  $$RRF(d) = \frac{1}{60 + r_{lex}(d)} + \frac{1}{60 + r_{sem}(d)}$$
- **Ranking Driver Classification:** Analyzes the relative weight of components and diagnoses whether the document was ranked primarily by *Exact Lexical Keyword Saturation*, *Dense Semantic Proximity*, or *Balanced Synergy*.

### Feature 4: Multi-Strategy Query Expansion (`src/query_expansion.py`)
- **WordNet Synset Synonyms:** Utilizes NLTK WordNet lemmas to expand queries with linguistically valid synonyms.
- **Curated Academic CS Acronyms:** Expands domain shorthand (e.g., *"ML"* $\to$ *"machine learning"*, *"IR"* $\to$ *"information retrieval"*, *"GNN"* $\to$ *"graph neural networks"*, *"LLM"* $\to$ *"large language models"*).
- **Pseudo-Relevance Feedback (PRF):** Mines salient, high-frequency co-occurring terms from the top-3 documents retrieved in an initial first-pass retrieval run.
- **Interactive Control:** Can be enabled, disabled, or configured via CLI (`--expand`) and the Streamlit sidebar.

### Feature 5: Transparent Explainability Layer (`src/explanation.py`)
- **Backward Compatibility:** Preserves the original `format_explanation` function for TF-IDF / Cosine similarity.
- **BM25 Attribution:** Explains term saturation factor, IDF, and title vs body match.
- **Semantic Attribution:** Explains semantic cosine similarity, vector dimension, and alignment tier (Strong, Moderate, Contextual).
- **Hybrid Breakdown:** Explains the linear fusion formula, raw and normalized scores, and ranking driver.
- **No Hallucinated Explanations:** Every displayed explanation derives directly from underlying mathematical calculations.

### Feature 6: Scaled Academic Dataset (`src/fetch_dataset.py`, `data/arxiv_corpus.json`)
- **1,500 Academic Papers:** Harvested from arXiv across 18 specialized computer science domains.
- **Rich Schema:** Contains `document_id`, `title`, `body` (abstract), `authors`, `year`, `categories`, and `abstract_url`.
- **Validation & Deduplication:** Removes duplicates by ID and title, filters out incomplete records, and cleans text formatting.

### Feature 7: University-Grade Streamlit Application (`src/app.py`)
- **4 Interactive Navigation Tabs:**
  1. *Search & Explainability:* Instant search with expandable mathematical explanations and component bar charts.
  2. *Multi-Algorithm Comparison:* Side-by-side comparison of 4 algorithms for the same query.
  3. *Evaluation Dashboard:* Live leaderboard, metric cards, alpha sensitivity curves, and ablation tables.
  4. *Search History:* Session audit log.
- **Interactive Controls:** Algorithm dropdown, hybrid alpha slider, BM25 $k_1$ and $b$ sliders, category multiselect, year slider, Top-K selector, and IR debug inspector.

### Feature 8: Advanced Evaluation Suite (`src/evaluation.py`)
- Benchmarks 10 systems across 19 queries and 1,500 documents.
- Evaluates P@5, P@10, R@5, R@10, F1@5, F1@10, MAP, MRR, and NDCG@10.
- Alpha optimization sweep and BM25 component ablation studies.
- Automatically generates high-resolution academic plots in `results/`.
