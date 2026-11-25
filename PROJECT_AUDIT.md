# Comprehensive Project Audit: Explainable Academic Search Engine

**Author:** Hemanth Grandhi (Roll No.: 2310110121)  
**Course:** CSD358: Information Retrieval, Final Semester Project  
**Repository:** https://github.com/hemanth-grandhi/IR  
**Date of Audit:** November 19, 2026  

---

## 1. Executive Summary

This audit establishes the technical foundation, verified baseline performance, and transition roadmap for upgrading the mid-term Information Retrieval project (*"Explainable Academic Search using Inverted Index, TF-IDF, Cosine Similarity, and Zone Weighting"*) into a comprehensive final-semester research project:

> **"Explainable Hybrid Academic Search Engine Using BM25, Semantic Retrieval, Hybrid Ranking, and Query Expansion"**

The audit evaluates all existing source code, verifies execution and reproducibility, checks unit tests and baseline evaluations, documents empirical limitations, and establishes a phased engineering plan.

---

## 2. Existing Architecture & Source Code Inventory

### 2.1 Codebase File Map

| Component File | Role / Responsibility | Key Classes / Functions | Status |
| :--- | :--- | :--- | :--- |
| `src/preprocessing.py` | Tokenization, case folding, stopword filtering, Porter stemming | `Preprocessor.process(text)` | Verified Working |
| `src/index.py` | In-memory inverted index, title/body term frequencies, postings lists | `InvertedIndex`, `add_document`, `get_postings`, `get_document_frequency` | Verified Working |
| `src/tfidf.py` | Term frequency (log-scale), Inverse Document Frequency ($log_{10}(N/df)$), Vector space representation (ltc.ltc & lnc.ltc), L2 normalization | `TFIDFScorer`, `_compute_tf`, `_compute_idf`, `build_vectors`, `get_query_vector` | Verified Working |
| `src/ranking.py` | Dot product, cosine similarity, zone weighting (title vs. body/abstract), min-heap Top-K selection | `Ranker.score_document`, `Ranker.get_top_k` | Verified Working |
| `src/jaccard.py` | Set intersection over union baseline | `JaccardBaseline.get_top_k` | Verified Working |
| `src/explanation.py` | Decomposition of cosine similarity into per-term contributions and zone provenance | `Explainer.format_explanation` | Verified Working |
| `src/fetch_dataset.py` | arXiv API harvester (Atom XML parsing) | `fetch_arxiv_papers` | Verified Working |
| `src/evaluation.py` | Automated evaluation comparing 6 retrieval variants across P@5, Recall, Precision, F1 | `evaluate_system`, `load_queries_and_qrels`, `BooleanAndBaseline`, `BooleanOrBaseline` | Verified Working |
| `src/main.py` | Interactive CLI with `--debug` and `--smart` flags | `main`, `print_debug_info` | Verified Working |
| `src/app.py` | Streamlit interactive search UI with query input and explainability view | `load_ir_system`, Streamlit widgets | Verified Working |
| `tests/test_pipeline.py` | Unit test suite covering indexing, TF, IDF, cosine similarity, and Jaccard | `TestIRPipeline` (6 tests) | 6/6 PASS |

### 2.2 Dataset Structure & Invariants

* **Current File:** `data/arxiv_corpus.json`
* **Corpus Size:** 300 academic research papers collected from arXiv CS categories (Machine Learning, Distributed Computing, Information Retrieval, Databases).
* **Record Schema:**
  ```json
  {
    "document_id": "string (arXiv identifier, e.g., '2504.05181v2')",
    "title": "string (clean single line)",
    "body": "string (abstract text)",
    "authors": ["string", ...],
    "year": "string (YYYY)",
    "source": "arXiv"
  }
  ```
* **Vocabulary Size:** 3,938 unique stems indexed across title and abstract zones.

---

## 3. Baseline Experimental Reproduction

The test suite and evaluation script were executed in the environment (Python 3.12.10, NLTK 3.10.4, Matplotlib 3.11.2, Scikit-learn 1.9.1). The baseline metrics were reproduced identically to the mid-term report:

### Verified Baseline Metrics (14 Evaluated Queries, $K=5$)

| Retrieval System Variant | Precision@5 | Recall | Overall Precision | F1-Score | Ranking Latency (avg/q) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Boolean AND Baseline** | 0.4714 | 0.5125 | 0.6036 | 0.4921 | ~1.2 ms |
| **Boolean OR Baseline** | 0.2286 | 0.2901 | 0.2286 | 0.1792 | ~2.1 ms |
| **Jaccard Baseline** | 0.4571 | 0.5687 | 0.4571 | 0.4068 | ~5.8 ms |
| **TF-IDF + Cosine Similarity** | **0.5000** | **0.6416** | **0.5000** | **0.4347** | ~3.4 ms |
| **TF-IDF + Zone Weighting (0.7 / 0.3)** | 0.4429 | 0.5416 | 0.4429 | 0.3653 | ~4.1 ms |
| **SMART `lnc.ltc` + Zone Weighting** | 0.4143 | 0.4767 | 0.4143 | 0.3197 | ~3.9 ms |

### Empirical Analysis of Baseline Results:
1. **Vector Space Dominance:** Standard Vector Space Retrieval (TF-IDF + Cosine similarity) achieved the highest overall score among baselines ($P@5=0.5000, Recall=0.6416, F1=0.4347$), outperforming discrete set-based matching (Boolean AND, Boolean OR, and Jaccard).
2. **Zone Weighting Penalty:** Overweighting titles ($0.7$ title / $0.3$ body) reduced recall from $0.6416$ to $0.5416$, because relevant papers frequently mention specific multi-word query concepts in their abstracts while employing concise, high-level titles.
3. **Log TF vs Document IDF:** SMART `lnc.ltc` (document without IDF weighting) underperformed standard `ltc.ltc` because rare, highly discriminating terms in the document corpus lost their relative importance.

---

## 4. Architectural Limitations of Mid-Term Implementation

1. **Vocabulary Mismatch & Synonym Blindness:**
   - The inverted index matches exact morphological stems (Porter stemmer). Queries using synonyms or abbreviations (e.g., *"ML"* vs *"machine learning"*, *"NLP"* vs *"natural language processing"*, *"LLM"* vs *"large language model"*) fail to retrieve highly relevant papers lacking the exact literal tokens.
2. **Absence of BM25 Probabilistic Ranking:**
   - Standard TF-IDF log-weighting lacks asymptotic term-frequency saturation and document-length normalization. Longer papers with repeated terms are disproportionately favored over dense, concise papers.
3. **No Dense Semantic Vector Search:**
   - The system possesses no conceptual understanding of semantics, sentence context, or embedding similarities.
4. **No Query Expansion:**
   - Short user queries (1–3 terms) suffer from low recall because no automatic pseudo-relevance feedback or lexical expansion is performed.
5. **Heuristic Evaluation (Pseudo-Ground Truth):**
   - The mid-term evaluation generated relevance judgements by string substring containment in titles/bodies, which introduces pseudo-relevance bias.
6. **Small Corpus Scale:**
   - 300 papers is adequate for proof-of-concept, but does not provide adequate distributional diversity for challenging statistical IR experiments.
7. **Basic UI & Missing Analytics:**
   - The Streamlit interface lacks multi-algorithm side-by-side comparison, parameter sliders ($k_1, b, \alpha$), query expansion toggles, evaluation dashboards, and filter facets (categories, publication years).

---

## 5. Upgrade Plan & Final Project Roadmap

| Phase | Enhancement | Target Module | Scope & Method |
| :---: | :--- | :--- | :--- |
| **Phase 1** | Project Audit & Baseline Reproduction | `PROJECT_AUDIT.md` | Audit codebase, record baselines, document roadmap. |
| **Phase 2** | Probabilistic BM25 Retrieval | `src/bm25.py` | Implement Robertson-Spärck Jones BM25 with tunable $k_1 \in [1.2, 2.0]$ and $b \in [0.5, 0.85]$, length normalization, and term saturation. |
| **Phase 3** | Dense Semantic Search | `src/semantic_search.py` | Pretrained Sentence-Transformers (`all-MiniLM-L6-v2`) with dense cosine similarity, offline embedding persistence (`embeddings/`), and graceful LSA/SVD fallback. |
| **Phase 4** | Hybrid Retrieval Engine | `src/hybrid_search.py` | Convex linear combination $S_{hybrid} = \alpha S_{BM25}^{norm} + (1-\alpha) S_{semantic}^{norm}$ with min-max and rank-based normalization. |
| **Phase 5** | Query Expansion & Unified Explainability | `src/query_expansion.py`, `src/explanation.py` | WordNet synset expansion + Rocchio Pseudo-Relevance Feedback (PRF); multi-modal explanation breaking down BM25, semantic alignment, and zone provenance. |
| **Phase 6** | Dataset Expansion & Curation | `src/fetch_dataset.py`, `data/` | Scale dataset to 1,500+ curated arXiv papers across 8 CS categories with deduplication and metadata schema validation. |
| **Phase 7** | Modern University Demonstration UI | `src/app.py` | Professional Streamlit UI with algorithm switching, parameter controls, side-by-side comparative retrieval, facet filtering, and interactive evaluation charts. |
| **Phase 8** | Rigorous IR Evaluation Suite | `src/evaluation.py` | Benchmark all 10 systems across P@5, P@10, R@5, R@10, MAP, MRR, NDCG@10, F1, runtime latency, parameter sensitivity, and ablation studies. |
| **Phase 9** | Testing & Code Quality | `tests/` | Comprehensive test suite (>15 unit & integration tests) covering all algorithms and edge cases. |
| **Phase 10** | Final Research Paper & Documentation | `reports/`, `DEMO_GUIDE.md`, etc. | Complete 15-section academic research paper in PDF format, viva prep guide, demo walkthrough, changelog, and packaging into submission ZIP. |
