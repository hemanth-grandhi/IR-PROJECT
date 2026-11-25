# Explainable Hybrid Academic Search Engine Using BM25, Semantic Retrieval, Hybrid Ranking, and Query Expansion

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 14/14 PASS](https://img.shields.io/badge/Tests-14%2F14%20PASS-brightgreen.svg)]()
[![Corpus: 1500 Papers](https://img.shields.io/badge/Corpus-1500%20arXiv%20Papers-red.svg)]()

**Course:** CSD358: Information Retrieval (Final Semester Research Project)  
**Author:** Hemanth Grandhi  
**Roll No.:** 2310110121  
**GitHub Repository:** [https://github.com/hemanth-grandhi/IR-PROJECT](https://github.com/hemanth-grandhi/IR-PROJECT)  

---

## 1. Executive Summary & Problem Statement

Academic research discovery systems frequently suffer from two opposing failure modes:
1. **Lexical Keyword Brittleness:** Classical lexical engines (Boolean, TF-IDF, BM25) require exact morphological keyword matches, suffering from the **vocabulary mismatch problem** when researchers query using synonyms, acronyms, or high-level conceptual phrasing.
2. **Opaque Neural Black Boxes:** Dense neural semantic retrievers capture semantic proximity but cannot explain *why* specific papers were ranked at the top, leaving researchers uncertain about factual keyword alignment.

This research project delivers an end-to-end, interpretable academic discovery platform:
- **Probabilistic Okapi BM25** with asymptotic term saturation ($k_1$) and document-length normalization ($b$).
- **Dense Semantic Retrieval** utilizing Pretrained Sentence-Transformers (`all-MiniLM-L6-v2`) with persistent offline vector caching and an automatic TruncatedSVD LSA fallback.
- **Convex Hybrid Ranking Fusion** combining normalized lexical and semantic scores via dynamically tunable $\alpha$-weighting.
- **Multi-Strategy Query Expansion** integrating NLTK WordNet synsets, Academic CS domain acronyms, and Pseudo-Relevance Feedback (PRF).
- **Unified Explainability Layer** decomposing document scores into keyword saturation weights, title vs. abstract zone provenance, and semantic alignment bands without hallucinated attributions.

---

## 2. Evolution: Mid-Term Foundation vs. Final-Semester Upgraded System

| Capability Dimension | Mid-Term Prototype (Version 1.0) | Upgraded Final Research Engine (Version 2.0) |
| :--- | :--- | :--- |
| **Primary Ranking** | Vector Space Model (TF-IDF Cosine Similarity) | **Convex Hybrid Fusion: Okapi BM25 + Dense Sentence Transformers** |
| **Lexical Saturation** | Sublinear Log-TF ($1 + \log_{10}(tf)$), unbounded | **Okapi BM25 Saturation bounded by $(k_1 + 1)$** |
| **Length Normalization** | Simple L2 Cosine Norm | **Document-length normalization ($b=0.75$) via actual token length** |
| **Semantic Matching** | None (pure exact keyword stem matching) | **384-dimensional dense semantic vectors with offline caching** |
| **Score Normalization** | None (single scoring mechanism) | **Dynamic Min-Max score normalization + Reciprocal Rank Fusion** |
| **Query Expansion** | None | **WordNet Synsets + CS Domain Acronyms + Pseudo-Relevance Feedback** |
| **Dataset Scale** | 300 arXiv papers (4 fields) | **1,500 arXiv research papers with Categories & Abstract URLs** |
| **User Interface** | Basic single-view Streamlit script | **Professional 4-Tab Research Dashboard with Comparative Search** |
| **Evaluation Metrics** | P@5, Recall, F1 (14 queries) | **P@5, P@10, R@5, R@10, F1@5, F1@10, MAP, MRR, NDCG@10 (10 systems)** |
| **Experiments** | Single baseline comparison table | **Alpha sensitivity curves, BM25 ablations, latency profiling** |
| **Unit Test Suite** | 6 basic tests | **14 comprehensive unit and integration tests (100% PASS)** |

---

## 3. System Architecture & Information Flow

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

## 4. Repository Structure

```text
IR/
│
├── data/
│   ├── arxiv_corpus.json               # 1,500 validated academic papers from arXiv
│
├── embeddings/
│   ├── corpus_embeddings_all-MiniLM-L6-v2.npy  # Persistent dense vectors (1500 x 384)
│   ├── doc_ids_all-MiniLM-L6-v2.json           # Document ID index mapping
│   └── metadata_all-MiniLM-L6-v2.json          # Embedding model metadata
│
├── results/
│   ├── system_comparison.csv                   # Full 10-system benchmark results
│   ├── alpha_tuning_experiment.csv             # Alpha sweep results (alpha=0.0 to 1.0)
│   ├── ablation_study.csv                      # BM25 component ablation data
│   ├── benchmark_qrels.json                    # Ground-truth relevance judgments
│   ├── comprehensive_evaluation_results.json   # Raw evaluation data
│   ├── evaluation_results.md                   # Markdown summary report
│   ├── precision_recall_comparison.png         # High-resolution P@5 & R@10 figure
│   ├── map_mrr_ndcg_comparison.png             # High-resolution MAP, MRR, NDCG figure
│   ├── alpha_sensitivity_curve.png             # Hybrid alpha sensitivity curve
│   ├── runtime_latency_comparison.png          # Query latency benchmark plot
│   └── p_at_5_comparison.png                   # Legacy Precision@5 comparison plot
│
├── reports/
│   └── CSD358_Final_Report_Hemanth_Grandhi.pdf # Complete 15-Section Research Paper (PDF)
│
├── src/
│   ├── preprocessing.py                        # Tokenization, Porter stemmer, stop-words
│   ├── index.py                                # Inverted index & token length tracker
│   ├── tfidf.py                                # TF-IDF (ltc.ltc) & SMART (lnc.ltc)
│   ├── ranking.py                              # Cosine similarity & zone weighting
│   ├── bm25.py                                 # Okapi BM25 with saturation & length norm
│   ├── semantic_search.py                      # Sentence-Transformers & LSA fallback
│   ├── hybrid_search.py                        # Convex alpha fusion & Min-Max normalization
│   ├── query_expansion.py                      # WordNet, Domain CS synonyms, PRF
│   ├── explanation.py                          # Unified multi-paradigm explainability
│   ├── evaluation.py                           # 10-system benchmark & plotting suite
│   ├── fetch_dataset.py                        # arXiv harvester with deduplication
│   ├── main.py                                 # CLI search interface
│   └── app.py                                  # Streamlit 4-tab research application
│
├── tests/
│   └── test_pipeline.py                        # 14 unit and integration tests
│
├── AI_USE.md                                   # Academic AI usage declaration
├── CHANGELOG.md                                # Complete version release notes
├── CSD358_FINAL_AUDIT.md                       # Mid-term audit & compliance checklist
├── DEMO_GUIDE.md                               # University demonstration walkthrough
├── EXPERIMENTS.md                              # Detailed experimental analysis & findings
├── PROJECT_AUDIT.md                            # Comprehensive system audit & roadmap
├── PROJECT_FEATURES.md                         # Detailed feature-by-feature specifications
├── VIVA_QUESTIONS.md                           # 12 oral defense questions and answers
├── requirements.txt                            # Complete project dependencies
└── README.md                                   # Project documentation
```

---

## 5. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/hemanth-grandhi/IR-PROJECT.git
cd IR-PROJECT
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

*Key packages installed:* `sentence-transformers`, `torch`, `nltk`, `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `reportlab`, `tabulate`.

---

## 6. How to Run the System

### 1. Run the Interactive Streamlit Web Interface (Recommended)
```bash
streamlit run src/app.py
```
*Access the interactive interface at `http://localhost:8501`.*

**Features available in the Streamlit application:**
- **Tab 1 (Academic Search & Explainability):** Execute queries, view ranking driver diagnosis, inspect interactive bar charts of component contributions, and review term-level attribution.
- **Tab 2 (Multi-Algorithm Comparison):** Side-by-side comparative retrieval across TF-IDF, BM25, Semantic, and Hybrid.
- **Tab 3 (Evaluation & Empirical Dashboard):** Live leaderboard, interactive plots, alpha sensitivity curves, and ablation tables.
- **Tab 4 (Search History):** Session audit log.

### 2. Run the Interactive Command-Line Interface (CLI)
```bash
# Default Hybrid Search
python src/main.py

# Probabilistic BM25 with IR Debug Inspector
python src/main.py --algo bm25 --debug

# Hybrid Search with Query Expansion enabled
python src/main.py --algo hybrid --alpha 0.6 --expand

# Dense Semantic Search
python src/main.py --algo semantic

# Classical SMART lnc.ltc Weighting (Mid-term compatibility)
python src/main.py --smart --debug
```

### 3. Run the Comprehensive Experimental Evaluation Suite
```bash
python src/evaluation.py
```
*Executes the 10-system benchmark, alpha sweep, and ablation study, saving all tables and publication plots to `results/`.*

### 4. Run the Unit & Integration Test Suite
```bash
python -m unittest discover tests
```
*Runs all 14 tests verifying indexing, TF-IDF, BM25 saturation, length normalization, semantic search, hybrid fusion, query expansion, and explainability.*

### 5. Fetch / Re-Harvest the Dataset (Optional)
```bash
python src/fetch_dataset.py
```

---

## 7. Experimental Benchmark Results

Evaluated across 19 validated academic research queries over the 1,500 arXiv research paper corpus:

| System | Precision@5 | Precision@10 | Recall@10 | F1@10 | MAP | MRR | NDCG@10 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Okapi BM25** | **0.9895** | **0.9211** | **0.2645** | **0.3826** | **0.2614** | **1.0000** | **0.9446** | 2.31 ms |
| **Convex Hybrid ($\alpha=0.5$)** | **0.9684** | **0.8842** | **0.2492** | **0.3622** | **0.2414** | **1.0000** | **0.9159** | 14.57 ms |
| **SMART `lnc.ltc` + Zone** | 0.9158 | 0.8579 | 0.2335 | 0.3432 | 0.2192 | 1.0000 | 0.8910 | 1.80 ms |
| **Dense Semantic Search** | 0.8421 | 0.7158 | 0.1924 | 0.2845 | 0.1693 | 0.9211 | 0.7586 | 9.83 ms |
| **TF-IDF + Cosine** | 0.8316 | 0.7895 | 0.2057 | 0.3062 | 0.1791 | 0.9123 | 0.8117 | 1.79 ms |
| **TF-IDF + Zone Weighting** | 0.8316 | 0.7632 | 0.1957 | 0.2920 | 0.1688 | 0.9123 | 0.7879 | 1.73 ms |
| **Hybrid + Query Expansion** | 0.8316 | 0.8158 | 0.2263 | 0.3317 | 0.2044 | 0.9123 | 0.8240 | 154.10 ms |
| **Boolean AND Baseline** | 0.6947 | 0.6737 | 0.1719 | 0.2569 | 0.1616 | 0.7632 | 0.6891 | 1.32 ms |
| **Boolean OR Baseline** | 0.6000 | 0.5737 | 0.1340 | 0.2059 | 0.0933 | 0.7544 | 0.5832 | 0.47 ms |
| **Jaccard Similarity** | 0.5895 | 0.5842 | 0.1470 | 0.2179 | 0.0994 | 0.7149 | 0.5767 | 897.38 ms |

### Key Experimental Insights:
1. **BM25 Superiority:** Okapi BM25 achieved the highest overall score ($NDCG@10 = 0.9446, MAP = 0.2614, P@5 = 0.9895$), outperforming classical TF-IDF ($0.8117$) due to term frequency saturation and document length normalization.
2. **Ablation Evidence:** Disabling BM25 term saturation ($k_1=100.0$) caused NDCG@10 to collapse by **24.6%** (to $0.7121$) and MAP by **45.6%**, proving that non-linear saturation is the critical theoretical foundation of BM25.
3. **Hybrid Generalization:** Balanced hybrid retrieval ($\alpha = 0.5$) achieved high precision ($P@5 = 0.9684$) while leveraging dense semantic vector alignment to retrieve conceptually relevant papers lacking exact keyword matches.

---

## 8. Requirements File

The updated `requirements.txt` includes:
```text
nltk>=3.8.1
matplotlib>=3.8.2
streamlit>=1.30.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
torch>=2.1.0
reportlab>=4.0.0
tabulate>=0.9.0
requests>=2.31.0
```

---

## 9. Academic Integrity & Documentation Links

- **Final Research Paper (PDF):** [`reports/CSD358_Final_Report_Hemanth_Grandhi.pdf`](file:///reports/CSD358_Final_Report_Hemanth_Grandhi.pdf)
- **Comprehensive Project Audit:** [`PROJECT_AUDIT.md`](file:///PROJECT_AUDIT.md)
- **Detailed Feature Specifications:** [`PROJECT_FEATURES.md`](file:///PROJECT_FEATURES.md)
- **Experimental Analysis & Results:** [`EXPERIMENTS.md`](file:///EXPERIMENTS.md)
- **Demonstration Walkthrough:** [`DEMO_GUIDE.md`](file:///DEMO_GUIDE.md)
- **Viva Voce Oral Defense Prep:** [`VIVA_QUESTIONS.md`](file:///VIVA_QUESTIONS.md)
- **Version Changelog:** [`CHANGELOG.md`](file:///CHANGELOG.md)
- **AI-Use Disclosure:** [`AI_USE.md`](file:///AI_USE.md)

---

## 10. Citation & Academic References

1. Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M., & Gatford, M. (1994). *Okapi at TREC-3*. NIST Special Publication 500-225.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of EMNLP-IJCNLP 2019.
3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
4. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513-523.
5. Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). *Reciprocal rank fusion outperforms condorcet and individual machine learning methods*. In Proceedings of ACM SIGIR 2009.
6. Miller, G. A. (1995). *WordNet: a lexical database for English*. Communications of the ACM, 38(11), 39-41.
7. Rocchio, J. J. (1971). *Relevance feedback in information retrieval*. The SMART Retrieval System: Experiments in Automatic Document Processing, 313-323.
