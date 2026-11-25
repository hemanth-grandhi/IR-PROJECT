# Academic AI-Use Declaration

**Student Name:** Hemanth Grandhi  
**Roll No.:** 2310110121  
**Course Code:** CSD358: Information Retrieval  
**Project Title:** Explainable Hybrid Academic Search Engine Using BM25, Semantic Retrieval, Hybrid Ranking, and Query Expansion  
**Date:** November 19, 2026  

---

## 1. Compliance Statement

In adherence to university academic integrity guidelines, ethical research practices, and the CSD358 project instructions, this declaration documents the transparent and responsible utilization of AI coding assistants during the design, development, and evaluation of this research project.

All theoretical concepts, mathematical formulations, architectural designs, experimental hypotheses, and analytical conclusions were verified, tested, and validated by the author.

---

## 2. Tools & Models Consulted

- **Primary AI Assistant:** Google DeepMind Gemini (via Antigravity Autonomous Coding Agent)
- **Supplementary Tools:** GitHub Copilot / LLM Code Completion

---

## 3. Scope of AI Assistance

### 3.1 Mid-Term Foundation (Version 1.0)
1. **Repository Scaffolding:** Assisting with repository directory layout, basic classes, and CLI argument parsing.
2. **Classical IR Formulas:** Verifying standard textbook formulas ($1 + \log_{10}(tf)$, $\log_{10}(N/df)$, vector norm calculation) against lecture slides.
3. **API Scraper:** Drafting the initial arXiv Atom XML parsing function (`fetch_dataset.py`).
4. **Unit Test Scaffolding:** Writing standard `unittest` test skeletons.

### 3.2 Final Semester Research Upgrades (Version 2.0)
1. **Mathematical Implementation of BM25:**
   - Drafting the Robertson-Spärck Jones non-negative smoothed IDF formula and Okapi BM25 term saturation and length normalization functions in `src/bm25.py`.
2. **Dense Vector Search Engineering:**
   - Structuring `src/semantic_search.py` with Sentence-Transformer embeddings, offline `.npy` disk persistence, and a dense TruncatedSVD LSA fallback engine.
3. **Convex Hybrid Fusion & Normalization:**
   - Implementing min-max score normalization and convex blending ($S_{hybrid} = \alpha S_{BM25} + (1-\alpha) S_{semantic}$) in `src/hybrid_search.py`.
4. **Multi-Strategy Query Expansion:**
   - Integrating NLTK WordNet synsets and curated academic CS domain acronym mappings in `src/query_expansion.py`.
5. **Experimental Framework & Statistical Evaluation:**
   - Assisting with the implementation of P@K, R@K, F1@K, MAP, MRR, and NDCG@10 metrics in `src/evaluation.py` and automated Matplotlib plotting routines.
6. **ReportLab Automated Typesetting:**
   - Generating Python typesetting scripts to produce the formal university research paper in IEEE-compliant PDF format.

---

## 4. Human Verification & Original Contributions

- **Mathematical Soundness:** The author independently verified that all IR formulas (BM25 saturation, L2 vector norms, Robertson-Spärck Jones IDF, reciprocal rank fusion, discounted cumulative gain) are mathematically rigorous and adhere to peer-reviewed literature.
- **Empirical Authenticity:** No experimental metrics or benchmark numbers were fabricated or hallucinated. All 10 systems were genuinely executed against the 1,500-paper corpus, and the raw performance results were exported directly to disk (`results/system_comparison.csv`, `results/comprehensive_evaluation_results.json`).
- **Explainability Architecture:** The term-level attribution and multi-paradigm score breakdown framework was designed specifically for this research project.
- **Critical Review:** All code changes, unit tests, and documentation were reviewed, refactored, and tested to ensure zero regressions in legacy functionality.
