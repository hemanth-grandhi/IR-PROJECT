# University Viva Voce & Oral Defense Preparation Guide

**Course:** CSD358: Information Retrieval  
**Student Name:** Hemanth Grandhi (Roll No.: 2310110121)  
**Project Title:** Explainable Hybrid Academic Search Engine Using BM25, Semantic Retrieval, Hybrid Ranking, and Query Expansion  

---

## Category 1: Classical IR, Inverted Indexes & Postings

### Q1: What is an Inverted Index and why is it preferred over a forward index for search?
**Answer:**  
A forward index maps documents to the list of words they contain ($D \to \{w_1, w_2, \dots\}$). To answer a query across $N$ documents using a forward index, one would have to scan every document sequentially ($O(N)$ linear scan), which is prohibitively slow for millions of documents.  
An Inverted Index reverses this mapping by indexing terms to the list of documents containing them ($w \to \{d_1, d_2, \dots\}$), known as a **postings list**. Query processing reduces to intersecting or merging the postings lists of the query terms, achieving $O(\sum |L_i|)$ lookup time where $|L_i| \ll N$. In our project, `InvertedIndex` in `src/index.py` stores postings with field-specific term frequencies (`title_tf` and `body_tf`).

### Q2: What preprocessing steps did you implement, and what is the difference between Stemming and Lemmatization?
**Answer:**  
Our `Preprocessor` (`src/preprocessing.py`) executes four sequential stages:
1. **Case Folding:** Lowercasing all characters to remove case sensitivity.
2. **Punctuation Stripping & Tokenization:** Splitting on whitespace and punctuation boundaries.
3. **Stop-word Removal:** Filtering out high-frequency function words (e.g., *"the"*, *"is"*, *"and"*) that carry negligible discriminating power.
4. **Stemming:** Applying Porter's Stemming algorithm.  
*Difference:* Stemming is a heuristic, rule-based process that chops off affixes (prefixes/suffixes) to produce a base stem (e.g., *"retrieval"* $\to$ *"retriev"*, which may not be a valid linguistic word). Lemmatization uses morphological vocabulary analysis and vocabulary synsets (such as WordNet) to return the grammatically valid dictionary base form known as the *lemma* (e.g., *"better"* $\to$ *"good"*).

---

## Category 2: Vector Space Model, TF-IDF & SMART Weighting

### Q3: Why do we use Log-Frequency TF ($1 + \log_{10}(tf)$) instead of raw term frequency?
**Answer:**  
Raw term frequency implies that a document mentioning a term 20 times is 20 times more relevant than a document mentioning it once. In real text, relevance does not scale linearly with term repetition; the twenty-first occurrence conveys significantly less incremental relevance than the first. The sublinear formulation $w_{t,d} = 1 + \log_{10}(tf)$ for $tf > 0$ introduces diminishing marginal utility while preserving monotonic ordering.

### Q4: Explain the SMART notation `lnc.ltc` used in your project.
**Answer:**  
The SMART notation (developed by Gerard Salton at Cornell) specifies weighting schemes using a 3-letter triplet for Document and Query representations: `[term frequency].[document frequency].[normalization]`.
In `lnc.ltc`:
- **Document (`lnc`):**
  - `l`: Logarithmic term frequency ($1 + \log_{10}(tf)$).
  - `n`: No IDF weighting ($idf = 1$).
  - `c`: Cosine length normalization ($\frac{1}{\sqrt{\sum w_i^2}}$).
- **Query (`ltc`):**
  - `l`: Logarithmic term frequency.
  - `t`: Standard IDF weighting ($\log_{10}(N / df)$).
  - `c`: Cosine length normalization.  
*Rationale:* Omitting IDF from the document (`lnc`) avoids double-penalizing terms that occur frequently in both document and corpus, while retaining query discriminative power.

---

## Category 3: Probabilistic Retrieval & Okapi BM25

### Q5: What is Okapi BM25, and what two major advantages does it hold over TF-IDF?
**Answer:**  
Okapi BM25 (Robertson et al.) is a non-linear probabilistic ranking function derived from the 2-Poisson indexing model. Its score formula is:
$$BM25(D, Q) = \sum_{q \in Q} IDF(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
BM25 provides two fundamental advantages over classical TF-IDF:
1. **Asymptotic Term Frequency Saturation ($k_1$):** In TF-IDF, log-frequency still grows without an upper bound as $tf \to \infty$. In BM25, as $tf \to \infty$, the term factor asymptotically converges to an absolute upper bound of $(k_1 + 1)$, completely neutralizing keyword stuffing.
2. **Document Length Normalization ($b$):** It differentiates between documents that are long because they cover multiple distinct topics (verbose) versus documents that simply repeat words. Documents longer than the corpus average ($|D| > \text{avgdl}$) are penalized by factor $(1 - b + b \cdot \frac{|D|}{\text{avgdl}})$.

### Q6: What was your finding in the BM25 Ablation Study?
**Answer:**  
When we ablated term frequency saturation ($k_1 = 100.0$, scaling linearly), NDCG@10 dropped precipitously from **0.9446 to 0.7121 (-24.6%)** and MAP dropped from **0.2614 to 0.1422 (-45.6%)**. When length normalization was removed ($b = 0$), NDCG dropped to 0.9365. This proves mathematically that **term saturation is the single most vital component of the BM25 model**.

---

## Category 4: Dense Semantic Retrieval & Sentence Transformers

### Q7: What is the "Vocabulary Mismatch Problem" and how does Dense Semantic Search solve it?
**Answer:**  
The vocabulary mismatch problem arises in lexical retrieval (TF-IDF, BM25) because documents and queries must share identical morphological tokens. If a user searches for *"deep neural architectures"* but a relevant paper uses the phrase *"convolutional networks"*, lexical search cannot match them.  
Dense Semantic Search (`src/semantic_search.py`) maps text into a continuous latent vector space using a pretrained Sentence Transformer (`all-MiniLM-L6-v2`). The model places semantically related concepts close together in 384-dimensional space regardless of exact wording. Cosine similarity $\frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2}$ measures conceptual alignment directly.

### Q8: How did you ensure fast querying without recomputing embeddings repeatedly?
**Answer:**  
We implemented an offline embedding persistence pipeline. On first initialization, the 1,500 documents are encoded in batches of 32 and persisted to disk in `embeddings/corpus_embeddings_all-MiniLM-L6-v2.npy` and `embeddings/doc_ids_all-MiniLM-L6-v2.json`. On subsequent queries or app reboots, the pre-normalized matrix is loaded in $<10$ ms. Query execution requires encoding only the single query vector ($\approx 15$ ms) and executing a single matrix-vector product ($O(N \cdot D)$) that evaluates all 1,500 documents in $<1$ ms.

---

## Category 5: Convex Hybrid Fusion & Query Expansion

### Q9: Why is score normalization essential in Hybrid Search, and what formula did you use?
**Answer:**  
BM25 scores and Semantic Cosine similarities inhabit completely different mathematical scales:
- BM25 scores are unbounded positive numbers typically ranging from $0$ to $25+$.
- Semantic cosine similarities are bounded between $0.0$ and $1.0$.  
Adding them directly would cause BM25 to completely overpower the semantic signal. We implemented dynamic Min-Max Normalization over the candidate pool:
$$S^{\text{norm}}(d) = \frac{S(d) - \min_D S}{\max_D S - \min_D S + \epsilon}$$
The normalized scores are then combined via convex linear blending:
$$\text{HybridScore}(d, q) = \alpha \cdot S_{BM25}^{\text{norm}}(d, q) + (1 - \alpha) \cdot S_{\text{semantic}}^{\text{norm}}(d, q)$$

### Q10: What are the trade-offs of Query Expansion?
**Answer:**  
Query expansion improves **Recall** by appending synonyms, domain acronyms, or pseudo-relevance feedback terms, enabling the retrieval of papers that lack the user's initial literal terms. However, on already specific multi-word queries, adding too many synsets causes **Query Drift**, where the search focus shifts to peripheral topics, slightly decreasing Precision@5. Our system mitigates this by restricting expansion to top-$3$ high-confidence domain terms and providing an interactive user toggle.

---

## Category 6: Evaluation Metrics & Scientific Rigor

### Q11: Explain MAP, MRR, and NDCG@10. Which is best suited for academic literature search?
**Answer:**  
- **MRR (Mean Reciprocal Rank):** $\frac{1}{|Q|} \sum \frac{1}{\text{rank}_1}$. Evaluates where the *very first* relevant document occurs. Ideal for question-answering or navigation where the user seeks a single correct answer.
- **MAP (Mean Average Precision):** Evaluates precision across *all* relevant documents at their respective ranks. Rewards systems that return all relevant documents near the top.
- **NDCG@10 (Normalized Discounted Cumulative Gain):** Evaluates graded or binary relevance with logarithmic position discounting ($\frac{1}{\log_2(\text{rank} + 1)}$), normalized against the ideal ranking ($IDCG$).  
*Suitability:* **NDCG@10** is best suited for academic literature search because researchers primarily inspect the top-10 papers on page 1, and relevant papers placed at rank #1 or #2 are far more valuable than those at rank #9 or #10.

---

## Category 7: Explainable Information Retrieval (XIR)

### Q12: Why is explainability critical in academic search, and how does your system explain rankings?
**Answer:**  
Researchers frequently distrust "black-box" neural ranking models because it is impossible to know why a particular paper was recommended over another. Our `Explainer` (`src/explanation.py`) provides authentic, mathematically grounded attribution:
1. Decomposes hybrid scores into percentage contribution from Lexical BM25 vs. Dense Semantic embeddings.
2. Identifies whether the match occurred in the **Title** or the **Abstract**.
3. Diagnoses the **Primary Ranking Driver** (*Lexical Keyword Match*, *Dense Semantic Alignment*, or *Synergy*).
4. Lists exact keyword weights without fabricating explanations.
