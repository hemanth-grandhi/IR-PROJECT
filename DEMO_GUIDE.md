# University Demonstration Guide: Explainable Academic IR Search Engine

**Project Title:** Explainable Hybrid Academic Search Engine Using BM25, Semantic Retrieval, Hybrid Ranking, and Query Expansion  
**Author:** Hemanth Grandhi (Roll No.: 2310110121)  
**Course:** CSD358: Information Retrieval  

---

## 1. Quick Start & Execution Commands

### Prerequisites
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Option A: Launch the Interactive Streamlit Web Interface (Recommended for University Demo)
```bash
streamlit run src/app.py
```
*The web interface will open in your browser at `http://localhost:8501`.*

### Option B: Run the Interactive Command-Line Interface (CLI)
```bash
# Default Hybrid Search
python src/main.py

# Probabilistic BM25 with IR Debug Mode
python src/main.py --algo bm25 --debug

# Hybrid Search with Query Expansion
python src/main.py --algo hybrid --alpha 0.6 --expand

# Dense Semantic Search
python src/main.py --algo semantic

# Classical SMART lnc.ltc Weighting (Mid-term compatibility)
python src/main.py --smart --debug
```

### Option C: Run the Comprehensive Evaluation Suite & Generate Plots
```bash
python src/evaluation.py
```
*Generates all benchmark CSVs, JSON data, and high-resolution comparison plots in `results/`.*

### Option D: Run Unit & Integration Test Suite
```bash
python -m unittest discover tests
```
*Executes all 14 unit and integration tests.*

---

## 2. Interactive Demonstration Walkthrough (Step-by-Step for Examiners)

### Demo Step 1: The Core Academic Search Tab
1. Open the web interface at `http://localhost:8501`.
2. Notice the **Corpus Statistics** in the sidebar: 1,500 arXiv research papers loaded from `data/arxiv_corpus.json`.
3. In the search box, enter the query:
   ```text
   distributed training neural networks
   ```
4. Click **Search Papers** with algorithm set to **Hybrid Search (BM25 + Semantic)** ($\alpha=0.5$).
5. **What to highlight to the examiner:**
   - The paper *Distributed Training for Deep Learning Models...* is ranked #1.
   - Expand the paper card to show the **Ranking Explainability & Mathematical Attribution**:
     - The **Primary Ranking Driver** is diagnosed automatically: *Lexical Keyword Match (Exact Term Saturation)*.
     - The interactive **Component Bar Chart** shows the exact contribution: $\approx 50\%$ from BM25 and $\approx 37\%$ from Dense Semantic vector proximity.
     - The **LaTeX Formulation** demonstrates the exact score fusion:
       $$\text{HybridScore} = 0.50 \times S_{\text{BM25}}^{\text{norm}} + 0.50 \times S_{\text{semantic}}^{\text{norm}}$$
     - The **Keyword Badges** show term-level contributions and identify where terms matched (e.g., `'distribut'` in Title & Abstract, `'train'` in Title & Abstract).

### Demo Step 2: Demonstrating Semantic Search Generalization
1. Change the algorithm dropdown in the sidebar to **Semantic Search (Sentence-Transformer)**.
2. Enter a conceptual query that does not use typical textbook keywords:
   ```text
   teaching artificial agents through rewards
   ```
3. Click **Search Papers**.
4. **What to highlight to the examiner:**
   - The system retrieves papers on **Reinforcement Learning** (e.g., policy optimization, Q-learning, reward modeling) even though the words *"reinforcement learning"* were not literally present in the query!
   - Highlight that classical inverted index keyword matching would return 0 results here, but dense vector embeddings locate conceptually related research.

### Demo Step 3: Demonstrating Query Expansion
1. Enable the **Enable Query Expansion** checkbox in the sidebar.
2. Set expansion strategy to **Domain Synonyms + WordNet**.
3. In the search box, enter the domain acronym:
   ```text
   ml for image classification
   ```
4. **What to highlight to the examiner:**
   - The blue notification shows:
     `Query Expansion Active: Original: 'ml for image classification' -> Expanded: 'ml for image classification machine learning'`
   - Papers containing only the phrase *"Machine Learning"* are successfully retrieved and promoted in the rankings.

### Demo Step 4: Multi-Algorithm Comparison Tab
1. Click on **Tab 2: ⚖️ Multi-Algorithm Comparison**.
2. Enter query:
   ```text
   deep learning image classification
   ```
3. **What to highlight to the examiner:**
   - Side-by-side 4-column display showing how **TF-IDF**, **BM25**, **Dense Semantic**, and **Hybrid** rank different papers for the exact same query.
   - Point out how BM25 normalizes document length to prevent long papers from dominating, while Hybrid surfaces papers that possess both high lexical precision and semantic relevance.

### Demo Step 5: Evaluation & Empirical Dashboard Tab
1. Click on **Tab 3: 📊 Evaluation & Empirical Dashboard**.
2. **What to highlight to the examiner:**
   - **Leaderboard:** Shows the exact performance across all 10 systems evaluated on 19 queries.
   - Point out that **BM25 achieves NDCG@10 of 0.9446 and P@5 of 0.9895**, significantly outperforming baseline TF-IDF ($0.8117$) and Boolean OR ($0.5832$).
   - Point out the **Figure 3: Alpha Sensitivity Curve**, showing how NDCG varies smoothly as $\alpha$ increases from $0.0$ to $1.0$.
   - Point out the **BM25 Ablation Table**:
     - Without length normalization ($b=0$), NDCG drops to $0.9365$.
     - Without term saturation ($k_1=100$), NDCG drops severely to $0.7121$ (-24.6%), proving mathematically why saturation is essential!

### Demo Step 6: IR Debug Inspector
1. Toggle **Enable IR Debug Inspector** in the sidebar.
2. Return to Tab 1.
3. Show the examiner the live vocabulary size (8,234 stems), document frequencies, and exact calculated IDF values ($IDF_{\text{TF-IDF}}$ vs. $IDF_{\text{BM25}}$).

---

## 3. Practical Demo Query Portfolio

| Query String | Target Paradigm to Highlight | Key Expected Behavior |
| :--- | :--- | :--- |
| `distributed training neural networks` | Hybrid / BM25 | Exact technical term match in title and body; balanced score decomposition. |
| `ml privacy preservation` | Query Expansion | Expands "ml" to "machine learning"; surfaces federated learning papers. |
| `approximate nearest neighbor search` | BM25 vs TF-IDF | Length normalization prevents verbose survey papers from overtaking concise papers. |
| `teaching agents through trial and error` | Semantic Search | Pure conceptual retrieval of reinforcement learning papers without literal keyword overlap. |
| `cross-encoder reranking dense retrieval` | Multi-Algorithm Comparison | Demonstrates high lexical precision in BM25 combined with contextual semantic ranking. |
