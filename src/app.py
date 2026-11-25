import streamlit as st
import os
import json
import pandas as pd
import numpy as np

from preprocessing import Preprocessor
from index import InvertedIndex
from tfidf import TFIDFScorer
from ranking import Ranker
from bm25 import BM25Scorer
from semantic_search import SemanticSearcher
from hybrid_search import HybridSearcher
from query_expansion import QueryExpander
from explanation import Explainer

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Explainable Academic IR Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for University Research Grade UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #2563EB;
        margin-bottom: 10px;
    }
    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-secondary {
        background-color: #E5E7EB;
        color: #374151;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Caching System Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing High-Performance IR Subsystems...")
def load_all_ir_resources():
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_corpus.json')
    
    # 1. Preprocessor & Inverted Index
    p = Preprocessor()
    idx = InvertedIndex(p)
    idx.build_from_corpus(corpus_path)
    
    # 2. Classical TF-IDF Scorers
    tfidf_ltc = TFIDFScorer(idx, smart_weighting="ltc.ltc")
    tfidf_ltc.build_vectors()
    
    tfidf_lnc = TFIDFScorer(idx, smart_weighting="lnc.ltc")
    tfidf_lnc.build_vectors()
    
    # 3. Probabilistic BM25 Scorer
    bm25 = BM25Scorer(idx, k1=1.5, b=0.75, title_weight=0.6, body_weight=0.4)
    
    # 4. Dense Semantic Searcher
    semantic = SemanticSearcher()
    semantic.build_or_load_embeddings(corpus_path)
    
    # 5. Hybrid Search Engine
    hybrid = HybridSearcher(bm25, semantic, default_alpha=0.5)
    
    # 6. Query Expander
    expander = QueryExpander(preprocessor=p, inverted_index=idx)
    
    return p, idx, tfidf_ltc, tfidf_lnc, bm25, semantic, hybrid, expander

preprocessor, idx, tfidf_ltc, tfidf_lnc, bm25, semantic, hybrid, expander = load_all_ir_resources()

# Session State for Search History
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# ---------------------------------------------------------
# Sidebar Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/search-in-browser.png", width=64)
    st.title("Search Controls")
    
    algorithm = st.selectbox(
        "Retrieval Algorithm",
        options=[
            "Hybrid Search (BM25 + Semantic)",
            "BM25 (Probabilistic Okapi)",
            "Semantic Search (Sentence-Transformer)",
            "TF-IDF + Cosine (Vector Space)",
            "SMART lnc.ltc + Zone Weighting"
        ],
        index=0
    )
    
    # Algorithm-specific parameter sliders
    alpha_val = 0.5
    if "Hybrid" in algorithm:
        st.markdown("**Hybrid Convex Weight (α)**")
        alpha_val = st.slider("Weight on Lexical BM25 (1-α for Semantic)", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"Lexical: {alpha_val*100:.0f}% | Semantic: {(1-alpha_val)*100:.0f}%")
        
    if "BM25" in algorithm or "Hybrid" in algorithm:
        with st.expander("BM25 Hyperparameters"):
            k1_param = st.slider("k1 (Term Saturation)", 0.5, 3.0, 1.5, 0.1)
            b_param = st.slider("b (Length Normalization)", 0.0, 1.0, 0.75, 0.05)
            bm25.k1 = k1_param
            bm25.b = b_param
            
    st.markdown("---")
    st.subheader("Query Expansion")
    enable_expansion = st.checkbox("Enable Query Expansion", value=False)
    expansion_method = "hybrid"
    if enable_expansion:
        expansion_method = st.selectbox(
            "Expansion Strategy",
            options=["hybrid", "domain", "wordnet", "prf"],
            format_func=lambda x: {
                "hybrid": "Domain Synonyms + WordNet",
                "domain": "Academic CS Domain Acronyms",
                "wordnet": "WordNet Linguistic Synsets",
                "prf": "Pseudo-Relevance Feedback (PRF)"
            }[x]
        )
        
    st.markdown("---")
    st.subheader("Filters & Constraints")
    top_k = st.slider("Top-K Results", 1, 20, 5)
    
    # Extract unique categories and years
    all_categories = set()
    all_years = set()
    for meta in idx.doc_metadata.values():
        for cat in meta.get('categories', []):
            all_categories.add(cat)
        yr = meta.get('year', '')
        if yr and yr.isdigit():
            all_years.add(int(yr))
            
    sorted_cats = sorted(list(all_categories))
    cat_filter = st.multiselect("Filter by arXiv Category", options=sorted_cats, default=[])
    
    min_year = min(all_years) if all_years else 2010
    max_year = max(all_years) if all_years else 2026
    year_range = st.slider("Publication Year Range", min_year, max_year, (min_year, max_year))
    
    st.markdown("---")
    enable_debug = st.checkbox("Enable IR Debug Inspector", value=False)

# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Explainable Academic Information Retrieval Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Hybrid Scientific Literature Search: <b>BM25</b>, <b>Sentence-Transformers</b>, <b>Convex Fusion</b>, and <b>Query Expansion</b>.</div>', unsafe_allow_html=True)

tab_search, tab_compare, tab_eval, tab_history = st.tabs([
    "🔍 Academic Search & Explainability",
    "⚖️ Multi-Algorithm Comparison",
    "📊 Evaluation & Empirical Dashboard",
    "📜 Search History & Session Analytics"
])

# ---------------------------------------------------------
# TAB 1: Search & Explainability
# ---------------------------------------------------------
with tab_search:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_input = st.text_input(
            "Enter your research query or topic:",
            placeholder="e.g., neural semantic search using transformer models, or distributed training",
            label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("Search Papers", use_container_width=True, type="primary")

    # Sample Quick Queries
    st.markdown("""
    <div style="font-size: 0.85rem; color: #6B7280; margin-top: -10px; margin-bottom: 15px;">
        <b>Suggested queries:</b> 
        <code>distributed training neural networks</code> | 
        <code>neural semantic search</code> | 
        <code>bm25 probabilistic retrieval</code> | 
        <code>large language models evaluation</code> | 
        <code>graph neural networks knowledge</code>
    </div>
    """, unsafe_allow_html=True)

    if query_input:
        # Preprocessing & Query Expansion
        raw_query = query_input.strip()
        tokens = preprocessor.process(raw_query)
        
        expanded_query = raw_query
        expanded_tokens = tokens
        expansion_info = {}
        
        if enable_expansion:
            ranker_for_prf = bm25 if "BM25" in algorithm else Ranker(tfidf_ltc)
            expanded_query, expanded_tokens, expansion_info = expander.expand_query(
                raw_query, method=expansion_method, max_terms=3, ranker=ranker_for_prf
            )
            st.info(f"💡 **Query Expansion Active ({expansion_method}):** Original: `{raw_query}` ➔ Expanded: `{expanded_query}` (Added: `{', '.join(expansion_info.get('added_terms', []))}`)")

        # Record to Search History
        if raw_query not in [h['query'] for h in st.session_state.search_history]:
            st.session_state.search_history.append({
                'query': raw_query,
                'algorithm': algorithm,
                'expanded': enable_expansion,
                'time': pd.Timestamp.now().strftime("%H:%M:%S")
            })

        # Retrieval Dispatch
        results = []
        explanations = {}
        
        # Candidate pool for filtering
        pool_depth = max(50, top_k * 5)
        
        if "Hybrid" in algorithm:
            results, explanations = hybrid.get_top_k(
                expanded_tokens, expanded_query, k=pool_depth, alpha=alpha_val
            )
        elif "BM25" in algorithm:
            results, explanations = bm25.get_top_k(expanded_tokens, k=pool_depth)
        elif "Semantic" in algorithm:
            results, explanations = semantic.get_top_k(expanded_query, k=pool_depth)
        elif "SMART" in algorithm:
            ranker_smart = Ranker(tfidf_lnc, title_weight=0.7, body_weight=0.3)
            results, explanations = ranker_smart.get_top_k(expanded_tokens, k=pool_depth)
        else: # Standard TF-IDF
            ranker_tfidf = Ranker(tfidf_ltc, title_weight=0.7, body_weight=0.3)
            results, explanations = ranker_tfidf.get_top_k(expanded_tokens, k=pool_depth)

        # Apply Category & Year Filtering
        filtered_results = []
        for score, doc_id in results:
            meta = idx.doc_metadata.get(doc_id, {})
            # Category filter
            doc_cats = meta.get('categories', [])
            if cat_filter and not any(c in cat_filter for c in doc_cats):
                continue
            # Year filter
            doc_yr = meta.get('year', '')
            if doc_yr and doc_yr.isdigit():
                yr_int = int(doc_yr)
                if yr_int < year_range[0] or yr_int > year_range[1]:
                    continue
            filtered_results.append((score, doc_id))
            if len(filtered_results) >= top_k:
                break

        st.subheader(f"Retrieved {len(filtered_results)} Papers for '{query_input}' (Algorithm: {algorithm})")

        if not filtered_results:
            st.warning("No papers matched your search criteria and filters. Try broadening your query or clearing category filters.")
        else:
            for rank, (score, doc_id) in enumerate(filtered_results, 1):
                meta = idx.doc_metadata.get(doc_id, {})
                title = meta.get('title', 'Unknown Title')
                authors = ", ".join(meta.get('authors', []))
                year = meta.get('year', 'N/A')
                body = meta.get('body', '')
                cats = ", ".join(meta.get('categories', []))
                url = meta.get('abstract_url', f"https://arxiv.org/abs/{doc_id}")

                with st.expander(f"**#{rank}** — {title}  (Score: `{score:.4f}`)", expanded=(rank == 1)):
                    # Metadata chips
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    c1.markdown(f"**Authors:** {authors}")
                    c2.markdown(f"**Year:** `{year}`")
                    c3.markdown(f"**arXiv ID:** `{doc_id}`")
                    c4.markdown(f"[📄 View arXiv]({url})")
                    
                    if cats:
                        st.caption(f"**Categories:** `{cats}`")
                        
                    st.markdown("**Abstract:**")
                    st.write(body)
                    
                    st.markdown("---")
                    st.markdown("### 🔬 Ranking Explainability & Mathematical Attribution")

                    # Detailed Paradigm-Specific Attribution
                    if "Hybrid" in algorithm and doc_id in explanations:
                        h_info = explanations[doc_id]
                        driver = h_info.get('ranking_driver', 'Fused Evidence')
                        lex_part = h_info.get('lexical_contribution', 0.0)
                        sem_part = h_info.get('semantic_contribution', 0.0)
                        
                        st.markdown(f"**Primary Driver:** `{driver}`")
                        
                        # Score breakdown chart
                        chart_data = pd.DataFrame({
                            'Component': ['Lexical (BM25)', 'Semantic (Dense Embeddings)'],
                            'Normalized Contribution': [lex_part, sem_part]
                        })
                        st.bar_chart(chart_data.set_index('Component'), height=180)
                        
                        # Mathematical formulation breakdown
                        st.latex(rf"\text{{HybridScore}} = {alpha_val:.2f} \times {h_info.get('norm_lexical', 0.0):.4f} + {(1-alpha_val):.2f} \times {h_info.get('norm_semantic', 0.0):.4f} = {score:.4f}")
                        
                        # Lexical term details
                        lex_terms = h_info.get('lexical_terms', {})
                        if lex_terms:
                            st.markdown("**Matched Query Keywords & Term Saturation:**")
                            cols = st.columns(min(len(lex_terms), 4))
                            for idx_col, (t, t_data) in enumerate(lex_terms.items()):
                                col = cols[idx_col % len(cols)]
                                t_zones = []
                                if t_data.get('title_score', 0) > 0: t_zones.append("Title")
                                if t_data.get('body_score', 0) > 0: t_zones.append("Abstract")
                                col.metric(label=f"Keyword '{t}'", value=f"{t_data.get('total', 0.0):.3f}", help=f"Zones: {', '.join(t_zones)}")

                    elif "BM25" in algorithm and doc_id in explanations:
                        bm_contribs = explanations[doc_id]
                        st.markdown("**BM25 Term Saturation & Length Normalization Contribution:**")
                        term_table = []
                        for t, d in bm_contribs.items():
                            term_table.append({
                                'Term': t,
                                'IDF': round(d.get('idf', 0.0), 3),
                                'Title TF': d.get('title_tf', 0),
                                'Body TF': d.get('body_tf', 0),
                                'Title BM25': round(d.get('title_score', 0.0), 4),
                                'Body BM25': round(d.get('body_score', 0.0), 4),
                                'Total Score': round(d.get('total', 0.0), 4)
                            })
                        if term_table:
                            st.dataframe(pd.DataFrame(term_table), use_container_width=True)

                    elif "Semantic" in algorithm and doc_id in explanations:
                        sem_info = explanations[doc_id]
                        st.markdown(f"**Semantic Similarity:** `{sem_info.get('semantic_score', 0.0):.4f}` ({sem_info.get('alignment_band', 'Alignment')})")
                        st.markdown(f"**Embedding Model:** `{sem_info.get('model_name', 'all-MiniLM-L6-v2')}` (384-dimensional dense representation)")
                        st.caption("Paper was retrieved due to conceptual context proximity in the vector space.")

                    else: # Classical TF-IDF
                        if doc_id in explanations:
                            st.text(Explainer.format_explanation(doc_id, meta, score, explanations[doc_id]))

    # IR Debug Inspector
    if enable_debug:
        st.markdown("---")
        st.subheader("🛠️ IR Debug Inspector")
        d1, d2, d3 = st.columns(3)
        d1.metric("Indexed Documents", idx.total_docs)
        d2.metric("Vocabulary Stems", len(idx.index))
        d3.metric("Processed Tokens", f"{len(tokens)}: {tokens}")
        
        if query_input:
            term_debug = []
            for t in tokens:
                df = idx.get_document_frequency(t)
                idf = tfidf_ltc.idf_map.get(t, 0.0)
                bm_idf = bm25.compute_idf(t)
                term_debug.append({'Token': t, 'Document Frequency (DF)': df, 'TF-IDF IDF': round(idf, 4), 'BM25 IDF': round(bm_idf, 4)})
            st.table(pd.DataFrame(term_debug))

# ---------------------------------------------------------
# TAB 2: Multi-Algorithm Comparison
# ---------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Side-by-Side Algorithm Comparison")
    st.markdown("Directly compare how different retrieval paradigms rank documents for the same search query.")
    
    comp_query = st.text_input("Enter comparison query:", value="deep learning image classification", key="comp_q")
    comp_k = st.slider("Comparison Top-K", 3, 10, 5, key="comp_k")
    
    if comp_query:
        c_tokens = preprocessor.process(comp_query)
        
        # 1. Classical TF-IDF
        r_tfidf = Ranker(tfidf_ltc, title_weight=0.7, body_weight=0.3)
        t_tfidf, _ = r_tfidf.get_top_k(c_tokens, k=comp_k)
        
        # 2. BM25
        t_bm25, _ = bm25.get_top_k(c_tokens, k=comp_k)
        
        # 3. Semantic
        t_sem, _ = semantic.get_top_k(comp_query, k=comp_k)
        
        # 4. Hybrid
        t_hyb, _ = hybrid.get_top_k(c_tokens, comp_query, k=comp_k, alpha=0.5)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 1. TF-IDF + Cosine")
            for r, (s, d) in enumerate(t_tfidf, 1):
                t = idx.doc_metadata.get(d, {}).get('title', d)[:45]
                st.markdown(f"**#{r}** `{s:.3f}` — {t}...")
                
        with col2:
            st.markdown("#### 2. Okapi BM25")
            for r, (s, d) in enumerate(t_bm25, 1):
                t = idx.doc_metadata.get(d, {}).get('title', d)[:45]
                st.markdown(f"**#{r}** `{s:.3f}` — {t}...")
                
        with col3:
            st.markdown("#### 3. Dense Semantic")
            for r, (s, d) in enumerate(t_sem, 1):
                t = idx.doc_metadata.get(d, {}).get('title', d)[:45]
                st.markdown(f"**#{r}** `{s:.3f}` — {t}...")
                
        with col4:
            st.markdown("#### 4. Hybrid (α=0.5)")
            for r, (s, d) in enumerate(t_hyb, 1):
                t = idx.doc_metadata.get(d, {}).get('title', d)[:45]
                st.markdown(f"**#{r}** `{s:.3f}` — {t}...")

# ---------------------------------------------------------
# TAB 3: Evaluation & Empirical Dashboard
# ---------------------------------------------------------
with tab_eval:
    st.subheader("📊 Empirical Evaluation & Experimental Benchmarks")
    st.markdown("Empirical results across 10 Information Retrieval systems evaluated on the curated 1,500-paper corpus.")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    csv_systems = os.path.join(results_dir, 'system_comparison.csv')
    csv_alpha = os.path.join(results_dir, 'alpha_tuning_experiment.csv')
    csv_ablation = os.path.join(results_dir, 'ablation_study.csv')
    
    if os.path.exists(csv_systems):
        df_sys = pd.read_csv(csv_systems)
        
        # Top KPI Cards
        best_p5 = df_sys.loc[df_sys['Precision@5'].idxmax()]
        best_map = df_sys.loc[df_sys['MAP'].idxmax()]
        best_ndcg = df_sys.loc[df_sys['NDCG@10'].idxmax()]
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Highest Precision@5", f"{best_p5['Precision@5']:.4f}", f"{best_p5['System']}")
        k2.metric("Highest MAP", f"{best_map['MAP']:.4f}", f"{best_map['System']}")
        k3.metric("Highest NDCG@10", f"{best_ndcg['NDCG@10']:.4f}", f"{best_ndcg['System']}")
        k4.metric("Benchmark Corpus Size", f"{idx.total_docs} Papers", "arXiv CS")
        
        st.markdown("---")
        st.markdown("### 🏆 Comprehensive System Leaderboard")
        st.dataframe(df_sys.style.highlight_max(axis=0, color='#D1FAE5'), use_container_width=True)
        
        # Interactive Visualizations
        st.markdown("---")
        st.markdown("### 📈 Evaluation Plots & Diagnostic Curves")
        p_col1, p_col2 = st.columns(2)
        
        fig_pr = os.path.join(results_dir, 'precision_recall_comparison.png')
        fig_ndcg = os.path.join(results_dir, 'map_mrr_ndcg_comparison.png')
        fig_alpha = os.path.join(results_dir, 'alpha_sensitivity_curve.png')
        fig_lat = os.path.join(results_dir, 'runtime_latency_comparison.png')
        
        if os.path.exists(fig_pr):
            p_col1.image(fig_pr, caption="Figure 1: Precision@5 vs Recall@10 Comparison", use_container_width=True)
        if os.path.exists(fig_ndcg):
            p_col2.image(fig_ndcg, caption="Figure 2: Ranking Quality (MAP, MRR, NDCG@10)", use_container_width=True)
            
        p_col3, p_col4 = st.columns(2)
        if os.path.exists(fig_alpha):
            p_col3.image(fig_alpha, caption="Figure 3: Hybrid Alpha Sensitivity Curve", use_container_width=True)
        if os.path.exists(fig_lat):
            p_col4.image(fig_lat, caption="Figure 4: Query Latency Benchmarking (ms)", use_container_width=True)

        if os.path.exists(csv_ablation):
            st.markdown("---")
            st.markdown("### 🔬 BM25 Component Ablation Analysis")
            df_ab = pd.read_csv(csv_ablation)
            st.table(df_ab)
            st.caption("Demonstrates the essential role of term saturation (k1) and length normalization (b) in preventing document length bias.")
    else:
        st.warning("Evaluation benchmark results not found. Please run `python src/evaluation.py` to generate the benchmark datasets.")

# ---------------------------------------------------------
# TAB 4: Search History
# ---------------------------------------------------------
with tab_history:
    st.subheader("📜 Session Search History")
    if not st.session_state.search_history:
        st.info("No queries executed yet in this session.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.search_history), use_container_width=True)
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.rerun()

st.markdown("---")
st.caption("CSD358: Information Retrieval Final Semester Research Project | Author: Hemanth Grandhi (Roll No.: 2310110121) | GitHub: https://github.com/hemanth-grandhi/IR")
