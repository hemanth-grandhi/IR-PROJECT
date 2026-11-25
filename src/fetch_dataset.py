import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import os
import re

def clean_text(text):
    """Normalizes whitespace and removes XML/HTML artifacts."""
    if not text:
        return ""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_arxiv_papers(query, start=0, max_results=100, timeout=15):
    """Queries the official arXiv API for research papers."""
    base_url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': f'all:{query}',
        'start': str(start),
        'max_results': str(max_results),
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print(f"  Fetching up to {max_results} papers for query: '{query}' (start={start})...")
    
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'CSD358-AcademicIRProject/2.0 (University Research; hemanth.grandhi@university.edu)'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read().decode('utf-8')
    except Exception as e:
        print(f"  [Warning] arXiv request failed for query '{query}': {e}")
        return []
        
    try:
        root = ET.fromstring(data)
    except Exception as e:
        print(f"  [Warning] Failed to parse XML: {e}")
        return []
        
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    papers = []
    for entry in root.findall('atom:entry', ns):
        title_el = entry.find('atom:title', ns)
        summary_el = entry.find('atom:summary', ns)
        published_el = entry.find('atom:published', ns)
        id_el = entry.find('atom:id', ns)
        
        if title_el is None or summary_el is None or id_el is None:
            continue
            
        title = clean_text(title_el.text)
        summary = clean_text(summary_el.text)
        raw_id = id_el.text.strip()
        doc_id = raw_id.split('/')[-1]
        
        published = published_el.text if published_el is not None else ""
        year = published[:4] if (published and len(published) >= 4) else "2024"
        
        # Authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name_el = author.find('atom:name', ns)
            if name_el is not None and name_el.text:
                authors.append(clean_text(name_el.text))
                
        # Categories
        categories = []
        for cat in entry.findall('atom:category', ns):
            term = cat.attrib.get('term')
            if term:
                categories.append(term)
                
        # Abstract URL
        abs_url = f"https://arxiv.org/abs/{doc_id}"
        
        # Validation: Ignore records without valid title or abstract
        if len(title) < 5 or len(summary) < 20:
            continue
            
        papers.append({
            "document_id": doc_id,
            "title": title,
            "body": summary,
            "authors": authors,
            "year": year,
            "categories": categories if categories else ["cs.AI"],
            "abstract_url": abs_url,
            "source": "arXiv"
        })
        
    return papers

def expand_and_validate_corpus(output_path, target_papers=1500):
    """
    Collects, deduplicates, validates, and persists academic paper corpus.
    Preserves all existing papers and enriches the dataset to target_papers.
    """
    existing_papers = []
    seen_ids = set()
    seen_titles = set()
    
    # 1. Load existing corpus to preserve historical baseline records
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_papers = json.load(f)
            print(f"Loaded {len(existing_papers)} existing papers from {output_path}.")
            for p in existing_papers:
                # Add default categories and url if missing from legacy records
                if 'categories' not in p:
                    p['categories'] = ["cs.IR", "cs.AI"]
                if 'abstract_url' not in p:
                    p['abstract_url'] = f"https://arxiv.org/abs/{p['document_id']}"
                seen_ids.add(p['document_id'])
                seen_titles.add(p['title'].lower().strip())
        except Exception as e:
            print(f"Notice: Could not read existing corpus ({e}). Starting fresh.")

    all_papers = list(existing_papers)
    
    # Curated, diverse academic search topics covering modern IR & AI
    queries = [
        "information retrieval ranking",
        "neural semantic search",
        "bm25 probabilistic retrieval",
        "query expansion wordnet",
        "dense passage retrieval",
        "transformer models for search",
        "distributed training neural networks",
        "federated learning privacy",
        "large language models evaluation",
        "graph neural networks knowledge",
        "machine learning image classification",
        "cross-encoder reranking",
        "vector database indexing",
        "recommendation systems collaborative",
        "deep learning natural language processing",
        "approximate nearest neighbor search",
        "multimodal information retrieval",
        "zero-shot academic search"
    ]
    
    print(f"\nTargeting ~{target_papers} academic papers across {len(queries)} research domains...")
    
    for q in queries:
        if len(all_papers) >= target_papers:
            break
            
        needed = target_papers - len(all_papers)
        batch_size = min(100, max(50, needed))
        
        batch = fetch_arxiv_papers(q, start=0, max_results=batch_size)
        added_count = 0
        for p in batch:
            clean_t = p['title'].lower().strip()
            if p['document_id'] not in seen_ids and clean_t not in seen_titles:
                all_papers.append(p)
                seen_ids.add(p['document_id'])
                seen_titles.add(clean_t)
                added_count += 1
                if len(all_papers) >= target_papers:
                    break
                    
        print(f"  + Added {added_count} unique papers for '{q}'. Total collected: {len(all_papers)}")
        time.sleep(3) # Polite delay per arXiv API policy

    # Validation pass
    valid_papers = []
    for p in all_papers:
        if (p.get('document_id') and p.get('title') and p.get('body') and 
            len(p['title']) >= 5 and len(p['body']) >= 20):
            valid_papers.append(p)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(valid_papers, f, indent=2, ensure_ascii=False)
        
    print(f"\n==========================================")
    print(f"Dataset Collection & Validation Complete!")
    print(f"Total Valid Papers: {len(valid_papers)}")
    print(f"Output File: {output_path}")
    print(f"==========================================\n")
    return valid_papers

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_corpus.json')
    expand_and_validate_corpus(out_file, target_papers=1500)
