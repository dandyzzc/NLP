import json
import os
import sys
import logging
import contextlib
import numpy as np
import faiss
from model2vec import StaticModel
from ragatouille import RAGPretrainedModel
from tqdm import tqdm

# ================= Path Configuration =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "indices", "potion_8m") 
OUTPUT_FILE = "static_prediction.jsonl"

COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
QUERY_PATH = os.path.join(DATA_DIR, "validation.jsonl")

# Model configuration
EMBED_MODEL_NAME = "minishlab/potion-base-8M"
RERANK_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

BATCH_SIZE = 1024
RETRIEVAL_TOP_K = 50  # Initial screening quantity (candidate set for Reranker)
FINAL_TOP_K = 10      # Final submission quantity

# ================= Utility Functions =================
@contextlib.contextmanager
def suppress_output():
    """Suppress RAGatouille internal verbose output"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_jsonl(path):
    print(f"Loading data from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_or_load_index(corpus_data):
    """
    Load or build Model2Vec FAISS index
    """
    ensure_dir(INDEX_DIR)
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    id_map_path = os.path.join(INDEX_DIR, "doc_ids.json")

    # 1. Try to load existing index
    if os.path.exists(index_path) and os.path.exists(id_map_path):
        print(f"Loading existing FAISS index from {INDEX_DIR}...")
        index = faiss.read_index(index_path)
        with open(id_map_path, 'r', encoding='utf-8') as f:
            doc_ids = json.load(f)
        return index, doc_ids

    # 2. Build new index
    print(f"Index not found. Building with {EMBED_MODEL_NAME}...")
    model = StaticModel.from_pretrained(EMBED_MODEL_NAME)
    
    corpus_texts = [doc['text'] for doc in corpus_data]
    doc_ids = [doc['id'] for doc in corpus_data]
    
    print(f"Encoding {len(corpus_texts)} documents...")
    embeddings_list = []
    for i in tqdm(range(0, len(corpus_texts), BATCH_SIZE), desc="Encoding Corpus"):
        batch_texts = corpus_texts[i : i + BATCH_SIZE]
        batch_emb = model.encode(batch_texts)
        embeddings_list.append(batch_emb)
    
    embeddings = np.vstack(embeddings_list)
    
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)
    
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(doc_ids, f)
        
    return index, doc_ids

def main():
    # Set log level
    logging.getLogger("ragatouille").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # 1. Load corpus (Reranker needs original text, so must load)
    corpus_data = load_jsonl(COLLECTION_PATH)
    # Create id -> text mapping for Reranker to lookup content
    corpus_map = {doc['id']: doc['text'] for doc in corpus_data}

    # 2. Prepare retrieval index (Model2Vec)
    index, doc_ids = build_or_load_index(corpus_data)
    
    # 3. Prepare Reranker (ColBERT)
    print(f"Loading Reranker ({RERANK_MODEL_NAME})...")
    with suppress_output():
        reranker = RAGPretrainedModel.from_pretrained(RERANK_MODEL_NAME)

    # 4. Process Queries
    query_data = load_jsonl(QUERY_PATH)
    query_texts = [q['text'] for q in query_data]
    query_ids = [q['id'] for q in query_data]
    
    print(f"Encoding queries with Model2Vec...")
    embed_model = StaticModel.from_pretrained(EMBED_MODEL_NAME)
    query_embeddings = embed_model.encode(query_texts)
    faiss.normalize_L2(query_embeddings)
    
    # 5. First-stage retrieval (Retrieval Top-50)
    print(f"Retrieving top {RETRIEVAL_TOP_K} candidates via FAISS...")
    D, I = index.search(query_embeddings, RETRIEVAL_TOP_K)
    
    # 6. Second-stage reranking
    output_results = []
    print("Reranking candidates...")
    
    for i in tqdm(range(len(query_data)), desc="Reranking", ncols=100):
        qid = query_ids[i]
        q_text = query_texts[i]
        
        # Get candidate document contents
        candidate_docs = []
        candidate_ids = []
        
        for idx in I[i]:
            doc_id = doc_ids[idx]
            candidate_ids.append(doc_id)
            candidate_docs.append(corpus_map[doc_id])
            
        # ColBERT Rerank
        with suppress_output():
            rerank_results = reranker.rerank(
                query=q_text, 
                documents=candidate_docs, 
                k=FINAL_TOP_K
            )
            
        # Format results
        formatted_docs = []
        for item in rerank_results:
            original_doc_id = candidate_ids[item['result_index']]
            score = float(item['score'])
            formatted_docs.append([original_doc_id, score])
            
        output_results.append({
            "id": qid,
            "question": q_text,
            "answer": query_data[i].get("answer", ""),
            "retrieved_docs": formatted_docs
        })

    # 7. Save results
    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in output_results:
            f.write(json.dumps(item) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()