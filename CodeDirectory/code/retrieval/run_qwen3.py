import json
import os
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from tqdm import tqdm

# ================= Path Configuration =================
DATA_DIR = "./data"
OUTPUT_FILE = "qwen3_prediction.jsonl"
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
QUERY_PATH = os.path.join(DATA_DIR, "validation.jsonl")

# Index storage path
INDEX_DIR = "./indices/qwen3"
INDEX_FILE_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Model configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" 

# Qwen-Instruct
QUERY_INSTRUCTION = "Instruct: Given a multi-hop reasoning question requiring evidence from multiple Wikipedia passages, retrieve the most relevant supporting passages that contain the bridging entities, comparison facts, or direct evidence needed to answer the query.\nQuery: "

# Retrieval configuration
RETRIEVAL_TOP_K = 50   # First-stage dense retrieval quantity
RERANK_TOP_K = 10      # Final submission quantity
BATCH_SIZE = 8         # Batch size for embedding and reranking

def load_jsonl(path):
    data = []
    print(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # 1. Load text data
    corpus_data = load_jsonl(COLLECTION_PATH)
    query_data = load_jsonl(QUERY_PATH)

    corpus_ids = [doc['id'] for doc in corpus_data]
    # Create mapping for Reranker usage
    corpus_map = {doc['id']: doc['text'] for doc in corpus_data}

    # ==========================================
    # 2. Dense Index Construction and Loading (FAISS)
    # ==========================================
    
    # Check if saved FAISS index exists
    if os.path.exists(INDEX_FILE_PATH):
        print(f"Loading existing FAISS index from {INDEX_FILE_PATH}...")
        # Read index
        index = faiss.read_index(INDEX_FILE_PATH)
        
    else:
        print(f"Index not found. Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        # Load Embedding model
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        
        # Some models require setting maximum sequence length
        embedder.max_seq_length = 1024 

        print("Encoding corpus (this may take a while)...")
        corpus_texts = [doc['text'] for doc in corpus_data]
        
        # Batch encode documents
        corpus_embeddings = embedder.encode(
            corpus_texts, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        
        print("Building FAISS index...")
        d = corpus_embeddings.shape[1] # Vector dimension
        
        # Use Inner Product (IP) index
        index = faiss.IndexFlatIP(d) 
        index.add(corpus_embeddings)
        
        print(f"Saving FAISS index to {INDEX_FILE_PATH}...")
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_FILE_PATH)
        
        # Release Embedding model GPU memory to prevent OOM in later Rerank stage
        del embedder
        del corpus_embeddings
        torch.cuda.empty_cache()
        print("Embedding model offloaded.")

    print(f"FAISS Index ready. Total vectors: {index.ntotal}")

    # ==========================================
    # 3. Query Encoding and Retrieval
    # ==========================================
    print(f"Encoding queries with instruction...")
    
    # Reload model only for Query encoding 
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    
    # Add Instruction to Query
    queries_with_instruction = [f"{QUERY_INSTRUCTION}{q['text']}" for q in query_data]
    
    query_embeddings = embedder.encode(
        queries_with_instruction,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Execute retrieval
    print(f"Searching FAISS for top {RETRIEVAL_TOP_K}...")
    # D: distance/score, I: index IDs
    D, I = index.search(query_embeddings, RETRIEVAL_TOP_K)
    
    # Release GPU memory
    del embedder
    del query_embeddings
    torch.cuda.empty_cache()

    # ==========================================
    # 4. ColBERT Reranking
    # ==========================================
    print("Loading ColBERT model for reranking...")
    reranker = RAGPretrainedModel.from_pretrained("answerdotai/answerai-colbert-small-v1")

    output_results = []
    query_texts = [q['text'] for q in query_data] # Use original Query for Rerank, without Instruction
    query_ids = [q['id'] for q in query_data]

    for i in tqdm(range(len(query_data)), desc="Reranking"):
        qid = query_ids[i]
        q_text = query_texts[i]
        
        # Get indices retrieved by FAISS
        retrieved_indices = I[i]
        
        candidate_docs = []
        candidate_ids = []
        
        for idx in retrieved_indices:
            if idx == -1: continue # FAISS padding marker
            doc_id = corpus_ids[idx]
            candidate_ids.append(doc_id)
            candidate_docs.append(corpus_map[doc_id])
            
        # ColBERT Rerank
        rerank_results = reranker.rerank(
            query=q_text, 
            documents=candidate_docs, 
            k=RERANK_TOP_K
        )
        
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

    # ==========================================
    # 5. Save Results
    # ==========================================
    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in output_results:
            f.write(json.dumps(item) + "\n")
    
    print("Done!")

if __name__ == "__main__":
    main()