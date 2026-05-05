import json
import os
import argparse
import numpy as np
import faiss
import torch
import bm25s
import Stemmer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from collections import defaultdict

# Path configurations
DATA_DIR = "./data"
OUTPUT_DIR = "./result"
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")

# Index paths
BM25_INDEX_DIR = "./indices/bm25s"
BGE_INDEX_DIR = "./indices/bge_large"
BGE_INDEX_FILE = os.path.join(BGE_INDEX_DIR, "bge_faiss.index")

# Parameter settings
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

RETRIEVAL_TOP_K = 50   # Number of docs retrieved per method
RERANK_TOP_K = 10      # Final number of docs after reranking
BATCH_SIZE = 32

class HybridRetrievalSystem:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # In-memory data structures
        self.corpus_ids = []       # Ordered list of document IDs
        self.id2text = {}          # Map from doc ID to text
        self.bm25_retriever = None
        self.dense_index = None
        self.embed_model = None
        self.reranker = None
        self.stemmer = Stemmer.Stemmer("english")

        # Load corpus mappings once
        self._load_corpus_mapping()

    def _load_corpus_mapping(self):
        print(f"Loading corpus from {COLLECTION_PATH}...")
        with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                item = json.loads(line)
                self.corpus_ids.append(item['id'])
                self.id2text[item['id']] = item['text']

    # BM25 Module
    def init_bm25(self):
        print("\n[Initializing BM25]")
        if os.path.exists(BM25_INDEX_DIR) and len(os.listdir(BM25_INDEX_DIR)) > 0:
            print(f"Loading existing BM25 index from {BM25_INDEX_DIR}...")
            self.bm25_retriever = bm25s.BM25.load(BM25_INDEX_DIR, load_corpus=False)
        else:
            print("Building BM25 index from scratch...")
            corpus_texts = list(self.id2text.values())
            corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=self.stemmer)
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(corpus_tokens)
            os.makedirs(BM25_INDEX_DIR, exist_ok=True)
            self.bm25_retriever.save(BM25_INDEX_DIR)
            print("BM25 index saved.")

    def retrieve_bm25(self, query_texts, k=50):
        print("Running BM25 retrieval...")
        query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=self.stemmer)
        doc_indices, scores = self.bm25_retriever.retrieve(query_tokens, k=k)
        
        results = []
        for i in range(len(query_texts)):
            # Convert int indices to string IDs
            ids = [self.corpus_ids[idx] for idx in doc_indices[i]]
            results.append(ids)
        return results

    # Dense (BGE) Module
    def init_dense(self):
        print("\n[Initializing Dense Retrieval]")
        # Load embedding model
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        # Load or build FAISS index
        if os.path.exists(BGE_INDEX_FILE):
            print(f"Loading FAISS index from {BGE_INDEX_FILE}...")
            self.dense_index = faiss.read_index(BGE_INDEX_FILE)
        else:
            print("Building FAISS index...")
            os.makedirs(BGE_INDEX_DIR, exist_ok=True)
            corpus_texts = list(self.id2text.values())
            embeddings = self.embed_model.encode(
                corpus_texts, 
                batch_size=64, 
                show_progress_bar=True, 
                normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            self.dense_index = faiss.IndexFlatIP(dim)
            self.dense_index.add(embeddings.astype('float32'))
            faiss.write_index(self.dense_index, BGE_INDEX_FILE)
            print("FAISS index saved.")

    def retrieve_dense(self, query_texts, k=50):
        print("Running Dense retrieval...")
        # BGE requires specific instruction for query encoding
        instruction = "Represent this sentence for searching relevant passages: "
        q_texts_with_instruction = [instruction + q for q in query_texts]
        
        q_embs = self.embed_model.encode(
            q_texts_with_instruction, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        scores, indices = self.dense_index.search(q_embs.astype('float32'), k)
        
        results = []
        for i in range(len(query_texts)):
            ids = []
            for idx in indices[i]:
                if idx != -1:  # Skip invalid indices
                    ids.append(self.corpus_ids[idx])
            results.append(ids)
        return results

    # Rerank Module
    def init_reranker(self):
        print(f"\n[Loading Reranker: {RERANK_MODEL_NAME}]")
        self.reranker = RAGPretrainedModel.from_pretrained(RERANK_MODEL_NAME)

    def rerank(self, query, doc_ids, k=10):
        # Prepare document texts and filter valid IDs
        doc_texts = [self.id2text[did] for did in doc_ids if did in self.id2text]
        doc_ids_clean = [did for did in doc_ids if did in self.id2text]
        
        if not doc_texts:
            return []
            
        results = self.reranker.rerank(query=query, documents=doc_texts, k=k)
        
        final_results = []
        for item in results:
            # Map back to original document IDs
            original_id = doc_ids_clean[item['result_index']]
            final_results.append([original_id, float(item['score'])])
            
        return final_results

# Fusion Algorithm (RRF)
def rrf_fusion(list1, list2, k=60):
    """Reciprocal Rank Fusion to combine two ranked lists"""
    scores = defaultdict(float)
    
    # Process first list (BM25)
    for rank, doc_id in enumerate(list1):
        scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Process second list (Dense)
    for rank, doc_id in enumerate(list2):
        scores[doc_id] += 1.0 / (k + rank + 1)
        
    # Sort by score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]

# Main Workflow
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(DATA_DIR, "validation.jsonl"))
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "hybrid_prediction.jsonl"))
    args = parser.parse_args()

    # 1. Initialize retrieval system
    system = HybridRetrievalSystem()
    
    # 2. Prepare retrievers
    system.init_bm25()
    system.init_dense()
    system.init_reranker()

    # 3. Load queries
    print(f"\nLoading queries from {args.input}...")
    queries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    
    query_texts = [q['text'] for q in queries]
    query_ids = [q['id'] for q in queries]

    # 4. Batch retrieval
    bm25_results_list = system.retrieve_bm25(query_texts, k=RETRIEVAL_TOP_K)
    dense_results_list = system.retrieve_dense(query_texts, k=RETRIEVAL_TOP_K)

    # 5. Fusion and reranking for each query
    print(f"\nStarting Fusion and Reranking for {len(queries)} queries...")
    final_output = []
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for i in tqdm(range(len(queries))):
        qid = query_ids[i]
        q_text = query_texts[i]
        
        # Merge results with RRF
        candidates = rrf_fusion(bm25_results_list[i], dense_results_list[i])
        
        # Rerank top candidates (limit to 100 for efficiency)
        rerank_candidates = candidates[:100] 
        
        top_docs = system.rerank(q_text, rerank_candidates, k=RERANK_TOP_K)
        
        final_output.append({
            "id": qid,
            "question": q_text,
            "answer": queries[i].get("answer", ""),
            "retrieved_docs": top_docs
        })

    # 6. Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final_output:
            f.write(json.dumps(item) + "\n")
    print("Done!")

if __name__ == "__main__":
    main()