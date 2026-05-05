import json
import os
import argparse
import numpy as np
import faiss
import torch
import pickle
from tqdm import tqdm

# 1. Import BGE-M3 dedicated class
from FlagEmbedding import BGEM3FlagModel
from ragatouille import RAGPretrainedModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= Path Configuration =================
DATA_DIR = "./data"
INDEX_DIR = "./indices/bge_m3_hybrid" 
OUTPUT_DIR = "./result"

COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
# Dense index (FAISS)
INDEX_FILE = os.path.join(INDEX_DIR, "bge_m3_dense.index")
# Sparse index (Pickle saving weight dictionaries)
SPARSE_INDEX_FILE = os.path.join(INDEX_DIR, "bge_m3_sparse.pkl")
DOC_ID_FILE = os.path.join(INDEX_DIR, "doc_ids.json")

# ================= Model Configuration =================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

# Memory optimization configuration
BATCH_SIZE = 12       
MAX_SEQ_LENGTH = 1024

# Hybrid retrieval configuration
HYBRID_ALPHA = 0.5    # Hybrid weight: Final = Dense + alpha * Sparse 
RETRIEVAL_TOP_K = 50  # Number of candidates passed to reranker
CANDIDATE_TOP_K = 500 # Candidate pool size during hybrid retrieval 
RERANK_TOP_K = 10     # Final output number

class RetrievalSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # 1. Use BGEM3FlagModel (supports Dense + Sparse + ColBERT)
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = BGEM3FlagModel(
            EMBEDDING_MODEL_NAME, 
            use_fp16=True, 
            device=self.device
        )
        
        self.embedding_dim = 1024
        
        # 2. Initialize index containers
        self.index = None        # FAISS Dense Index
        self.sparse_index = []   # List of Dicts (Sparse Weights)
        self.doc_ids = []
        
        self._load_or_build_index()

        # 3. Rerank model
        self.reranker = None

    def _load_reranker(self):
        if self.reranker is None:
            print(f"Loading Rerank Model: {RERANK_MODEL_NAME}...")
            self.reranker = RAGPretrainedModel.from_pretrained(RERANK_MODEL_NAME)

    def _load_or_build_index(self):
        # Check if both Dense and Sparse indexes exist
        if os.path.exists(INDEX_FILE) and os.path.exists(DOC_ID_FILE) and os.path.exists(SPARSE_INDEX_FILE):
            print("Found existing hybrid index. Loading...")
            
            # Load Dense Index
            self.index = faiss.read_index(INDEX_FILE)
            
            # Load Doc IDs
            with open(DOC_ID_FILE, 'r') as f:
                self.doc_ids = json.load(f)
                
            # Load Sparse Weights
            print("Loading sparse embeddings from disk (this may take a moment)...")
            with open(SPARSE_INDEX_FILE, 'rb') as f:
                self.sparse_index = pickle.load(f)
                
            print(f"Index loaded. Total documents: {len(self.doc_ids)}")
        else:
            print("Index missing. Building new hybrid index...")
            self._build_index()

    def _build_index(self):
        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR)

        documents = []
        doc_ids = []
        print(f"Reading collection from {COLLECTION_PATH}...")
        with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                item = json.loads(line)
                doc_ids.append(item['id'])
                documents.append(item.get('text', ''))
        
        self.doc_ids = doc_ids

        print(f"Encoding {len(documents)} documents (Dense + Sparse)...")
        output = self.embed_model.encode(
            documents, 
            batch_size=BATCH_SIZE, 
            max_length=MAX_SEQ_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        # 1. Process Dense vectors
        print("Building FAISS Dense index...")
        dense_embeddings = output['dense_vecs']
        dense_embeddings = np.array(dense_embeddings).astype('float32')
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(dense_embeddings)
        
        # 2. Process Sparse vectors
        print("Saving Sparse index...")
        self.sparse_index = output['lexical_weights'] 
        
        # 3. Save all files
        print("Saving artifacts to disk...")
        faiss.write_index(self.index, INDEX_FILE)
        
        with open(DOC_ID_FILE, 'w') as f:
            json.dump(self.doc_ids, f)
            
        with open(SPARSE_INDEX_FILE, 'wb') as f:
            pickle.dump(self.sparse_index, f)
            
        print("Index build complete.")

    def retrieve_hybrid(self, query, top_k=50, candidate_k=200):
        """Hybrid Retrieval: Dense + Sparse"""
        # 1. Encode Query (get both Dense and Sparse simultaneously)
        q_output = self.embed_model.encode_queries(
            [query], 
            return_dense=True, 
            return_sparse=True
        )
        
        q_dense = np.array(q_output['dense_vecs']).astype('float32')
        q_sparse = q_output['lexical_weights'][0] # Only one query
        
        # 2. Dense Retrieval (FAISS) - Recall larger candidate set
        D, I = self.index.search(q_dense, candidate_k)
        
        candidates = []
        for rank, (score_dense, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1: continue
            
            # Get corresponding Sparse vector
            doc_sparse = self.sparse_index[idx]
            
            # Compute Sparse score
            score_sparse = self.embed_model.compute_lexical_matching_score(q_sparse, doc_sparse)
            # Hybrid Fusion
            final_score = score_dense + (HYBRID_ALPHA * score_sparse)
            
            candidates.append({
                'id': self.doc_ids[idx],
                'score': float(final_score),
                'dense_score': float(score_dense),
                'sparse_score': float(score_sparse)
            })
        
        # 3. Re-sort by hybrid score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. Truncate and return Top-K
        return candidates[:top_k]

    def perform_rerank(self, query, doc_texts, k=10):
        self._load_reranker()
        # ragatouille rerank
        results = self.reranker.rerank(
            query=query, 
            documents=doc_texts, 
            k=k
        )
        return results

class Pipeline:
    def __init__(self):
        self.retriever = RetrievalSystem()
        self.id2text = {}
        print("Loading document texts for reranking lookup...")
        with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                item = json.loads(line)
                self.id2text[item['id']] = item['text']

    def run(self, input_file, output_file):
        print(f"Processing {input_file}...")
        results = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line) for line in f]

        for item in tqdm(queries):
            query_text = item['text']
            query_id = item['id']
            
            # 1. Hybrid retrieval
            candidates = self.retriever.retrieve_hybrid(
                query_text, 
                top_k=RETRIEVAL_TOP_K, 
                candidate_k=CANDIDATE_TOP_K
            )
            
            candidate_texts = []
            valid_candidates = []
            
            for c in candidates:
                doc_id = c['id']
                if doc_id in self.id2text:
                    candidate_texts.append(self.id2text[doc_id])
                    valid_candidates.append(c)
            
            if not candidate_texts:
                final_docs = []
            else:
                # 2. Rerank (ColBERT)
                reranked_results = self.retriever.perform_rerank(
                    query=query_text, 
                    doc_texts=candidate_texts, 
                    k=RERANK_TOP_K
                )
                
                final_docs = []
                for res in reranked_results:
                    idx = res['result_index']
                    doc_id = valid_candidates[idx]['id']
                    score = res['score']
                    final_docs.append([doc_id, score])
            
            out_obj = {
                "id": query_id,
                "question": query_text,
                "answer": "Placeholder", 
                "retrieved_docs": final_docs
            }
            results.append(out_obj)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
            
        print(f"Writing results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(DATA_DIR, "validation.jsonl"))
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "bge_m3_hybrid_prediction.jsonl"))
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pipeline = Pipeline()
    pipeline.run(args.input, args.output)