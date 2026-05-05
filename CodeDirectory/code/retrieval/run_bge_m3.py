import json
import os
import argparse
import numpy as np
import faiss
import torch
from tqdm import tqdm
# 1. Import official libraries
from FlagEmbedding import FlagModel 
from ragatouille import RAGPretrainedModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= Path Configuration =================
DATA_DIR = "./data"
INDEX_DIR = "./indices/bge_m3" 
OUTPUT_DIR = "./result"

COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
INDEX_FILE = os.path.join(INDEX_DIR, "bge_m3.index")
DOC_ID_FILE = os.path.join(INDEX_DIR, "doc_ids.json")

# ================= Model Configuration =================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"

# Memory optimization configuration
BATCH_SIZE = 16       # FlagModel also recommends controlling batch size
MAX_SEQ_LENGTH = 1024 # Limit length to prevent OOM

RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 10

class RetrievalSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # 1. Load BGE-M3 using FlagModel
        print(f"Loading Embedding Model via FlagEmbedding: {EMBEDDING_MODEL_NAME}...")
        
        self.embed_model = FlagModel(
            EMBEDDING_MODEL_NAME, 
            query_instruction_for_retrieval="", # M3 Dense does not require instruction, keep empty
            use_fp16=True,  # Enable half-precision, saves memory and is faster, officially recommended
            device=self.device
        )
        
        self.embedding_dim = 1024
        
        # 2. Initialize or load FAISS index
        self.index = None
        self.doc_ids = []
        self._load_or_build_index()

        # 3. Rerank model
        self.reranker = None

    def _load_reranker(self):
        if self.reranker is None:
            print(f"Loading Rerank Model: {RERANK_MODEL_NAME}...")
            self.reranker = RAGPretrainedModel.from_pretrained(RERANK_MODEL_NAME)

    def _load_or_build_index(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(DOC_ID_FILE):
            print("Found existing index. Loading...")
            self.index = faiss.read_index(INDEX_FILE)
            with open(DOC_ID_FILE, 'r') as f:
                self.doc_ids = json.load(f)
            print(f"Index loaded. Total documents: {len(self.doc_ids)}")
        else:
            print("Index not found. Building new index...")
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

        print(f"Encoding {len(documents)} documents with FlagModel...")
        

        embeddings = self.embed_model.encode(
            documents, 
            batch_size=BATCH_SIZE, 
            max_length=MAX_SEQ_LENGTH, # Explicitly limit length here

        )
        
        print("Building FAISS index...")
        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        print("Saving index to disk...")
        faiss.write_index(self.index, INDEX_FILE)
        with open(DOC_ID_FILE, 'w') as f:
            json.dump(self.doc_ids, f)
        print("Index build complete.")

    def retrieve(self, query, k=50):
        """Dense Retrieval"""
        q_emb = self.embed_model.encode_queries([query])
        q_emb = np.array(q_emb).astype('float32')
        
        scores, indices = self.index.search(q_emb, k)
        
        retrieved_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc_id = self.doc_ids[idx]
                retrieved_docs.append({'id': doc_id, 'score': float(score)})
        
        return retrieved_docs

    def perform_rerank(self, query, doc_texts, k=10):
        self._load_reranker()
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
            
            candidates = self.retriever.retrieve(query_text, k=RETRIEVAL_TOP_K)
            
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
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "bge_m3_prediction.jsonl"))
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pipeline = Pipeline()
    pipeline.run(args.input, args.output)