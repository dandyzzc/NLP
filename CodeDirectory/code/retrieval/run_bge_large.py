import json
import os
import argparse
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ================= Path Configuration =================
DATA_DIR = "./data"
INDEX_DIR = "./indices/bge_large"
OUTPUT_DIR = "./result"

COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
INDEX_FILE = os.path.join(INDEX_DIR, "bge_faiss.index")
DOC_ID_FILE = os.path.join(INDEX_DIR, "doc_ids.json")


# ================= Model Configuration =================
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
BATCH_SIZE = 64
RETRIEVAL_TOP_K = 50  # Initial retrieval count
RERANK_TOP_K = 10     # Final retained count

class RetrievalSystem:
    def __init__(self, device='cuda'.org' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # 1. Load Embedding model (BGE)
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        
        # 2. Initialize or load FAISS index
        self.index = None
        self.doc_ids = []
        self._load_or_build_index()

        # 3. Rerank model placeholder (lazy loading)
        self.reranker = None

    def _load_reranker(self):
        """Lazy load the reranker to avoid occupying GPU memory at startup, load only when needed"""
        if self.reranker is None:
            print(f"Loading Rerank Model: {RERANK_MODEL_NAME}...")
            self.reranker = RAGPretrainedModel.from_pretrained(RERANK_MODEL_NAME)

    def _load_or_build_index(self):
        """Check if index exists locally; build if not, load if present"""
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
        """Read all documents, encode them, and build FAISS index"""
        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR)

        # 1. Read documents
        documents = []
        doc_ids = []
        print(f"Reading collection from {COLLECTION_PATH}...")
        with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                item = json.loads(line)
                doc_ids.append(item['id'])
                content = item.get('text', '')
                documents.append(content)
        
        self.doc_ids = doc_ids

        # 2. Batch encoding
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.embed_model.encode(
            documents, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        # 3. Build index (using Inner Product)
        print("Building FAISS index...")
        embeddings = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        # 4. Save to disk
        print("Saving index to disk...")
        faiss.write_index(self.index, INDEX_FILE)
        with open(DOC_ID_FILE, 'w') as f:
            json.dump(self.doc_ids, f)
        print("Index build complete.")

    def retrieve(self, query, k=50):
        """Dense Retrieval"""
        query_instruction = "Represent this sentence for searching relevant passages: "
        q_emb = self.embed_model.encode([query_instruction + query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype('float32')
        
        scores, indices = self.index.search(q_emb, k)
        
        retrieved_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc_id = self.doc_ids[idx]
                retrieved_docs.append({'id': doc_id, 'score': float(score)})
        
        return retrieved_docs

    def perform_rerank(self, query, doc_texts, k=10):
        """
        Encapsulate rerank call and ensure model is loaded
        """
        self._load_reranker()
        
        # Ragatouille rerank
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
        
        # Read query file
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line) for line in f]

        for item in tqdm(queries):
            query_text = item['text']
            query_id = item['id']
            
            # 1. Dense Retrieval (Top-50)
            candidates = self.retriever.retrieve(query_text, k=RETRIEVAL_TOP_K)
            
            # 2. Prepare input for reranking
            candidate_texts = []
            valid_candidates = [] # Keep corresponding metadata for recovery
            
            for c in candidates:
                doc_id = c['id']
                if doc_id in self.id2text:
                    candidate_texts.append(self.id2text[doc_id])
                    valid_candidates.append(c)
            
            if not candidate_texts:
                final_docs = []
            else:
                # 3. Reranking (Top-10)
                reranked_results = self.retriever.perform_rerank(
                    query=query_text, 
                    doc_texts=candidate_texts, 
                    k=RERANK_TOP_K
                )
                
                # 4. Format output
                final_docs = []
                for res in reranked_results:
                    # ragatouille returns result_index as the index in candidate_texts list
                    idx = res['result_index'] 
                    doc_id = valid_candidates[idx]['id']
                    score = res['score']
                    final_docs.append([doc_id, score])
            
            # Construct output object
            out_obj = {
                "id": query_id,
                "question": query_text,
                "answer": "Placeholder", 
                "retrieved_docs": final_docs
            }
            results.append(out_obj)

        # Write results
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
            
        print(f"Writing results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(DATA_DIR, "validation.jsonl"), help="Input query file")
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "bge_prediction.jsonl"), help="Output prediction file")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pipeline = Pipeline()
    pipeline.run(args.input, args.output)