import json
import os
import argparse
import numpy as np
import faiss
import torch
import pickle
from tqdm import tqdm
import math
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Path Configuration =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "./data"
INDEX_DIR = "./indices/bge_m3_hybrid" 
OUTPUT_DIR = "./result"

COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
# Index file paths
INDEX_FILE = os.path.join(INDEX_DIR, "bge_m3_dense.index")
SPARSE_INDEX_FILE = os.path.join(INDEX_DIR, "bge_m3_sparse.pkl")
DOC_ID_FILE = os.path.join(INDEX_DIR, "doc_ids.json")

# ================= Model Configuration =================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# ================= Qwen Reranker Configuration =================
QWEN_INSTRUCTION = (
    "Given a multi-hop reasoning question requiring evidence from multiple Wikipedia passages, "
    "retrieve the most relevant supporting passages that contain the bridging entities, "
    "comparison facts, or direct evidence needed to answer the query."
)

# ================= Runtime Parameters =================
EMBED_BATCH_SIZE = 12       
MAX_SEQ_LENGTH = 1024 

# Hybrid retrieval parameters
HYBRID_ALPHA = 0.5    # Hybrid weight: Final = Dense + alpha * Sparse 
RETRIEVAL_TOP_K = 50  # Number of candidates passed to reranker (Top-50)
CANDIDATE_TOP_K = 500 # Initial coarse retrieval from FAISS (Top-500)
RERANK_TOP_K = 10     # Final number of documents to output

# Qwen Reranker parameters
QWEN_MAX_LENGTH = 1024
RERANK_BATCH_SIZE =2

class QwenRerankerWrapper:
    """
    Wrapper for Qwen3-Reranker-0.6B (used in CausalLM mode)
    """
    def __init__(self, model_name, device):
        self.device = device
        print(f"Loading Qwen Reranker: {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (bf16/fp16 to save memory)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True
        ).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_input(self, query, doc):
        """Construct input string using the instruction"""
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=QWEN_INSTRUCTION, 
            query=query, 
            doc=doc
        )

    def compute_scores_batch(self, pairs):
        """Compute relevance scores in batches to avoid OOM"""
        all_scores = []
        
        for i in range(0, len(pairs), RERANK_BATCH_SIZE):
            batch_pairs = pairs[i : i + RERANK_BATCH_SIZE]
            
            raw_inputs = [self.format_input(q, d) for q, d in batch_pairs]
            
            # 1. Tokenize content part
            # Reserve space for prefix and suffix
            max_len_content = QWEN_MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens)
            
            inputs = self.tokenizer(
                raw_inputs, 
                padding=False, 
                truncation='longest_first',
                return_attention_mask=False, 
                max_length=max_len_content
            )
            
            # 2. Manually concatenate: prefix + content + suffix
            input_ids_list = []
            for ele in inputs['input_ids']:
                input_ids_list.append(self.prefix_tokens + ele + self.suffix_tokens)
            
            # 3. Pad sequences
            padded = self.tokenizer.pad(
                {'input_ids': input_ids_list}, 
                padding=True, 
                return_tensors="pt", 
                max_length=QWEN_MAX_LENGTH
            )
            
            input_ids = padded['input_ids'].to(self.device)
            attention_mask = padded['attention_mask'].to(self.device)

            # 4. Inference
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_logits = outputs.logits[:, -1, :] # Logits of the last token
                
                # Extract yes/no logits
                true_vector = batch_logits[:, self.token_true_id]
                false_vector = batch_logits[:, self.token_false_id]
                
                # Softmax to compute probability
                combined = torch.stack([false_vector, true_vector], dim=1)
                log_probs = torch.nn.functional.log_softmax(combined, dim=1)
                
                # Take the probability of "yes"
                scores = log_probs[:, 1].exp().tolist()
                all_scores.extend(scores)
                
        return all_scores

    def rerank(self, query, documents, k=10):
        if not documents:
            return []
            
        pairs = [(query, doc) for doc in documents]
        
        # Compute scores for all documents
        scores = self.compute_scores_batch(pairs)
        
        # Combine results
        results = []
        for idx, score in enumerate(scores):
            results.append({
                'result_index': idx,
                'score': score
            })
            
        # Sort descending by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]


class RetrievalSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # 1. BGE-M3 Embedding
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = BGEM3FlagModel(
            EMBEDDING_MODEL_NAME, 
            use_fp16=True, 
            device=self.device
        )
        self.embedding_dim = 1024
        
        # 2. Index containers
        self.index = None
        self.sparse_index = []
        self.doc_ids = []
        
        self._load_or_build_index()

        # 3. Rerank
        self.reranker = None

    def _load_reranker(self):
        if self.reranker is None:
            self.reranker = QwenRerankerWrapper(RERANK_MODEL_NAME, self.device)

    def _load_or_build_index(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(DOC_ID_FILE) and os.path.exists(SPARSE_INDEX_FILE):
            print("Found existing hybrid index. Loading...")
            self.index = faiss.read_index(INDEX_FILE)
            with open(DOC_ID_FILE, 'r') as f:
                self.doc_ids = json.load(f)
            print("Loading sparse embeddings (this may take a while)...")
            with open(SPARSE_INDEX_FILE, 'rb') as f:
                self.sparse_index = pickle.load(f)
            print(f"Index loaded. Total docs: {len(self.doc_ids)}")
        else:
            print("Index missing. Building...")
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

        print(f"Encoding {len(documents)} documents...")
        output = self.embed_model.encode(
            documents, 
            batch_size=EMBED_BATCH_SIZE, 
            max_length=MAX_SEQ_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        print("Building FAISS Dense index...")
        dense_embeddings = np.array(output['dense_vecs']).astype('float32')
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(dense_embeddings)
        
        print("Saving Sparse index...")
        self.sparse_index = output['lexical_weights']
        
        print("Saving artifacts...")
        faiss.write_index(self.index, INDEX_FILE)
        with open(DOC_ID_FILE, 'w') as f:
            json.dump(self.doc_ids, f)
        with open(SPARSE_INDEX_FILE, 'wb') as f:
            pickle.dump(self.sparse_index, f)
        print("Index build complete.")

    def retrieve_hybrid(self, query, top_k=50, candidate_k=500):
        # Encode Query
        q_output = self.embed_model.encode_queries(
            [query], 
            return_dense=True, 
            return_sparse=True
        )
        q_dense = np.array(q_output['dense_vecs']).astype('float32')
        q_sparse = q_output['lexical_weights'][0]
        
        # Dense Retrieval (Candidates)
        D, I = self.index.search(q_dense, candidate_k)
        
        candidates = []
        # Compute hybrid score
        for rank, (score_dense, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1: continue
            
            doc_sparse = self.sparse_index[idx]
            score_sparse = self.embed_model.compute_lexical_matching_score(q_sparse, doc_sparse)
            
            # Final Score = Dense + 0.5 * Sparse
            final_score = score_dense + (HYBRID_ALPHA * score_sparse)
            
            candidates.append({
                'id': self.doc_ids[idx],
                'score': float(final_score)
            })
        
        # Sort and slice
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    def perform_rerank(self, query, doc_texts, k=10):
        self._load_reranker()
        return self.reranker.rerank(query, doc_texts, k)

class Pipeline:
    def __init__(self):
        self.retriever = RetrievalSystem()
        self.id2text = {}
        print("Loading document texts for reranking lookup...")
        if os.path.exists(COLLECTION_PATH):
            with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    self.id2text[item['id']] = item['text']

    def run(self, input_file, output_file):
        print(f"Processing queries from {input_file}...")
        results = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line) for line in f]

        for item in tqdm(queries):
            query_text = item['text']
            query_id = item['id']
            
            # 1. Hybrid Retrieval
            # Get RETRIEVAL_TOP_K candidates
            candidates = self.retriever.retrieve_hybrid(
                query_text, 
                top_k=RETRIEVAL_TOP_K, 
                candidate_k=CANDIDATE_TOP_K
            )
            
            # Prepare data for Rerank
            candidate_texts = []
            valid_candidates = []
            
            for c in candidates:
                doc_id = c['id']
                if doc_id in self.id2text:
                    candidate_texts.append(self.id2text[doc_id])
                    valid_candidates.append(c)
            
            final_docs = []
            if candidate_texts:
                # 2. Qwen Rerank
                reranked_results = self.retriever.perform_rerank(
                    query=query_text, 
                    doc_texts=candidate_texts, 
                    k=RERANK_TOP_K
                )
                
                # Map back to doc_id
                for res in reranked_results:
                    idx = res['result_index']
                    doc_id = valid_candidates[idx]['id']
                    score = res['score']
                    final_docs.append([doc_id, score])
            
            # Construct output
            out_obj = {
                "id": query_id,
                "question": query_text,
                "answer": "Placeholder", 
                "retrieved_docs": final_docs
            }
            results.append(out_obj)

        # Save results
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
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "bge_m3_qwen_prediction.jsonl"))
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pipeline = Pipeline()
    pipeline.run(args.input, args.output)