import json
import os
import argparse
from annotated_types import doc
import numpy as np
import faiss
import torch
import bm25s
import Stemmer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Path configurations
DATA_DIR = "./data"
OUTPUT_DIR = "./result"
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")

BM25_INDEX_DIR = "./indices/bm25s"
BGE_INDEX_DIR = "./indices/bge_large"
BGE_INDEX_FILE = os.path.join(BGE_INDEX_DIR, "bge_faiss.index")

# Model configurations
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

RETRIEVAL_TOP_K = 50   
RERANK_TOP_K = 10      
BATCH_SIZE = 32        
RERANK_BATCH_SIZE = 2  

class HybridRetrievalSystem:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.corpus_ids = []
        self.id2text = {}
        self.bm25_retriever = None
        self.dense_index = None
        self.embed_model = None
        
        # Reranker components
        self.rerank_model = None
        self.rerank_tokenizer = None
        self.rerank_prefix_tokens = None
        self.rerank_suffix_tokens = None
        self.token_true_id = None
        self.token_false_id = None
        
        self.stemmer = Stemmer.Stemmer("english")
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
            ids = [self.corpus_ids[idx] for idx in doc_indices[i]]
            results.append(ids)
        return results

    # Dense (BGE) Module
    def init_dense(self):
        print("\n[Initializing Dense Retrieval]")
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        if os.path.exists(BGE_INDEX_FILE):
            print(f"Loading FAISS index from {BGE_INDEX_FILE}...")
            self.dense_index = faiss.read_index(BGE_INDEX_FILE)
        else:
            print("Building FAISS index...")
            os.makedirs(BGE_INDEX_DIR, exist_ok=True)
            corpus_texts = list(self.id2text.values())
            embeddings = self.embed_model.encode(
                corpus_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            self.dense_index = faiss.IndexFlatIP(dim)
            self.dense_index.add(embeddings.astype('float32'))
            faiss.write_index(self.dense_index, BGE_INDEX_FILE)
            print("FAISS index saved.")

    def retrieve_dense(self, query_texts, k=50):
        print("Running Dense retrieval...")
        instruction = "Represent this sentence for searching relevant passages: "
        q_texts_with_instruction = [instruction + q for q in query_texts]
        q_embs = self.embed_model.encode(
            q_texts_with_instruction, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
        )
        scores, indices = self.dense_index.search(q_embs.astype('float32'), k)
        results = []
        for i in range(len(query_texts)):
            ids = []
            for idx in indices[i]:
                if idx != -1:
                    ids.append(self.corpus_ids[idx])
            results.append(ids)
        return results

    # Qwen Rerank Module
    def init_reranker(self):
        print(f"\n[Loading Reranker: {RERANK_MODEL_NAME}]")
        
        # Load tokenizer
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME, padding_side='left')
        
        # Load model with fp16 for memory efficiency
        self.rerank_model = AutoModelForCausalLM.from_pretrained(
            RERANK_MODEL_NAME, 
            torch_dtype=torch.float16,
            device_map="cuda" if self.device == 'cuda' else "cpu"
        ).eval()

        # Prepare prompt tokens
        self.token_false_id = self.rerank_tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.rerank_tokenizer.convert_tokens_to_ids("yes")
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
        
        self.rerank_prefix_tokens = self.rerank_tokenizer.encode(prefix, add_special_tokens=False)
        self.rerank_suffix_tokens = self.rerank_tokenizer.encode(suffix, add_special_tokens=False)
        
        print("Qwen Reranker loaded.")

    def _format_qwen_instruction(self, query, doc, instruction=None):
        if instruction is None:
            instruction = "Given a multi-hop reasoning question requiring evidence from multiple Wikipedia passages, retrieve the most relevant supporting passages that contain the bridging entities, comparison facts, or direct evidence needed to answer the query."
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )

    def _process_qwen_inputs(self, pairs, max_length=1024):
        # Tokenize instructions
        inputs = self.rerank_tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=max_length - len(self.rerank_prefix_tokens) - len(self.rerank_suffix_tokens)
        )
        
        # Add prefix and suffix tokens
        final_input_ids = [self.rerank_prefix_tokens + ele + self.rerank_suffix_tokens for ele in inputs['input_ids']]
        
        # Pad inputs
        padded_inputs = self.rerank_tokenizer.pad(
            {"input_ids": final_input_ids}, 
            padding=True, 
            return_tensors="pt", 
            max_length=max_length
        )
        
        # Move to model device
        for key in padded_inputs:
            padded_inputs[key] = padded_inputs[key].to(self.rerank_model.device)
            
        return padded_inputs

    def rerank(self, query, doc_ids, k=10):
        """Rerank documents using Qwen3 Reranker"""
        doc_ids_clean = [did for did in doc_ids if did in self.id2text]
        if not doc_ids_clean:
            return []
        
        doc_texts = [self.id2text[did] for did in doc_ids_clean]
        
        # Prepare input pairs with instruction
        instruction = "Given a multi-hop reasoning question requiring evidence from multiple Wikipedia passages, retrieve the most relevant supporting passages that contain the bridging entities, comparison facts, or direct evidence needed to answer the query."
        formatted_pairs = [self._format_qwen_instruction(query, doc, instruction) for doc in doc_texts]
        
        scores = []
        
        # Batch processing to avoid OOM
        with torch.no_grad():
            for i in range(0, len(formatted_pairs), RERANK_BATCH_SIZE):
                batch_pairs = formatted_pairs[i : i + RERANK_BATCH_SIZE]
                
                inputs = self._process_qwen_inputs(batch_pairs)
                outputs = self.rerank_model(** inputs)
                batch_logits = outputs.logits[:, -1, :]  # Last token logits
                
                # Extract yes/no scores
                true_vector = batch_logits[:, self.token_true_id]
                false_vector = batch_logits[:, self.token_false_id]
                
                # Calculate probabilities
                result_stack = torch.stack([false_vector, true_vector], dim=1)
                result_log_probs = torch.nn.functional.log_softmax(result_stack, dim=1)
                batch_scores = result_log_probs[:, 1].exp().tolist()
                scores.extend(batch_scores)

        # Sort and return top k
        final_results = [[doc_ids_clean[i], score] for i, score in enumerate(scores)]
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:k]

# RRF fusion for combining two retrieval results
def rrf_fusion(list1, list2, k=60):
    scores = defaultdict(float)
    for rank, doc_id in enumerate(list1):
        scores[doc_id] += 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(list2):
        scores[doc_id] += 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# Main process
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(DATA_DIR, "validation.jsonl"))
    parser.add_argument("--output", type=str, default=os.path.join(OUTPUT_DIR, "hybrid_qwen_prediction.jsonl"))
    args = parser.parse_args()

    # Initialize system
    system = HybridRetrievalSystem()
    system.init_bm25()
    system.init_dense()
    system.init_reranker()

    # Load queries
    print(f"\nLoading queries from {args.input}...")
    queries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    
    query_texts = [q['text'] for q in queries]
    query_ids = [q['id'] for q in queries]

    # Retrieval phase
    bm25_results_list = system.retrieve_bm25(query_texts, k=RETRIEVAL_TOP_K)
    dense_results_list = system.retrieve_dense(query_texts, k=RETRIEVAL_TOP_K)

    # Reranking phase
    print(f"\nStarting Fusion and Reranking for {len(queries)} queries...")
    final_output = []
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for i in tqdm(range(len(queries))):
        qid = query_ids[i]
        q_text = query_texts[i]
        
        # Fusion with RRF
        candidates = rrf_fusion(bm25_results_list[i], dense_results_list[i])
        rerank_candidates = candidates[:100] 
        
        # Rerank and get top results
        top_docs = system.rerank(q_text, rerank_candidates, k=RERANK_TOP_K)
        
        final_output.append({
            "id": qid,
            "question": q_text,
            "answer": queries[i].get("answer", ""),
            "retrieved_docs": top_docs
        })

    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final_output:
            f.write(json.dumps(item) + "\n")
    print("Done!")

if __name__ == "__main__":
    main()