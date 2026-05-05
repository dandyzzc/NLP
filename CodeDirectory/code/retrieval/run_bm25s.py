import json
import os
import bm25s
import Stemmer
from ragatouille import RAGPretrainedModel
from tqdm import tqdm

# Path Configurations
DATA_DIR = "./data"
OUTPUT_FILE = "bm25s_prediction.jsonl"
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
QUERY_PATH = os.path.join(DATA_DIR, "validation.jsonl")

INDEX_DIR = "./indices/bm25s"

# Retrieval hyperparameters
BM25_TOP_K = 50
RERANK_TOP_K = 10
BATCH_SIZE = 32

def load_jsonl(path):
    """Load JSONL file into list of dicts"""
    data = []
    print(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # Load data
    corpus_data = load_jsonl(COLLECTION_PATH)
    query_data = load_jsonl(QUERY_PATH)

    # Map corpus IDs and texts
    corpus_ids = [doc['id'] for doc in corpus_data]
    corpus_map = {doc['id']: doc['text'] for doc in corpus_data}

    # Initialize stemmer for tokenization
    stemmer = Stemmer.Stemmer("english")

    # Load or build BM25 index
    print("Checking for existing BM25 index...")
    if os.path.exists(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0:
        print(f"Loading existing BM25 index from {INDEX_DIR}...")
        retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=False)
    else:
        print("Index not found. Building from scratch...")
        corpus_texts = [doc['text'] for doc in corpus_data]
        
        print("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
        
        print("Indexing corpus...")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        print(f"Saving index to {INDEX_DIR}...")
        os.makedirs(INDEX_DIR, exist_ok=True)
        retriever.save(INDEX_DIR)
    
    # Initialize reranker
    print("Loading ColBERT model...")
    reranker = RAGPretrainedModel.from_pretrained("answerdotai/answerai-colbert-small-v1")

    # Retrieve and rerank
    print(f"Starting retrieval for {len(query_data)} queries...")
    output_results = []
    
    query_texts = [q['text'] for q in query_data]
    query_ids = [q['id'] for q in query_data]
    
    # BM25 retrieval
    query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer)
    bm25_docs, bm25_scores = retriever.retrieve(query_tokens, k=BM25_TOP_K)
    
    # Process each query
    for i in tqdm(range(len(query_data)), desc="Reranking"):
        qid = query_ids[i]
        q_text = query_texts[i]
        retrieved_indices = bm25_docs[i]
        
        candidate_docs = []
        candidate_ids = []
        for idx in retrieved_indices:
            doc_id = corpus_ids[idx]
            candidate_ids.append(doc_id)
            candidate_docs.append(corpus_map[doc_id])
        
        # Rerank with ColBERT
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

    # Save results
    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in output_results:
            f.write(json.dumps(item) + "\n")
    
    print("Done!")

if __name__ == "__main__":
    main()