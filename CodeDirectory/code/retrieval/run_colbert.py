import json
import os
import torch
import shutil
import voyager
from tqdm import tqdm
from pylate import indexes, models, retrieve
from ragatouille import RAGPretrainedModel

# Path configurations
DATA_DIR = "./data"
OUTPUT_FILE = "colbert.jsonl"
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl")
QUERY_PATH = os.path.join(DATA_DIR, "validation.jsonl")

# Index storage
INDEX_DIR = "./indices/mxbai_plaid_index"
INDEX_NAME = "collection_index"

# Model settings
RETRIEVER_MODEL = "mixedbread-ai/mxbai-edge-colbert-v0-17m"
RERANKER_MODEL = "answerdotai/answerai-colbert-small-v1"

# Pipeline configs
RETRIEVAL_TOP_K = 50   
RERANK_TOP_K = 10      
BATCH_SIZE = 32        

def load_jsonl(path):
    """Load data from a JSONL file"""
    data = []
    print(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # Stage 1: Mxbai Index Construction (Voyager Backend)
    print(f"=== Stage 1: Retrieval with {RETRIEVER_MODEL} ===")
    
    corpus_data = load_jsonl(COLLECTION_PATH)
    query_data = load_jsonl(QUERY_PATH)
    
    corpus_map = {str(doc['id']): doc['text'] for doc in corpus_data}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever_model = models.ColBERT(
        model_name_or_path=RETRIEVER_MODEL,
        device=device
    )

    # Check if index exists
    index_exists = os.path.exists(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0
    
    # Initialize Voyager index
    index = indexes.Voyager(
        index_folder=INDEX_DIR,
        index_name=INDEX_NAME,
        override=not index_exists
    )

    if index_exists:
        print(f"Loading existing index from {INDEX_DIR}...")
    else:
        print("Index not found. Encoding corpus...")
        
        # Critical: Adjust index dimension to 48 for the retriever model
        print(f"Patching Voyager index dimension to 48 for {RETRIEVER_MODEL}...")
        index.index = voyager.Index(voyager.Space.InnerProduct, num_dimensions=48)

        doc_texts = [doc['text'] for doc in corpus_data]
        doc_ids = [str(doc['id']) for doc in corpus_data]
        
        doc_embeddings = retriever_model.encode(
            doc_texts,
            batch_size=BATCH_SIZE,
            is_query=False,
            show_progress_bar=True
        )
        
        print("Adding documents to index...")
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=doc_embeddings
        )
        
        # Free memory
        del doc_embeddings
        torch.cuda.empty_cache()


    # Stage 2: Retrieval
    print("Encoding queries...")
    query_texts = [q['text'] for q in query_data]
    
    # Encode queries in batches
    query_embeddings = retriever_model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        is_query=True,
        show_progress_bar=True
    )
    
    retriever = retrieve.ColBERT(index=index)
    
    print(f"Retrieving top {RETRIEVAL_TOP_K} candidates...")
    retrieval_results = []
    
    # Retrieve for each query embedding
    for q_emb in tqdm(query_embeddings, desc="Searching Index"):
        single_res = retriever.retrieve(
            queries_embeddings=[q_emb], 
            k=RETRIEVAL_TOP_K
        )
        retrieval_results.append(single_res[0])
    
    # Clean up to prevent OOM
    del retriever_model
    del query_embeddings
    del retriever
    del index
    torch.cuda.empty_cache()
    
    # Stage 3: AnswerAI Rerank
    print(f"\n=== Stage 2: Reranking with {RERANKER_MODEL} ===")
    
    reranker = RAGPretrainedModel.from_pretrained(RERANKER_MODEL)

    final_results = []
    
    for i in tqdm(range(len(query_data)), desc="Reranking"):
        qid = query_data[i]['id']
        q_text = query_texts[i]
        
        # Get current retrieval candidates
        current_candidates = retrieval_results[i]
        candidate_ids = [str(item['id']) for item in current_candidates]
        
        # Fetch valid candidate documents
        candidate_docs = []
        valid_candidate_ids = []
        
        for did in candidate_ids:
            if did in corpus_map:
                candidate_docs.append(corpus_map[did])
                valid_candidate_ids.append(did)
        
        if not candidate_docs:
             final_results.append({
                "id": qid, 
                "question": q_text, 
                "answer": query_data[i].get("answer", ""),
                "retrieved_docs": []
            })
             continue

        # Rerank candidates
        rerank_output = reranker.rerank(
            query=q_text,
            documents=candidate_docs,
            k=RERANK_TOP_K
        )
        
        formatted_docs = []
        for item in rerank_output:
            original_idx = item['result_index']
            original_doc_id = valid_candidate_ids[original_idx]
            score = float(item['score'])
            
            # Try converting ID to int if applicable
            try:
                if isinstance(query_data[0]['id'], int) or str(original_doc_id).isdigit():
                    original_doc_id = int(original_doc_id)
            except:
                pass
                
            formatted_docs.append([original_doc_id, score])
            
        final_results.append({
            "id": qid,
            "question": q_text,
            "answer": query_data[i].get("answer", ""),
            "retrieved_docs": formatted_docs
        })

    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()