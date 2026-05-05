# COMP5423 Group Project: Retrieval-Augmented Generation (RAG) System

This repository contains the implementation of an end-to-end RAG system for the **HotpotQA (HQ-small)** dataset. The system integrates advanced retrieval techniques (Sparse, Dense, Hybrid) with a Generative LLM (Qwen2.5-7B via SiliconFlow API) to answer multi-hop complex queries.

The project implements:



**Basic RAG:** Single-turn QA with Chain-of-Thought (CoT).

**Feature A:** Multi-turn interaction with query rewriting.

**Feature B:** Agentic workflow (Analyst →→Editor) for reasoning and verification.

## 📂 Project Structure

```text
.
├── code
│   ├── evaluation
│   │   ├── eval_hotpotqa.py              # Evaluation script for HotpotQA-style QA (end-to-end RAG quality)
│   │   └── eval_retrieval.py             # Evaluation script for retrieval (e.g., recall / nDCG on candidate sets)
│   │
│   ├── generation  
│   │   └── rag_generation.py             # RAG answer generation with Qwen2.5-7B-Instruct (SiliconFlow API),
│   │                                     # supporting basic CoT and agentic (Analyst→Editor) reasoning over retrieved docs
│   │
│   └── retrieval
│       ├── run_bge_large.py              # BGE-large-en-v1.5 + answerai-colbert-small-v1 (reranker)
│       ├── run_bge_m3.py                 # BGE-M3 + answerai-colbert-small-v1 (reranker)
│       ├── run_bge_m3_hybrid.py          # BGE-M3 (Hybrid Mode:Dense + Sparse) + answerai-colbert-small-v1 (reranker)
│       ├── run_bge_m3_qwen_hybrid.py     # BGE-M3 (Hybrid Mode:Dense + Sparse) + Qwen3-Reranker-0.6B (reranker)
│       ├── run_bm25s.py                  # BM25S
│       ├── run_bm25s_bge_large_hybrid.py # Hybrid: BM25S + BGE-large-en-v1.5 +answerai-colbert-small-v1 (reranker)
│       ├── run_bm25s_bge_large_qwen_hybrid.py # Hybrid: BM25S + BGE-large-en-v1.5 + Qwen3-Reranker-0.6B (reranker)
│       ├── run_colbert.py                # mxbai-edge-colbert-v0-17m
│       ├── run_model2vec.py              # potion-base-8M
│       └── run_qwen3.py                  # Qwen3-Embedding-0.6B
│
├── data/                                 # Datasets
│   ├── collection.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── indices/                              #index files
│
│
├── result/
│   ├── qa/                               # Saved QA results
│   ├── retrieval/                        # Saved retrieval results
│   └── test_predict.jsonl                # final test prediction file
│
├── app.py                                # Streamlit front-end entry point for interactive demo
├── backend.py                            # Backend RAG engine (retrievers, rerankers, and generator orchestration)
├── retriever.py                          # Base Retreiver, 
│                                         # In this project ,we choose BM25S + BGE-large-en-v1.5 +answerai-colbert-small-v1
├── flowchart.png                         # Pipeline / system diagram
├── README.md                             
└── requirements.txt                      

```

## 🛠️ Environment Setup

Ensure you have Python 3.8+ installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

*Key libraries include: `bm25s`, `pystemmer`, `ragatouille`, `requests`, `streamlit`, `tqdm`.*

---

## 🚀 Usage Guide

### 1. Retrieval Module
The retrieval scripts are located in `code/retrieval/`. These scripts handle indexing the `collection.jsonl`, retrieving candidates, and reranking them (e.g., using ColBERT).

**Example: Running BM25s + ColBERT Reranker**
The script automatically builds the index if it doesn't exist in `./indices/bm25s`.

```bash
python code/retrieval/run_bm25s.py
```

**What it produces:**
1.  **Index:** Saves the BM25 index to `./indices/bm25s/`.
2.  **Output File:** A JSONL file in `./result/retrieval/` (e.g., `bm25s_prediction.jsonl`).
    *   **Format:** Complies with project submission requirements.
    ```json
    {
      "id": "q_1",
      "question": "What government position...",
      "answer": "Chief of Protocol",
      "retrieved_docs": [["doc_id_1", 0.98], ["doc_id_2", 0.85], ...]
    }
    ```

### 2. Generation Module
Use `code/generation/rag_generation.py` to generate answers for the retrieved documents. This script supports both **Basic (CoT)** and **Agentic (Feature B)** modes using the Qwen2.5-7B-Instruct model **(SiliconFlow API)**.

> **Note:** You must set your `DEFAULT_API_KEY` in the script or pass it via CLI.

**Option A: Basic Mode (Chain-of-Thought)**
Standard RAG generating a thought process followed by the answer.

```bash
python code/generation/rag_generation.py \
    --api_key "sk-xxxxxx" \
    --mode basic \
    --input "./result/retrieval/bm25s_prediction.jsonl" \
    --output "./result/qa/final_prediction_basic.jsonl"
```

**Option B: Agentic Workflow (Feature B)**
Runs the "Analyst $\to$ Editor" pipeline:
1.  **Analyst:** Decomposes the question and drafts a reasoning path.
2.  **Editor:** Verifies reasoning against documents and corrects hallucinations.

```bash
python code/generation/rag_generation.py \
    --api_key "sk-xxxxxx" \
    --mode agentic \
    --input "./result/retrieval/bm25s_prediction.jsonl" \
    --output "./result/qa/final_prediction_agentic.jsonl"
```

**What it produces:**

*   **Final Prediction File:** A JSONL file (`final_prediction_*.jsonl`) containing the generated answer string and reference documents, ready for evaluation.

### 3. Interactive Web UI
A Streamlit-based interface to demonstrate the system, supporting all features described in the project instruction.

**Run the App:**
```bash
streamlit run app.py
```

**Key Features in UI:**
*   **Configuration:** Input API Key and select Retrieval/Reranker models dynamically.
*   **Feature A (Multi-Turn):** Select "Multi-Turn" mode. The system maintains conversation history and rewrites follow-up queries (e.g., "What about his wife?") into standalone questions before retrieval.
*   **Feature B (Agentic):** Select "Agentic Workflow". The UI visualizes the intermediate steps:
    *   Step 1: Detective Analysis (Reasoning & Draft)
    *   Step 2: Editor Review (Verification)
*   **Evidence Display:** Shows the retrieved documents (with titles and scores) used to generate the answer.

---

## 📊 Evaluation & Submission

### 1. Metrics
Use the scripts in `code/evaluation/` to measure performance:
*   **Answer Accuracy:** Exact Match (EM) / F1.
*   **Retrieval Quality:** nDCG@10, Recall.

```bash
# Example evaluation
python code/evaluation/eval_hotpotqa.py ----gold data/validation.jsonl --pred_file result/qa/final_prediction.jsonl
```

### 2. Submission Format
For the final project submission, the output file (e.g., `result/qa/test_prediction.jsonl`) will follow the exact format required by the TAs:

```json
{
    "id": "5a8b57f25542995d1e6f1371", 
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?", 
    "answer": "yes", 
    "retrieved_docs": [
        ["Ed Wood", 0.9123],
        ["Scott Derrickson", 0.8912],
        ... (up to 10 docs)
    ]
}
```

