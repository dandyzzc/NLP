import json
import os
import argparse
import requests
import time
import re  # Added: for regex parsing
from tqdm import tqdm
from typing import List, Dict, Any

# ================= Configuration =================

DEFAULT_API_KEY = "sk-xxxxxx" 
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# Path configuration
DATA_DIR = "./data"
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.jsonl") 
INPUT_RETRIEVAL_FILE = "./result/retrieval/hybrid_qwen_prediction.jsonl"

OUTPUT_PRED_FILE = "./result/final_prediction_k8_cot.jsonl"

# ================= 1. API Client =================
class SiliconFlowClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages, temperature=0.1, max_tokens=200):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        for _ in range(3):
            try:
                response = requests.post(API_URL, json=payload, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"API Request failed: {e}, retrying...")
                time.sleep(2)
        return "error"

# ================= 2. Prompt Templates =================

# --- Basic Mode (CoT) ---
SYSTEM_PROMPT_COT = """You are answering questions using only the provided documents.

Rules:
1. First, think step-by-step to find the connection between documents (Bridge Entity).
2. Then, output the final answer after the prefix "Answer:".
3. The answer must be a concise phrase (Entity, Date, Yes/No).
4. NEVER output "unknown". Guess based on context if needed.

For example:
Document [1](Title: Ed Wood) Edward Davis Wood Jr. was an American filmmaker.
Document [2](Title: Scott Derrickson) Scott Derrickson is an American director.
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Thought: Ed Wood was American. Scott Derrickson is American. They match.
Answer: yes

Document [1](Title: Shirley Temple) Shirley Temple Black was an American diplomat.
Document [2](Title: Kiss and Tell (1945 film)) Kiss and Tell is a film starring Shirley Temple as Corliss Archer.
Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Thought: Corliss Archer was played by Shirley Temple. Document 1 says Shirley Temple was a diplomat/Ambassador. The question asks for the position. Based on common context, the specific title is Chief of Protocol.
Answer: Chief of Protocol

Document [1](Title: Animorphs) Animorphs is a science fantasy series written by Katherine Applegate.
Document [2](Title: The Hork-Bajir Chronicles) The Hork-Bajir Chronicles is a companion book to the "Animorphs" series.
Question: What science fantasy young adult series has a set of companion books?
Thought: The Hork-Bajir Chronicles is a companion to Animorphs. Animorphs is the series.
Answer: Animorphs"""

# --- Agentic Prompts (Analyst -> Editor) ---

# Step 1: Detective Analyst (Analyst) - Force output of reasoning process
DECOMPOSE_PROMPT_AGENT = """You are an expert Detective Analyst. 
Your goal is to answer the Question using the provided Documents.

Context:
{context}

Question: {question}

Instructions:
1. Identify the "Bridge Entity" connecting the documents if this is a multi-hop question.
2. Formulate a logical reasoning path based on the text.
3. If the answer is not explicitly stated, infer it reasonably from the context.
4. Format your output strictly as follows:

Reasoning: [Your step-by-step logic here]
Draft Answer: [The concise final entity/date/name/Yes/No]
"""

# Step 2: Senior Editor (Editor) - Check logic rather than simple matching
VERIFY_PROMPT_AGENT = """You are a Senior Editor. You have a Question, Documents, and a Draft Answer provided by an Analyst.

Context:
{context}

Question: {question}

Analyst's Reasoning: {reasoning}
Analyst's Draft Answer: {draft_answer}

Your Task:
1. Review the Analyst's reasoning. Does it logically follow the documents?
2. If the Analyst's answer is correct or reasonable, OUTPUT IT AS IS.
3. If the Analyst said "unknown" but you can find the answer in the documents, output the CORRECT answer.
4. Only if the documents contain NO information at all, output "unknown".
5. Output ONLY the final concise answer string. Do not output "The answer is...".

Final Answer:"""

# ================= 3. RAG System Core Class =================
class RAGSystem:
    def __init__(self, api_key, collection_dict):
        self.client = SiliconFlowClient(api_key)
        self.collection = collection_dict

    def format_docs(self, retrieved_docs_list, top_k=8):
        """Format documents with title extraction optimization"""
        context_str = ""
        for idx, item in enumerate(retrieved_docs_list[:top_k]):
            doc_id = item[0]
            full_text = self.collection.get(doc_id, "").strip()
            
            title = doc_id
            content = full_text

            # Optimization: Extract Title
            if "." in full_text:
                parts = full_text.split('.', 1)
                potential_title = parts[0].strip()
                if 0 < len(potential_title) < 80: 
                    title = potential_title
                    content = full_text 

            context_str += f"Document [{idx+1}](Title: {title}) {content}\n"
        return context_str

    def extract_cot_answer(self, text):
        """Specifically for Basic CoT mode answer extraction"""
        text = text.strip()
        if "Answer:" in text:
            final_part = text.split("Answer:")[-1].strip()
        else:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            final_part = lines[-1] if lines else ""

        if final_part.endswith('.'):
            final_part = final_part[:-1]

        prefixes = ["the answer is", "it is", "final answer:"]
        lower_part = final_part.lower()
        for prefix in prefixes:
            if lower_part.startswith(prefix):
                final_part = final_part[len(prefix):].strip()
                lower_part = final_part.lower()
        
        return final_part

    def _clean_final_answer(self, text):
        """[Helper] Deep cleaning for Agentic mode"""
        text = text.strip()
        if text.endswith('.'):
            text = text[:-1]
        
        lower_text = text.lower()
        prefixes = [
            "final answer:", "answer:", "the answer is", "it is", 
            "based on the documents,", "corrected answer:"
        ]
        
        for prefix in prefixes:
            if lower_text.startswith(prefix):
                text = text[len(prefix):].strip()
                lower_text = text.lower()
        
        return text

    # --- Basic Mode (CoT) ---
    def generate_basic(self, question, retrieved_docs):
        context_str = self.format_docs(retrieved_docs, top_k=8)
        
        user_content = f"{context_str}\n\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": user_content}
        ]
        
        # Increase token count to accommodate Thought
        raw_response = self.client.chat_completion(messages, max_tokens=300)
        
        final_answer = self.extract_cot_answer(raw_response)
        
        return final_answer

    # --- Agentic Mode: Analyst -> Editor ---
    def generate_agentic(self, question, retrieved_docs):
        context_str = self.format_docs(retrieved_docs, top_k=8)
        
        # Step 1: Analyst (Decompose & Draft)
        decompose_msg = [
            {"role": "user", "content": DECOMPOSE_PROMPT_AGENT.format(context=context_str, question=question)}
        ]
       
        raw_analysis = self.client.chat_completion(decompose_msg, max_tokens=350, temperature=0.3)
        
        # Parse Step 1 output (more robust regex extraction)
        reasoning = "No reasoning provided."
        draft_answer = "unknown"

        r_match = re.search(r"Reasoning:\s*(.*?)(?=Draft Answer:|$)", raw_analysis, re.DOTALL | re.IGNORECASE)
        if r_match:
            reasoning = r_match.group(1).strip()
        
        d_match = re.search(r"Draft Answer:\s*(.*)", raw_analysis, re.DOTALL | re.IGNORECASE)
        if d_match:
            draft_answer = d_match.group(1).strip()
        else:
            # Fallback: if no label found, take the last line
            lines = [l.strip() for l in raw_analysis.split('\n') if l.strip()]
            if lines: draft_answer = lines[-1]

        # Step 2: Editor (Context-Aware Verification)
        verify_msg = [
            {"role": "user", "content": VERIFY_PROMPT_AGENT.format(
                context=context_str, 
                question=question, 
                reasoning=reasoning, 
                draft_answer=draft_answer
            )}
        ]
        # Step 2 maintains low temperature (0.1), only for judgment
        final_raw = self.client.chat_completion(verify_msg, max_tokens=100, temperature=0.1)
        
        # Clean answer
        final_answer = self._clean_final_answer(final_raw)

        # --- Fallback Mechanism (Intelligent fallback) ---
        # If Editor outputs unknown too conservatively, but Analyst actually found the answer, fall back to Analyst's answer
        is_final_unknown = "unknown" in final_answer.lower()
        is_draft_unknown = "unknown" in draft_answer.lower()

        if is_final_unknown and not is_draft_unknown:
            # Consider Step 2 as false positive, fall back
            final_answer = self._clean_final_answer(draft_answer)
        
        return final_answer

# ================= 4. Main Program =================
def load_collection():
    print(f"Loading collection from {COLLECTION_PATH}...")
    collection = {}
    if not os.path.exists(COLLECTION_PATH):
        print(f"Error: Collection file {COLLECTION_PATH} not found!")
        return {}
        
    with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                collection[item['id']] = item['text']
            except json.JSONDecodeError:
                continue
    return collection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "agentic"], help="Generation mode")
    parser.add_argument("--input", type=str, default=INPUT_RETRIEVAL_FILE, help="Input retrieval file")
    parser.add_argument("--output", type=str, default=OUTPUT_PRED_FILE, help="Output prediction file")
    args = parser.parse_args()

    # 1. Load data
    collection = load_collection()
    if not collection: return

    rag = RAGSystem(args.api_key, collection)

    if not os.path.exists(args.input):
        print(f"Retrieval input file {args.input} not found.")
        return

    data_list = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))

    print(f"Processing {len(data_list)} items in {args.mode} mode...")
    
        # 2. Generate predictions
    results = []
    for item in tqdm(data_list):
        qid = item['id']
        question = item['question']
        retrieved_docs = item['retrieved_docs']
        
        try:
            if args.mode == "basic":
                ans = rag.generate_basic(question, retrieved_docs)
            else:
                ans = rag.generate_agentic(question, retrieved_docs)
        except Exception as e:
            print(f"Error on {qid}: {e}")
            ans = "error"
            
        
        results.append({
            "id": qid,
            "question": question,         
            "answer": ans,
            "retrieved_docs": retrieved_docs[:10] 
        })
        

    # 3. Save results
    print(f"Saving predictions to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()