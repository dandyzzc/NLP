import requests
import time
import json
import re

# ================= 配置 =================
DEFAULT_API_KEY = "" 
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# ================= 1. Prompts (从 rag_generation.py 移植) =================

# --- Basic Mode (现在升级为 CoT Mode) ---
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

# --- Feature A: Multi-Turn (UI 特有，保留) ---
REWRITE_PROMPT = """Given the following conversation history and a new follow-up question, rephrase the follow-up question to be a standalone question that contains all necessary context.

History:
{history}

Follow-up Question: {question}
Standalone Question:"""

# --- Feature B: Agentic Mode (从 rag_generation.py 移植) ---

# --- Step 1: Agentic Reasoning (Analyst) ---
# 改进点：强制要求输出 Reasoning 标签，供下一步使用
DECOMPOSE_PROMPT = """You are an expert Detective Analyst. 
Your goal is to answer the Question using the provided Documents.

Context:
{context}

Question: {question}

Instructions:
1. Identify the "Bridge Entity" connecting the documents.
2. Formulate a logical reasoning path.
3. If the answer is not explicitly stated, infer it reasonably from the context.
4. Format your output strictly as follows:

Reasoning: [Your step-by-step logic here]
Draft Answer: [The concise final entity/date/name]
"""

# --- Step 2: Agentic Refinement (Judge) ---
# 改进点：不再是盲目的 "Verify"，而是 "Review Reasoning"。
# 解决了 "Unknown" 问题：通过通过阅读 Step 1 的逻辑，模型有了上下文依赖，不敢轻易说 Unknown。
REFINE_PROMPT = """You are a Senior Editor. You have a Question, Documents, and a Draft Answer provided by an Analyst.

Context:
{context}

Question: {question}

Analyst's Reasoning: {reasoning}
Analyst's Draft Answer: {draft_answer}

Your Task:
1. Review the Analyst's reasoning. Does it follow the documents?
2. If the Analyst's answer is correct, OUTPUT IT AS IS.
3. If the Analyst said "unknown" but you can find the answer, output the CORRECT answer.
4. Only if the documents contain NO information at all, output "unknown".
5. Output ONLY the final concise answer string. Do not output "The answer is...".

Final Answer:"""

# ================= 2. 客户端与逻辑 =================
class SiliconFlowClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages, temperature=0.1, max_tokens=300):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        # 移植了你的 Retry 逻辑
        for _ in range(3):
            try:
                response = requests.post(API_URL, json=payload, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"API Request failed: {e}, retrying...")
                time.sleep(2)
        return "error"

class RAGGenerator:
    def __init__(self, api_key):
        self.client = SiliconFlowClient(api_key)

    def format_docs(self, docs, top_k=8):
        """
        格式化文档，带标题提取优化 (移植自 rag_generation.py)
        注意：app.py 传入的 docs 是 dict 列表 [{'id':..., 'text':...}]
        """
        context_str = ""
        for idx, item in enumerate(docs[:top_k]):
            # 兼容处理：app.py 传的是 dict
            doc_id = item.get('id', 'unknown')
            full_text = item.get('text', '').strip()
            
            title = doc_id
            content = full_text

            # 优化：提取 Title (你的核心逻辑)
            if "." in full_text:
                parts = full_text.split('.', 1)
                potential_title = parts[0].strip()
                if 0 < len(potential_title) < 80: 
                    title = potential_title
                    # content = full_text # 你原来的逻辑保留全文

            context_str += f"Document [{idx+1}](Title: {title}) {content}\n"
        return context_str

    def extract_cot_answer(self, text):
        """
        [New] 专门用于 Basic CoT 模式的答案提取 (移植自 rag_generation.py)
        """
        text = text.strip()
        # 1. 尝试分割 Answer:
        if "Answer:" in text:
            final_part = text.split("Answer:")[-1].strip()
        else:
            # 如果没有 Answer:，尝试取最后一行
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            final_part = lines[-1] if lines else ""

        # 2. 清洗末尾句号
        if final_part.endswith('.'):
            final_part = final_part[:-1]

        # 3. 清洗前缀
        prefixes = ["the answer is", "it is", "final answer:"]
        lower_part = final_part.lower()
        for prefix in prefixes:
            if lower_part.startswith(prefix):
                final_part = final_part[len(prefix):].strip()
                lower_part = final_part.lower()
        
        return final_part

    # --- Basic Mode (现在是 CoT) ---
    def generate_basic(self, question, docs):
        """
        返回最终答案字符串
        """
        context_str = self.format_docs(docs, top_k=8)
        
        user_content = f"{context_str}\n\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": user_content}
        ]
        
        # 增加 token 数以容纳 Thought
        raw_response = self.client.chat_completion(messages, max_tokens=300)
        
        # 提取答案
        final_answer = self.extract_cot_answer(raw_response)
        
        return final_answer

    # --- Feature A: Rewrite (UI Logic) ---
    def rewrite_query(self, user_input, history):
        if not history:
            return user_input
        
        hist_text = ""
        for msg in history:
            role = "Q" if msg['role'] == "user" else "A"
            hist_text += f"{role}: {msg['content']}\n"
            
        msg = [{"role": "user", "content": REWRITE_PROMPT.format(history=hist_text, question=user_input)}]
        new_query = self.client.chat_completion(msg, max_tokens=100)
        
        if ":" in new_query:
            new_query = new_query.split(":")[-1].strip()
        return new_query

    # --- Feature B: Agentic ---
    def generate_agentic(self, question, docs):
        """
        Improved Agentic Workflow: Reasoning -> Critique/Refine
        解决 Verify 阶段老是输出 unknown 的问题
        """
        context_str = self.format_docs(docs, top_k=8)
        steps_log = ""

        # --- Step 1: Reasoning & Draft (The Analyst) ---
        steps_log += "### Step 1: Detective Analysis (Decomposition)\n"
        
        decompose_msg = [
            {"role": "user", "content": DECOMPOSE_PROMPT.format(context=context_str, question=question)}
        ]
        
        # 使用稍高的 temperature (0.3) 让 Step 1 有一点联想能力，避免过早死板
        raw_analysis = self.client.chat_completion(decompose_msg, max_tokens=400, temperature=0.3)
        steps_log += f"{raw_analysis}\n\n"

        # 解析 Step 1 的输出
        reasoning = "No reasoning provided."
        draft_answer = "unknown"

        # 使用正则提取 Reasoning 和 Draft Answer
        r_match = re.search(r"Reasoning:\s*(.*?)(?=Draft Answer:|$)", raw_analysis, re.DOTALL | re.IGNORECASE)
        d_match = re.search(r"Draft Answer:\s*(.*)", raw_analysis, re.DOTALL | re.IGNORECASE)

        if r_match:
            reasoning = r_match.group(1).strip()
        if d_match:
            draft_answer = d_match.group(1).strip()

        # UI 显示提取结果
        steps_log += f"**Extracted Reasoning:** `{reasoning[:100]}...`\n"
        steps_log += f"**Extracted Draft:** `{draft_answer}`\n"

        # --- Step 2: Refinement (The Editor) ---
        # 这一步的关键是把 "Reasoning" 喂给模型，让它去 check 逻辑，而不是重新做一遍题
        steps_log += "\n### Step 2: Editor Review (Reasoning Check)\n"
        
        refine_msg = [
            {"role": "user", "content": REFINE_PROMPT.format(
                context=context_str, 
                question=question, 
                reasoning=reasoning,
                draft_answer=draft_answer
            )}
        ]
        
        # 使用极低的 temperature 确保最终输出稳定
        final_raw = self.client.chat_completion(refine_msg, max_tokens=100, temperature=0.1)
        
        # 清洗最终答案
        final_answer = self.extract_cot_answer(final_raw)
        
        # --- Fallback Mechanism (兜底策略) ---
        # 如果 Step 2 居然变成了 unknown，但 Step 1 给了明确的答案，
        # 且 Step 1 的答案不是 "unknown"，我们选择相信 Step 1 (因为 Step 2 可能过于保守)
        if "unknown" in final_answer.lower() and "unknown" not in draft_answer.lower():
            steps_log += f"⚠️ Editor returned 'unknown', falling back to Draft Answer: {draft_answer}\n"
            final_answer = self.extract_cot_answer(draft_answer)
        else:
            steps_log += f"**Editor Decision:** {final_answer}\n"
        
        return final_answer, steps_log
