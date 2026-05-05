import streamlit as st
import time
import json
from backend import RAGGenerator
from retriever import HybridRetrievalSystem, rrf_fusion

# ================= Page Configuration =================
st.set_page_config(
    page_title="HotpotQA RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔎 HotpotQA RAG System Demo")
st.markdown("COMP5423 Group Project | Supported by Qwen2.5-7B")

# ================= Sidebar: Settings & Modes =================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("SiliconFlow API Key", type="password", help="Enter your API Key here")
    if not api_key:
        st.warning("Please enter API Key to proceed.")
        st.stop()
        
    st.divider()
    
    # --- 🧬 Model Selection Area (UI Display Only - Actual logic in retriever.py) ---
    st.subheader("🧬 Model Selection")
    
    retrieval_options = [
        "Sparse: BM25s",
        "Static Embedding: model2vec (minishlab/potion-base-8M)",
        "Dense (Encoder): BGE-large-en-v1.5",
        "Dense (Encoder): BGE-M3",
        "Dense (Instruction): Qwen3-Embedding-0.6B",
        "Multi-vector: mixedbread-ai/mxbai-edge-colbert-v0-17m",
        "Hybrid: BGE-M3 (Sparse+Dense)",
        "Hybrid: BM25s + BGE-large-en-v1.5"
    ]
    
    selected_retrieval = st.selectbox(
        "Retrieval Methodology",
        options=retrieval_options,
        index=7,
        help="Select the retrieval architecture."
    )

    reranker_options = [
        "answerdotai/answerai-colbert-small-v1",
        "Qwen/Qwen3-Reranker-0.6B"
    ]
    
    selected_reranker = st.selectbox(
        "Reranker Model",
        options=reranker_options,
        index=1,
        help="Select the cross-encoder model."
    )
    
    st.divider()
    
    # --- 🤖 System Mode ---
    st.subheader("🤖 System Mode")
    mode = st.radio(
        "Select Operation Mode:",
        (
            "Basic Single-Turn",   
            "Feature A: Multi-Turn", 
            "Feature B: Agentic Workflow"
        )
    )
    
    st.divider()
    
    # Parameter tuning
    top_k_retrieval = st.slider("Retrieval Top-K (Search)", 10, 100, 50)
    # top_k set to 8 to match backend prompt context window
    top_k_gen = st.slider("Generation Top-K (Context)", 1, 15, 8) 

# ================= Initialize System (Cached for Performance) =================
@st.cache_resource
def load_retrieval_system():
    with st.spinner("Loading Retrieval System (Indices & Models)..."):
        system = HybridRetrievalSystem()
        if hasattr(system, 'init_bm25'): system.init_bm25()
        if hasattr(system, 'init_dense'): system.init_dense()
        if hasattr(system, 'init_reranker'): system.init_reranker()
        return system

try:
    retriever_system = load_retrieval_system()
    st.sidebar.success("✅ Retriever Loaded")
except Exception as e:
    st.error(f"Failed to load retriever: {e}")
    st.stop()

# Initialize generator (pass API Key)
generator = RAGGenerator(api_key)

# ================= Chat Logic =================

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Mode switching reset logic: if switching away from Multi-Turn, notify user
if mode != "Feature A: Multi-Turn" and "last_mode" in st.session_state and st.session_state.last_mode == "Feature A: Multi-Turn":
    st.toast("Switched out of Multi-Turn mode. Conversation history context cleared.", icon="ℹ️")
st.session_state.last_mode = mode

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ================= User Input Processing =================
if prompt := st.chat_input("Ask a complex question (e.g., bridging entities)..."):
    # 1. Display user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        st.caption(f"🔧 **System Config:** {selected_retrieval} | 🎯 **Reranker:** {selected_reranker}")

        # --- Step 1: Query Processing ---
        final_query = prompt
        
        if mode == "Feature A: Multi-Turn":
            with st.status("🔄 Rewrite Query (Multi-Turn)...", expanded=True) as status:
                # Get recent 6 messages as history
                history_for_rewrite = st.session_state.messages[-6:-1] 
                final_query = generator.rewrite_query(prompt, history_for_rewrite)
                st.write(f"**Original:** {prompt}")
                st.write(f"**Rewritten:** {final_query}")
                status.update(label="Query Rewritten", state="complete", expanded=False)
        
        # --- Step 2: Retrieval ---
        with st.spinner(f"🔍 Retrieving documents for: '{final_query}'..."):
            try:
                # 1. BM25 Retrieve
                bm25_res = retriever_system.retrieve_bm25([final_query], k=top_k_retrieval)[0]
                # 2. Dense Retrieve
                dense_res = retriever_system.retrieve_dense([final_query], k=top_k_retrieval)[0]
                # 3. RRF Fusion
                fusion_ids = rrf_fusion(bm25_res, dense_res)
                # 4. Rerank (Top-K candidates -> Top-K context)
                rerank_candidates = fusion_ids[:50] # Number of candidates for Reranker
                rerank_results = retriever_system.rerank(final_query, rerank_candidates, k=top_k_gen)
                
                # Construct document list (backend expects List[Dict])
                retrieved_docs = []
                for rid, score in rerank_results:
                    # Get full text from retriever's id2text mapping
                    text = getattr(retriever_system, 'id2text', {}).get(rid, "Content not found.")
                    retrieved_docs.append({"id": rid, "text": text, "score": score})
                    
            except Exception as e:
                st.error(f"Retrieval Logic Error: {e}")
                retrieved_docs = []

        # --- Step 3: Generation ---
        final_answer = "Error generating answer."
        
        if mode == "Feature B: Agentic Workflow":
            # Agentic mode: show detailed logs
            with st.status("🧠 Agentic Thinking (Analyst & Editor)...", expanded=True) as status:
                final_answer, steps_log = generator.generate_agentic(final_query, retrieved_docs)
                
                # Render logs
                st.markdown(steps_log) 
                
                status.update(label="Reasoning & Verification Complete", state="complete", expanded=False)
                
        else: 
            # Basic (CoT) / Multi-Turn mode
            with st.spinner("🤖 Generating Answer (CoT Mode)..."):
                final_answer = generator.generate_basic(final_query, retrieved_docs)

        # --- Display final result ---
        response_placeholder.markdown(f"**Answer:** {final_answer}")
        
        # --- Display reference documents (Evidence) ---
        with st.expander("📚 Referenced Documents"):
            if not retrieved_docs:
                st.write("No documents retrieved.")
            for i, doc in enumerate(retrieved_docs[:8]):
                score_display = f"(Score: {doc['score']:.4f})" if 'score' in doc else ""
                st.markdown(f"**Doc {i+1} [{doc['id']}]** {score_display}")
                # Show first 300 characters for preview
                st.caption(doc['text'][:300] + ("..." if len(doc['text'])>300 else ""))
                st.divider()

        # --- Update history ---
        st.session_state.messages.append({"role": "assistant", "content": final_answer})