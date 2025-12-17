import streamlit as st
import httpx
import asyncio
from typing import Dict, Any

# FastAPI backend URL
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Agentic RAG + Weather Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG + Weather Assistant")
st.markdown("Frontend powered by **Streamlit** | Backend powered by **FastAPI (Async)**")

# ============================================================================
# ASYNC HELPER FUNCTIONS
# ============================================================================

async def check_backend_health_async() -> Dict[str, Any]:
    """Check if FastAPI backend is healthy (async)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "error": "Backend returned non-200 status"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def query_backend_async(question: str) -> Dict[str, Any]:
    """Send query to FastAPI backend (async)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "detail": response.text
                }
    except Exception as e:
        return {"error": str(e)}

async def get_collection_info_async() -> Dict[str, Any]:
    """Get Qdrant collection info from backend (async)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/collection", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to fetch collection info"}
    except Exception as e:
        return {"error": str(e)}

async def reset_collection_async() -> Dict[str, Any]:
    """Reset Qdrant collection via backend (async)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{API_URL}/collection", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to reset collection"}
    except Exception as e:
        return {"error": str(e)}

# Sync wrappers for Streamlit
def check_backend_health() -> Dict[str, Any]:
    return asyncio.run(check_backend_health_async())

def query_backend(question: str) -> Dict[str, Any]:
    return asyncio.run(query_backend_async(question))

def get_collection_info() -> Dict[str, Any]:
    return asyncio.run(get_collection_info_async())

def reset_collection() -> Dict[str, Any]:
    return asyncio.run(reset_collection_async())

# ============================================================================
# SESSION STATE INITIALIZATION - ONE TIME ONLY
# ============================================================================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check health ONLY ONCE on first load
if "health_status" not in st.session_state:
    st.session_state.health_status = check_backend_health()

# Check collection ONLY ONCE on first load
if "collection_info" not in st.session_state:
    st.session_state.collection_info = get_collection_info()

# ============================================================================
# SIDEBAR - Backend Status & Controls
# ============================================================================

with st.sidebar:
    st.header("⚙️ Backend Status")
    
    # Use cached health status (only checked once on load)
    health = st.session_state.health_status
    
    if health.get("status") == "healthy":
        st.success("✅ FastAPI Backend: Online (Async)")
        
        # Use cached collection info
        collection_info = st.session_state.collection_info
        if "error" not in collection_info:
            st.metric("Vector Count", collection_info.get('vector_count', 0))
    else:
        st.error("❌ FastAPI Backend: Offline")
        st.warning(f"Error: {health.get('error', 'Unknown error')}")
        st.info("Start backend: `uvicorn main:app --reload --port 8000`")
    
    st.markdown("---")
    
    # Controls
    st.header("🎛️ Controls")
    
    if st.button("🔄 Refresh Status"):
        # Only refresh when button is clicked
        st.session_state.health_status = check_backend_health()
        st.session_state.collection_info = get_collection_info()
        st.rerun()
    
    if st.button("🗑️ Reset Collection"):
        with st.spinner("Resetting collection..."):
            result = reset_collection()
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(result.get('message', 'Collection reset successfully'))
                st.info(result.get('note', ''))
                # Refresh collection info after reset
                st.session_state.collection_info = get_collection_info()
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
**Features:**
- ⚡ Async processing
- 🌤️ Real-time weather
- 📄 PDF Q&A
- 🤖 Smart routing
- 🎯 Cohere reranking
- 📊 LangSmith tracing
""")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Display ALL previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display metadata if available for assistant messages
        if "metadata" in message and message["role"] == "assistant":
            with st.expander("📊 Details"):
                metadata = message["metadata"]
                
                # Show route
                route = metadata.get("route", "unknown")
                st.write(f"**Route:** `{route}`")
                
                # Show rerank scores if available (PDF queries)
                if route == "pdf" and "rerank_scores" in metadata and metadata["rerank_scores"]:
                    st.subheader("🎯 Reranking Scores")
                    cols = st.columns(len(metadata["rerank_scores"]))
                    for i, (col, score) in enumerate(zip(cols, metadata["rerank_scores"]), 1):
                        col.metric(f"Chunk {i}", f"{score:.4f}")
                    st.markdown("---")
                
                # Show evaluation
                if "evaluation" in metadata and metadata["evaluation"]:
                    st.subheader("📈 Evaluation")
                    eval_data = metadata["evaluation"]
                    if "relevance" in eval_data:
                        eval_cols = st.columns(3)
                        eval_cols[0].metric("Relevance", f"{eval_data.get('relevance', 'N/A')}/10")
                        eval_cols[1].metric("Accuracy", f"{eval_data.get('accuracy', 'N/A')}/10")
                        eval_cols[2].metric("Completeness", f"{eval_data.get('completeness', 'N/A')}/10")
                    st.markdown("---")
                
                # Show full metadata
                st.json(metadata)

# Chat input - Handle new messages
if prompt := st.chat_input("Ask about weather or documents...", disabled=(health.get("status") != "healthy")):
    
    # Step 1: Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Step 2: Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Step 3: Show thinking indicator and get response
    with st.chat_message("assistant"):
        # Create a placeholder for the thinking indicator
        thinking_placeholder = st.empty()
        
        # Show thinking spinner
        with thinking_placeholder:
            with st.spinner("🤔 Thinking..."):
                response_data = query_backend(prompt)
        
        # Clear the thinking indicator
        thinking_placeholder.empty()
        
        # Now show the response directly (no "Complete!" message)
        if "error" in response_data:
            error_msg = f"❌ Error: {response_data['error']}"
            st.error(error_msg)
            
            # Add error to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
        else:
            # Display answer
            answer = response_data.get("answer", "No response generated")
            st.markdown(answer)
            
            # Prepare metadata
            metadata = {
                "route": response_data.get("route", "unknown"),
                "evaluation": response_data.get("evaluation", {}),
                "api_endpoint": f"{API_URL}/query",
                "async": True
            }
            
            # Add route-specific metadata
            if response_data.get("route") == "weather":
                metadata["weather_data"] = response_data.get("weather_data")
            elif response_data.get("route") == "pdf":
                metadata["retrieved_chunks"] = len(response_data.get("retrieved_docs", []))
                metadata["rerank_scores"] = response_data.get("rerank_scores", [])
            
            # Show rerank scores immediately for PDF queries
            if response_data.get("route") == "pdf" and response_data.get("rerank_scores"):
                with st.expander("🎯 Reranking Scores", expanded=True):
                    cols = st.columns(len(response_data["rerank_scores"]))
                    for i, (col, score) in enumerate(zip(cols, response_data["rerank_scores"]), 1):
                        col.metric(f"Chunk {i}", f"{score:.4f}")
            
            # Show evaluation
            if response_data.get("evaluation"):
                eval_data = response_data["evaluation"]
                if "relevance" in eval_data:
                    with st.expander("📈 Evaluation Scores"):
                        eval_cols = st.columns(3)
                        eval_cols[0].metric("Relevance", f"{eval_data.get('relevance', 'N/A')}/10")
                        eval_cols[1].metric("Accuracy", f"{eval_data.get('accuracy', 'N/A')}/10")
                        eval_cols[2].metric("Completeness", f"{eval_data.get('completeness', 'N/A')}/10")
            
            # Full details expander
            with st.expander("📊 Full Details"):
                st.json(response_data)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata
            })
