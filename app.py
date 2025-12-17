import streamlit as st
from graph import graph_app
from utils import create_vector_store, get_vector_store
from config import PDF_PATH, QDRANT_URL, QDRANT_COLLECTION
from qdrant_client import QdrantClient
import os


# Page configuration
st.set_page_config(
    page_title="Agentic RAG + Weather Assistant (Qdrant + Reranker)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG + Weather Assistant")
st.markdown("Powered by **Qdrant + Cohere Reranker** | Ask questions about weather or your PDF documents!")


# Initialize vector store
@st.cache_resource
def initialize_vectorstore():
    """Initialize vector store on startup."""
    if os.path.exists(PDF_PATH):
        try:
            with st.spinner("📚 Initializing Qdrant vector database..."):
                vectorstore = create_vector_store(PDF_PATH)
                
                # Get collection info
                client = QdrantClient(url=QDRANT_URL)
                collection_info = client.get_collection(QDRANT_COLLECTION)
                doc_count = collection_info.points_count
                
                st.success(f"✅ Qdrant ready with {doc_count} document chunks")
                return True, doc_count
        except Exception as e:
            st.error(f"Error initializing Qdrant: {e}")
            st.info("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
            return False, 0
    else:
        st.error(f"PDF not found at {PDF_PATH}")
        return False, 0


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    ready, count = initialize_vectorstore()
    st.session_state.vectorstore_ready = ready
    st.session_state.doc_count = count


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display metadata if available
        if "metadata" in message:
            with st.expander("📊 Details"):
                metadata = message["metadata"]
                
                # Show rerank scores if available
                if "rerank_scores" in metadata and metadata["rerank_scores"]:
                    st.subheader("🎯 Reranking Scores")
                    for i, score in enumerate(metadata["rerank_scores"], 1):
                        st.metric(f"Chunk {i}", f"{score:.4f}")
                    st.markdown("---")
                
                st.json(metadata)


# Chat input
if prompt := st.chat_input("Ask about weather or your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Invoke graph
                result = graph_app.invoke({
                    "question": prompt,
                    "route": "",
                    "context": "",
                    "weather_data": {},
                    "retrieved_docs": [],
                    "rerank_scores": [],
                    "generation": "",
                    "evaluation": {}
                })
                
                response = result.get("generation", "I couldn't generate a response.")
                route = result.get("route", "unknown")
                evaluation = result.get("evaluation", {})
                rerank_scores = result.get("rerank_scores", [])
                
                st.markdown(response)
                
                # Show metadata
                metadata = {
                    "route": route,
                    "evaluation": evaluation,
                    "database": "Qdrant + Cohere Reranker"
                }
                
                if route == "weather" and result.get("weather_data"):
                    metadata["weather_data"] = result["weather_data"]
                elif route == "pdf":
                    if result.get("retrieved_docs"):
                        metadata["retrieved_chunks"] = len(result["retrieved_docs"])
                    if rerank_scores:
                        metadata["rerank_scores"] = rerank_scores
                
                # Show rerank scores prominently
                if rerank_scores:
                    with st.expander("🎯 Reranking Scores", expanded=True):
                        cols = st.columns(len(rerank_scores))
                        for i, (col, score) in enumerate(zip(cols, rerank_scores), 1):
                            col.metric(f"Chunk {i}", f"{score:.4f}")
                
                with st.expander("📊 Full Details"):
                    st.json(metadata)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": metadata
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info(f"""
    **Status:**
    - Vector DB: {'✅ Qdrant Ready' if st.session_state.vectorstore_ready else '❌ Not Ready'}
    - Documents: {st.session_state.get('doc_count', 0)} chunks
    - PDF Path: `{PDF_PATH}`
    - Qdrant URL: `{QDRANT_URL}`
    - Reranker: Cohere rerank-english-v3.0
    """)
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🗑️ Reset Qdrant Collection"):
        try:
            client = QdrantClient(url=QDRANT_URL)
            client.delete_collection(QDRANT_COLLECTION)
            st.success("Qdrant collection deleted successfully!")
            st.session_state.vectorstore_ready = False
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting collection: {e}")
    
    st.markdown("---")
    st.markdown("""
    **Features:**
    - 🌤️ Real-time weather data
    - 📄 PDF document Q&A
    - 🤖 Intelligent routing
    - 🎯 Cohere reranking
    - 📊 LangSmith evaluation
    - 🚀 Qdrant vector search
    """)
    
    st.markdown("---")
    st.markdown("""
    **Setup:**
    ```
    # Qdrant
    docker run -p 6333:6333 qdrant/qdrant
    
    # Get Cohere API key
    https://dashboard.cohere.com/
    ```
    """)
