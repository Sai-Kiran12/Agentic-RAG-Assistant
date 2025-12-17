from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict  # ✅ Add ConfigDict
from typing import Optional, List, Dict
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
from graph import graph_app
from utils import create_vector_store, get_vector_store
from config import PDF_PATH, QDRANT_URL, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
import os

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG + Weather Assistant API",
    description="Async API for weather queries and document Q&A using RAG with Qdrant and Cohere reranking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    question: str
    
    model_config = ConfigDict(  # ✅ New way
        json_schema_extra={
            "example": {
                "question": "What's the weather in Mumbai?"
            }
        }
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    context: str
    weather_data: Optional[Dict] = None
    retrieved_docs: Optional[List[str]] = None
    rerank_scores: Optional[List[float]] = None
    evaluation: Optional[Dict] = None
    
    model_config = ConfigDict(  # ✅ New way
        json_schema_extra={
            "example": {
                "question": "What's the weather in Mumbai?",
                "answer": "The weather in Mumbai is 32°C with 70% humidity...",
                "route": "weather",
                "context": "Temperature: 32°C...",
                "weather_data": {"temp": 32, "humidity": 70},
                "evaluation": {"relevance": 9, "accuracy": 9, "completeness": 8}
            }
        }
    )

class HealthResponse(BaseModel):
    status: str
    qdrant_status: str
    qdrant_url: str
    collection_name: str
    document_count: int
    pdf_path: str


class CollectionInfo(BaseModel):
    collection_name: str
    vector_count: int
    indexed: bool


# ============================================================================
# ASYNC HELPER FUNCTIONS
# ============================================================================

async def run_in_executor(func, *args):
    """Run blocking function in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


async def initialize_vector_store():
    """Initialize vector store asynchronously."""
    if os.path.exists(PDF_PATH):
        try:
            print(f"📚 Initializing Qdrant with PDF: {PDF_PATH}")
            # Run blocking operation in thread pool
            await run_in_executor(create_vector_store, PDF_PATH)
            
            # Use async client
            async_client = AsyncQdrantClient(url=QDRANT_URL)
            collection_info = await async_client.get_collection(QDRANT_COLLECTION)
            doc_count = collection_info.points_count
            await async_client.close()
            
            print(f"✅ Qdrant ready with {doc_count} document chunks")
            return True
        except Exception as e:
            print(f"❌ Error initializing Qdrant: {e}")
            return False
    else:
        print(f"⚠️ PDF not found at {PDF_PATH}")
        return False


async def get_qdrant_info():
    """Get Qdrant collection info asynchronously."""
    try:
        async_client = AsyncQdrantClient(url=QDRANT_URL)
        collection_info = await async_client.get_collection(QDRANT_COLLECTION)
        doc_count = collection_info.points_count
        await async_client.close()
        return doc_count, "healthy"
    except Exception as e:
        return 0, f"unhealthy: {str(e)}"


async def process_query_async(question: str):
    """Process query asynchronously in thread pool."""
    # LangGraph is synchronous, so run in thread pool
    def invoke_graph():
        return graph_app.invoke({
            "question": question,
            "route": "",
            "context": "",
            "weather_data": {},
            "retrieved_docs": [],
            "rerank_scores": [],
            "generation": "",
            "evaluation": {}
        })
    
    return await run_in_executor(invoke_graph)


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup."""
    print("🚀 Starting Async Agentic RAG + Weather API...")
    await initialize_vector_store()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down API...")
    executor.shutdown(wait=True)


# ============================================================================
# API ENDPOINTS (ALL ASYNC)
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic RAG + Weather Assistant API (Async)",
        "version": "1.0.0",
        "async": True,
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "collection": "/collection",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and Qdrant health status asynchronously."""
    doc_count, qdrant_status = await get_qdrant_info()
    
    return HealthResponse(
        status="healthy",
        qdrant_status=qdrant_status,
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        document_count=doc_count,
        pdf_path=PDF_PATH
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Process a query asynchronously - routes to either weather API or document RAG.
    
    - **question**: The user's question (weather or document-related)
    
    Returns:
    - **answer**: Generated response
    - **route**: Which route was taken (weather or pdf)
    - **context**: Retrieved context used for generation
    - **weather_data**: Weather API data (if weather route)
    - **retrieved_docs**: Retrieved document chunks (if pdf route)
    - **rerank_scores**: Cohere reranking scores (if pdf route)
    - **evaluation**: Quality evaluation scores
    """
    try:
        # Process query asynchronously
        result = await process_query_async(request.question)
        
        return QueryResponse(
            question=request.question,
            answer=result.get("generation", "No response generated"),
            route=result.get("route", "unknown"),
            context=result.get("context", ""),
            weather_data=result.get("weather_data") if result.get("route") == "weather" else None,
            retrieved_docs=result.get("retrieved_docs") if result.get("route") == "pdf" else None,
            rerank_scores=result.get("rerank_scores") if result.get("route") == "pdf" else None,
            evaluation=result.get("evaluation", {})
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/collection", response_model=CollectionInfo, tags=["Collection"])
async def get_collection_info():
    """Get information about the Qdrant collection asynchronously."""
    try:
        async_client = AsyncQdrantClient(url=QDRANT_URL)
        collection_info = await async_client.get_collection(QDRANT_COLLECTION)
        await async_client.close()
        
        return CollectionInfo(
            collection_name=QDRANT_COLLECTION,
            vector_count=collection_info.points_count,
            indexed=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching collection info: {str(e)}")


@app.delete("/collection", tags=["Collection"])
async def reset_collection():
    """Delete and reset the Qdrant collection asynchronously."""
    try:
        async_client = AsyncQdrantClient(url=QDRANT_URL)
        await async_client.delete_collection(QDRANT_COLLECTION)
        await async_client.close()
        
        return {
            "message": f"Collection '{QDRANT_COLLECTION}' deleted successfully",
            "note": "Restart the API to re-initialize with the PDF"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting collection: {str(e)}")


# ============================================================================
# BATCH QUERY ENDPOINT (ASYNC ADVANTAGE)
# ============================================================================

class BatchQueryRequest(BaseModel):
    questions: List[str]


class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    total_time: float


@app.post("/batch-query", response_model=BatchQueryResponse, tags=["Query"])
async def batch_query(request: BatchQueryRequest):
    """
    Process multiple queries concurrently (async advantage).
    
    - **questions**: List of questions to process
    
    Returns list of answers processed in parallel.
    """
    import time
    start_time = time.time()
    
    # Process all queries concurrently
    tasks = [process_query_async(question) for question in request.questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    responses = []
    for question, result in zip(request.questions, results):
        if isinstance(result, Exception):
            responses.append(QueryResponse(
                question=question,
                answer=f"Error: {str(result)}",
                route="error",
                context=""
            ))
        else:
            responses.append(QueryResponse(
                question=question,
                answer=result.get("generation", ""),
                route=result.get("route", "unknown"),
                context=result.get("context", ""),
                weather_data=result.get("weather_data"),
                retrieved_docs=result.get("retrieved_docs"),
                rerank_scores=result.get("rerank_scores"),
                evaluation=result.get("evaluation", {})
            ))
    
    total_time = time.time() - start_time
    
    return BatchQueryResponse(
        results=responses,
        total_time=total_time
    )


# ============================================================================
# RUN SERVER WITH ASYNC SUPPORT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1  # Use 1 worker for development, increase for production
    )
