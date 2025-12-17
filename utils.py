import fitz  # PyMuPDF
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import QDRANT_URL, QDRANT_COLLECTION, OPENAI_API_KEY


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def create_vector_store(pdf_path: str):
    """Create and populate Qdrant vector store with PDF embeddings."""
    # Extract and split text
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create Document objects
    from langchain_core.documents import Document
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if collection exists
    try:
        collection_info = client.get_collection(QDRANT_COLLECTION)
        existing_count = collection_info.points_count
        
        if existing_count > 0:
            print(f"✅ Using existing Qdrant collection with {existing_count} documents")
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings
            )
            return vectorstore
        else:
            print("📝 Adding documents to empty collection")
    except Exception as e:
        print(f"📝 Creating new collection: {e}")
        # Create collection if not exists
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    
    # Create vector store from documents
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        force_recreate=False
    )
    
    print(f"✅ Created Qdrant vector store with {len(documents)} documents")
    
    return vectorstore


def get_vector_store():
    """Get existing Qdrant vector store."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    client = QdrantClient(url=QDRANT_URL)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings
    )
    
    return vectorstore


def add_documents_to_store(pdf_path: str):
    """Add new documents to existing Qdrant store."""
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    from langchain_core.documents import Document
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    vectorstore = get_vector_store()
    vectorstore.add_documents(documents)
    
    print(f"✅ Added {len(documents)} new documents to Qdrant store")
    
    return vectorstore
