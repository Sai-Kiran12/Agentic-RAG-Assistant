import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agentic-rag-weather"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "pdf_documents"

# OpenWeather API
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# PDF Path
PDF_PATH = "documents/apj-abdul-kalam-biography.pdf" # Path to PDF document

# Retrieval Configuration
RETRIEVAL_TOP_K = 10  # Initial retrieval count
RERANK_TOP_N = 3      # Final reranked results
