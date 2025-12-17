# 🤖 Agentic RAG + Weather Assistant

An intelligent assistant powered by **LangGraph** and **FastAPI** that intelligently routes queries to either a weather API or a document Q&A system using RAG (Retrieval Augmented Generation) with advanced reranking.

## ✨ Features

### 🌤️ Weather Agent
- Extracts city names from natural language queries using LLM
- Fetches real-time weather data from OpenWeather API
- Returns structured weather summaries (temperature, humidity, wind speed, conditions)

### 📚 PDF RAG Agent
- Ingests any PDF document of your choice
- Chunks and embeds documents using OpenAI embeddings (text-embedding-3-small)
- Stores vectors in Qdrant for semantic search
- **Two-stage retrieval**: Top-10 candidates → Cohere reranking → Top-3 results
- Generates accurate answers grounded in your documents with relevance scores

### 🔀 Intelligent Routing
- Automatically determines whether to use weather API or document search
- LLM-powered intent classification (GPT-4o-mini)
- Zero manual configuration needed

### 📊 Answer Evaluation
- Scores responses on relevance, accuracy, and completeness (1-10 scale)
- Displayed in real-time in the UI
- Robust error handling with fallback mechanisms

### ⚡ Async Architecture
- **FastAPI backend** with async endpoints for high performance
- **Streamlit frontend** with instant message display
- Thread pool executor for concurrent LangGraph operations
- Health checks cached on initial load (no redundant API calls)

### 🔍 Full Observability
- LangSmith integration for tracing all operations
- Track performance, costs, and debug issues
- Complete request/response logging

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI (async) |
| **Frontend** | Streamlit |
| **Orchestration** | LangChain + LangGraph |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI text-embedding-3-small (1536d) |
| **Vector DB** | Qdrant (local Docker or cloud) |
| **Reranking** | Cohere rerank-english-v3.0 |
| **Weather API** | OpenWeatherMap |
| **Observability** | LangSmith |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Testing** | Pytest + pytest-mock |

---

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or
.\.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_key
OPENWEATHER_API_KEY=your_openweather_key
LANGCHAIN_API_KEY=your_langsmith_key
COHERE_API_KEY=your_cohere_key

# LangSmith Configuration (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag-weather
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 5. Start Qdrant (Local)

Run Qdrant via Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This exposes Qdrant at `http://localhost:6333` with no authentication required.

---

## 📖 Usage

### Starting the Application

**Terminal 1 - Start FastAPI Backend:**
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run app_frontend.py
```

The web interface will open at `http://localhost:8501` and allows you to:
- 🌤️ Ask weather questions: *"What's the weather in Mumbai?"*
- 📄 Ask document questions: *"Who was APJ Abdul Kalam?"*
- 📊 View reranking scores and evaluation metrics in real-time

### Adding Your Documents

1. Place your PDF file in the `documents/` folder
2. Update `PDF_PATH` in `config.py`:
   ```python
   PDF_PATH = "documents/your_document.pdf"
   ```
3. Restart the backend - it will automatically ingest and index your document on startup

---

## 🏗️ Architecture

### High-Level Flow

```
Streamlit Frontend (Port 8501)
         ↓ HTTP POST /query
FastAPI Backend (Port 8000)
         ↓
    LangGraph Workflow
         ↓
┌────────────────────────────────┐
│   Router Node (GPT-4o-mini)    │
│   Classify: weather or pdf     │
└────────────┬───────────────────┘
             ↓
    ┌────────┴────────┐
    ↓                 ↓
┌─────────────┐  ┌──────────────────┐
│ Weather Node│  │   RAG Node       │
│ - Extract   │  │ - Qdrant (top-10)│
│   city      │  │ - Cohere rerank  │
│ - Call API  │  │   (top-3)        │
│ - Format    │  │ - Return chunks  │
└──────┬──────┘  └────────┬─────────┘
       └─────────┬─────────┘
                 ↓
       ┌─────────────────┐
       │ Generation Node │
       │  GPT-4o-mini    │
       └────────┬────────┘
                ↓
       ┌─────────────────┐
       │ Evaluation Node │
       │  Score: 1-10    │
       └────────┬────────┘
                ↓
            LangSmith
            (Tracing)
```

### RAG Pipeline Details

1. **Initial Retrieval**: Qdrant semantic search returns top-10 chunks (RETRIEVAL_TOP_K=10)
2. **Reranking**: Cohere rerank-english-v3.0 reranks for relevance → top-3 (RERANK_TOP_N=3)
3. **Generation**: LLM uses reranked chunks as context with relevance scores
4. **Display**: Reranking scores (0.0-1.0) shown in expandable UI sections

### Async Backend Architecture

- **FastAPI** handles HTTP requests with async endpoints
- **ThreadPoolExecutor** runs synchronous LangGraph operations in background
- **Health checks** cached on initial load (no repeated calls)
- **Batch query endpoint** for concurrent processing (`/batch-query`)

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests (20 total)
pytest test_pipeline_v2.py -v

# RAG tests only (10 tests)
pytest test_pipeline_v2.py -k "rag" -v

# Weather tests only (10 tests)
pytest test_pipeline_v2.py -k "weather" -v

# With coverage report
pytest test_pipeline_v2.py --cov=nodes --cov-report=html -v
```

### Test Coverage Includes:
- ✅ 10 RAG retrieval tests (Kalam biography queries)
- ✅ 10 Weather API tests (Indian cities)
- ✅ Router logic and intent classification
- ✅ Error handling and edge cases
- ✅ Reranking score validation
- ✅ Mock-based testing (no real API calls)

---

## 📁 Project Structure

```
├── main.py                     # FastAPI async backend
├── app_frontend.py             # Streamlit UI (optimized)
├── config.py                   # Configuration and environment variables
├── nodes.py                    # LangGraph node implementations
├── graph.py                    # LangGraph workflow definition
├── utils.py                    # Vector store utilities (Qdrant)
├── test_pipeline_v2.py         # Unit tests (20 tests)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── documents/                  # Place your PDFs here
    └── apj-abdul-kalam-biography.pdf
```

---

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Retrieval Configuration
RETRIEVAL_TOP_K = 10      # Initial candidates from Qdrant
RERANK_TOP_N = 3          # Final reranked results from Cohere

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "pdf_documents"

# OpenWeather API
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# PDF Path
PDF_PATH = "documents/apj-abdul-kalam-biography.pdf"
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Backend health check |
| `/query` | POST | Process single query |
| `/batch-query` | POST | Process multiple queries concurrently |
| `/collection` | GET | Get Qdrant collection info |
| `/collection` | DELETE | Reset collection |

---

## 🔍 LangSmith Tracing

All operations are logged to LangSmith for debugging and evaluation:

1. Sign up at [smith.langchain.com](https://smith.langchain.com/)
2. Add your `LANGCHAIN_API_KEY` to `.env`
3. Set `LANGCHAIN_TRACING_V2=true`
4. View traces at: `https://smith.langchain.com/`

**What you can see:**
- 📊 Full execution traces for each query
- 💰 Token usage and costs per operation
- ⏱️ Latency per node (router → retrieval → generation → evaluation)
- 🔄 Input/output at each step
- 📈 Evaluation scores over time

---

## 🎯 Extending

Ideas to customize and enhance:

### Document Sources
- **Multiple PDFs**: Index multiple documents in same collection
- **Web scraping**: Add URLs as retrieval sources
- **Multiple collections**: Separate collections per domain (legal, medical, etc.)

### Additional Agents
- **News agent**: Fetch latest news via NewsAPI
- **Code search agent**: Search GitHub/Stack Overflow
- **Finance agent**: Stock prices, crypto data
- **Database agent**: SQL query generation

### Deployment
- **Cloud Qdrant**: Use Qdrant Cloud for production
- **Deploy backend**: Railway, Render, or AWS Lambda
- **Deploy frontend**: Streamlit Cloud (free tier available)
- **Containerization**: Docker Compose for both services

### Performance
- **Caching**: Redis for frequently asked questions
- **Hybrid search**: Combine semantic + BM25 keyword search
- **Streaming responses**: Stream LLM output token-by-token
- **Custom embeddings**: Try Cohere/Voyage embeddings

---

## 🐛 Troubleshooting

### Qdrant not connecting?
```bash
docker ps  # Check if Qdrant container is running
docker logs <container-id>  # Check logs
curl http://localhost:6333  # Test connectivity
```

### FastAPI backend not starting?
```bash
# Check if port 8000 is already in use
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Try a different port
uvicorn main:app --reload --port 8001
```

### Streamlit frontend errors?
```bash
# Clear cache
streamlit cache clear

# Check if backend is running
curl http://localhost:8000/health
```

### Import errors?
```bash
pip install --upgrade -r requirements.txt
```

### Empty search results?
- Verify PDF was ingested (check sidebar "Vector Count")
- Reset collection via UI sidebar button
- Check PDF path in `config.py`

### Slow responses?
- Reduce `RETRIEVAL_TOP_K` to 5 for faster retrieval
- Use a smaller embedding model
- Enable caching for repeated queries

---

## 📊 Performance Metrics

Typical response times (on standard hardware):

| Query Type | Retrieval | Reranking | Generation | Total |
|------------|-----------|-----------|------------|-------|
| Weather | N/A | N/A | 0.8s | ~1s |
| PDF (RAG) | 0.3s | 0.2s | 1.2s | ~1.7s |

*Times may vary based on document size, network latency, and API rate limits*

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - feel free to use this project for your own applications.

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Qdrant](https://qdrant.tech/)
- [Cohere](https://cohere.com/)
- [OpenAI](https://openai.com/)
- [OpenWeather](https://openweathermap.org/)

---

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the troubleshooting section above

---

**Happy Building! 🚀**
