# Agentic RAG + Weather Assistant

An intelligent assistant powered by LangGraph that routes queries to either a weather API or a document Q&A system using RAG (Retrieval Augmented Generation).

## Features

- **Weather Agent**  
  - Extracts city names from natural language queries
  - Fetches real-time weather data from OpenWeather API
  - Returns structured weather summaries (temperature, humidity, wind speed, conditions)

- **PDF RAG Agent**  
  - Ingests any PDF document of your choice
  - Chunks and embeds documents using OpenAI embeddings
  - Stores vectors in Qdrant for semantic search
  - Retrieves top-k candidates and reranks with Cohere
  - Generates accurate answers grounded in your documents

- **Intelligent Routing**  
  - Automatically determines whether to use weather API or document search
  - LLM-powered intent classification

- **Answer Evaluation**  
  - Scores responses on relevance, accuracy, and completeness
  - Robust error handling with fallback mechanisms

- **Full Observability**  
  - LangSmith integration for tracing all operations
  - Track performance, costs, and debug issues

---

## Tech Stack

- **LangChain / LangGraph** - Orchestration framework
- **OpenAI** - LLM and embeddings (gpt-4o-mini, text-embedding-3-small)
- **Qdrant** - Vector database (local Docker or cloud)
- **Cohere** - Reranking model (rerank-english-v3.0)
- **OpenWeather API** - Real-time weather data
- **LangSmith** - Tracing and evaluation
- **Streamlit** - Web UI
- **PyMuPDF** - PDF parsing
- **Pytest** - Unit testing

---

## Setup

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
.\.venv\Scriptsctivate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
OPENWEATHER_API_KEY=your_openweather_key
LANGCHAIN_API_KEY=your_langsmith_key
COHERE_API_KEY=your_cohere_key
```

### 5. Start Qdrant (Local)

Run Qdrant via Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This exposes Qdrant at `http://localhost:6333` with no authentication required.

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The web interface allows you to:
- Ask weather questions: *"What's the weather in Mumbai?"*
- Ask document questions: *"What are the key points in chapter 3?"*

### Adding Your Documents

1. Place your PDF file in the `documents/` folder
2. Update `PDF_PATH` in `config.py`:
   ```python
   PDF_PATH = "documents/your_document.pdf"
   ```
3. Restart the app - it will automatically ingest and index your document

---

## Architecture

```
User Query
    ↓
Router Node (LLM decides: weather or pdf)
    ↓
┌─────────────────┬─────────────────┐
│  Weather Node   │   RAG Node      │
│  - Extract city │   - Qdrant      │
│  - Call API     │   - Cohere      │
│  - Format data  │   - Rerank      │
└─────────────────┴─────────────────┘
    ↓
Generation Node (LLM generates answer)
    ↓
Evaluation Node (scores quality)
    ↓
LangSmith (logs trace)
```

### RAG Pipeline Details

1. **Initial Retrieval**: Qdrant returns top 10 semantically similar chunks
2. **Reranking**: Cohere reranks candidates for relevance (top 4 selected)
3. **Generation**: LLM uses reranked chunks as context
4. **Scoring**: Reranking scores displayed in UI

---

## Testing

Run the full test suite:

```bash
# All tests
pytest test_pipeline.py -v

# RAG tests only
pytest test_pipeline_v2.py -k "rag" -v

# Weather tests only
pytest test_pipeline_v2.py -k "weather" -v

# With coverage
pytest test_pipeline.py --cov=nodes --cov-report=html -v
```

Test coverage includes:
- 10 RAG retrieval tests
- 10 Weather API tests
- Router logic tests
- Error handling scenarios

---

## Project Structure

```
├── app.py                 # Streamlit UI
├── config.py              # Configuration and environment variables
├── nodes.py               # LangGraph node implementations
├── graph.py               # LangGraph workflow definition
├── utils.py               # Vector store utilities
├── test_pipeline.py       # Unit tests
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
└── documents/             # Place your PDFs here
    └── sample.pdf
```

---

## Configuration

Key settings in `config.py`:

```python
# Retrieval Configuration
RETRIEVAL_TOP_K = 10      # Initial candidates from Qdrant
RERANK_TOP_N = 3          # Final reranked results

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "pdf_documents"

# PDF Path
PDF_PATH = "documents/sample.pdf"
```

---

## LangSmith Tracing

All operations are logged to LangSmith for debugging and evaluation:

1. Sign up at [smith.langchain.com](https://smith.langchain.com/)
2. Add your `LANGCHAIN_API_KEY` to `.env`
3. View traces at: `https://smith.langchain.com/`

You can see:
- Full execution traces
- Token usage and costs
- Latency per node
- Input/output at each step
- Evaluation scores

---

## Extending

Some ideas to customize:

- **Add more document types**: Swap PDFs for any text source
- **Multiple collections**: Index different document sets
- **Additional tools**: Add news, finance, or code search agents
- **Cloud deployment**: Move Qdrant to cloud, deploy on Railway/Render
- **Custom embeddings**: Try different embedding models
- **Hybrid search**: Combine semantic + keyword search

---

## Troubleshooting

**Qdrant not connecting?**
```bash
docker ps  # Check if Qdrant container is running
docker logs <container-id>  # Check logs
```

**Import errors?**
```bash
pip install --upgrade -r requirements.txt
```

**Empty search results?**
- Verify your PDF was ingested (check Streamlit sidebar for chunk count)
- Reset collection via UI sidebar button

---

## License

MIT License - feel free to use this project for your own applications.

---

## Acknowledgments

Built with:
- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Qdrant](https://qdrant.tech/)
- [Cohere](https://cohere.com/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
