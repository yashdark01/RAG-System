# Advanced RAG System & Multi-Agent Architecture

A production-grade, asynchronous Retrieval-Augmented Generation (RAG) and autonomous multi-agent platform built with **FastAPI** and **LangChain**.

---

## 🌟 Key Architectural Features

- **Native LangChain Ecosystem**: Built entirely with modern LangChain Expression Language (LCEL), `create_tool_calling_agent`, and `AgentExecutor`.
- **Multi-Model Support with Fallbacks**:
  - **Google Gemini** (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`) via `langchain-google-genai`
  - **Groq** (`llama3-70b-8192`, `mixtral-8x7b-32768`) via `langchain-groq`
  - **OpenAI** (`gpt-4o`, `gpt-4o-mini`) via `langchain-openai`
  - **Anthropic** (`claude-3-5-sonnet`) via `langchain-anthropic`
  - Automatic cross-provider fallback chains prevent outages during rate limits.
  - Offline `FakeListChatModel` fallback for local development without credentials.
- **Hybrid Retrieval & Reranking**:
  - **Dense Semantic Retrieval**: High-precision vector similarity via `FAISS` and `sentence-transformers`.
  - **Sparse Keyword Retrieval**: BM25 keyword matching via `rank_bm25` (handles exact IDs, acronyms, and names).
  - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse candidates via LangChain `EnsembleRetriever`.
  - **Hierarchical Parent-Child Chunking**: Matches small child chunks for vector precision, while feeding larger parent context blocks to the LLM.
- **Agentic Web Search & Verification**:
  - Integrated **Tavily Search** (`TavilySearchResults`) for real-time web verification and research.
  - Dedicated claim verification tools to combat hallucinations.
- **Real-Time Streaming**:
  - Server-Sent Events (SSE) streaming for RAG generation (`/query/stream`) and agent tokens (`/execute/stream`).
- **Session Memory**:
  - Multi-turn conversation history powered by LangChain `ChatMessageHistory` and `RunnableWithMessageHistory`.

---

## 🏗️ Architecture Blueprint

```text
[ Offline Ingestion Pipeline ]
Uploaded Files (PDF, TXT, MD)
       │
       ▼
DocumentProcessor (RecursiveCharacterTextSplitter)
       ├─► Small Child Chunks (400 chars) ──► FAISS Vector Store + BM25 Tokenizer
       └─► Large Parent Chunks (2000 chars) ─► Persistent Document Store

[ Online Serving Pipeline ]
User Query / Session ID
       │
       ▼
Query Analysis & Expansion
       │
       ▼
EnsembleRetriever (Hybrid Search)
       ├─► FAISS Semantic Similarity Search (Weight: 60%)
       └─► BM25 Exact Keyword Match        (Weight: 40%)
       │
       ▼
Reciprocal Rank Fusion (RRF) & Context Assembly
       │
       ▼
Tool-Calling Agent / LCEL RAG Chain (Gemini / Groq / OpenAI)
       │
       ▼
Token-by-Token Streaming Output (SSE) + Citations
```

---

## 📁 Project Directory Structure

```text
RAG-System/
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git ignore rules for DBs, logs, and venvs
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Multi-service stack (App, PostgreSQL, Redis)
├── pyproject.toml                     # PEP 621 package and dependency definitions
├── requirements.txt                   # Complete dependencies list
├── setup.sh                           # Automated environment bootstrapping with uv
├── README.md                          # Production documentation
│
├── app/
│   ├── config.py                      # Pydantic Settings (Gemini, Tavily, Redis, DB)
│   ├── dependencies.py                # Centralized dependency injection container
│   ├── main.py                        # FastAPI application entry point & lifespan
│   │
│   ├── llm/
│   │   ├── provider.py                # Native LangChain BaseChatModel factory with fallbacks
│   │   └── prompt_templates.py        # LangChain ChatPromptTemplate definitions
│   │
│   ├── rag/
│   │   ├── document_processor.py      # RecursiveCharacterTextSplitter with parent-child chunks
│   │   ├── retriever.py               # BM25 + FAISS Hybrid Search with Reciprocal Rank Fusion
│   │   ├── vector_store.py            # FAISS vector store & LangChain Document bridge
│   │   └── embeddings.py              # HuggingFace & SentenceTransformer embeddings
│   │
│   ├── workflow/
│   │   ├── orchestrator.py            # create_tool_calling_agent & AgentExecutor with memory
│   │   ├── tools/
│   │   │   ├── search_tools.py        # Tavily Search tool integration
│   │   │   ├── rag_tools.py           # LangChain knowledge base retrieval @tool
│   │   │   └── verification_tools.py  # Fact checking & verification @tool
│   │   └── agents/                    # Multi-agent specialized agents
│   │
│   ├── services/
│   │   ├── rag_service.py             # RAG query execution & real-time token streaming
│   │   ├── agent_service.py           # Tool-calling agent execution & streaming
│   │   └── chat_service.py            # Multi-turn conversation management
│   │
│   └── api/v1/endpoints/
│       ├── health.py                  # Health and readiness probes
│       ├── documents.py               # Document upload, listing, and indexing
│       ├── rag.py                     # RAG query and /query/stream endpoints
│       ├── agents.py                  # Agent task execution and /execute/stream
│       └── chat.py                    # Multi-turn conversational chat
│
└── tests/
    ├── test_langchain_pipeline.py     # Tests for LangChain models, prompts, splitters, tools
    ├── test_agents.py                 # Unit tests for agent workflows
    ├── test_tools.py                  # Unit tests for search and verification tools
    └── test_integration.py            # End-to-end integration tests
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- `uv` (Recommended) or standard Python `venv` + `pip`
- Git

### 2. Setup with uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/RAG-System.git
cd RAG-System

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install all project and development dependencies
uv pip install -e ".[dev]"
```

*Alternatively, run the automated setup script:*
```bash
chmod +x setup.sh && ./setup.sh
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Key environment configurations:
```env
# Application
APP_NAME="RAG System"
APP_ENV=development
DEBUG=true

# LLM Configuration
LLM_PROVIDER=gemini              # Options: gemini, groq, openai, anthropic
LLM_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Web Search
TAVILY_API_KEY=your_tavily_api_key

# Vector Store & Embeddings
VECTOR_STORE_PATH=./data/vector_store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 4. Run the Application
With `uv`:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Or with activated virtual environment:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000/docs](http://localhost:8000/docs)** to view the interactive OpenAPI documentation.

---

## 🐳 Docker Deployment

To launch the full stack with PostgreSQL and Redis:
```bash
docker-compose up --build -d
```

Services:
- **FastAPI Backend**: `http://localhost:8000`
- **PostgreSQL Database**: `localhost:5432`
- **Redis Cache**: `localhost:6379`

To view logs:
```bash
docker-compose logs -f app
```

To stop:
```bash
docker-compose down
```

---

## 📡 API Reference & Examples

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Basic application health check |
| `GET` | `/api/v1/health/ready` | Deep readiness check (LLM, vector store, DB) |
| `POST` | `/api/v1/documents/upload` | Upload & hierarchical chunk index (PDF, TXT, MD) |
| `GET` | `/api/v1/documents/{id}` | Retrieve indexed document metadata |
| `POST` | `/api/v1/rag/query` | Standard RAG question answering with citations |
| `POST` | `/api/v1/rag/query/stream` | Server-Sent Events (SSE) token streaming |
| `POST` | `/api/v1/agents/execute` | Tool-calling autonomous agent execution |
| `POST` | `/api/v1/agents/execute/stream`| Real-time agent reasoning token streaming |
| `POST` | `/api/v1/chat` | Multi-turn conversational chat with session memory |

### Example 1: Ingest a Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample_report.pdf"
```

### Example 2: Standard RAG Query
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings of the report?",
    "top_k": 5
  }'
```

### Example 3: Stream RAG Query (Server-Sent Events)
```bash
curl -N -X POST "http://localhost:8000/api/v1/rag/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the conclusion in detail."
  }'
```

### Example 4: Autonomous Agent Execution (with Tavily Search)
```bash
curl -X POST "http://localhost:8000/api/v1/agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Research recent advancements in Agentic RAG and verify with web search.",
    "agent_type": "orchestrator"
  }'
```

---

## 🧪 Testing

Run unit and integration tests with `uv`:
```bash
uv run pytest tests/ -v
```
Or with activated venv:
```bash
pytest tests/ -v
```

---

## 📄 License
MIT License.
