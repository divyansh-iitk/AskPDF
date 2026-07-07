<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 📄 AskPDF — RAG-Powered Document Q&A

A production-grade, full-stack **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and have natural-language conversations with them. The system combines **hybrid search** (keyword + semantic), **neural reranking**, and **LLM-powered answer generation** — all orchestrated through a modular, scalable pipeline and deployable with a single command.

---

## 🎯 Features

| Feature | Description |
|---|---|
| **PDF Upload & Ingestion** | Upload PDFs via a polished chat UI; documents are automatically parsed, chunked, embedded, and indexed into a persistent vector store. |
| **Hybrid Retrieval** | Combines BM25 keyword search with dense vector similarity search using LangChain's `EnsembleRetriever` for high recall. |
| **Neural Reranking** | Retrieved candidates are reranked using Cohere's `rerank-v3.5` cross-encoder model for higher precision. |
| **Conversational Memory** | Multi-turn chat with LLM-based query rewriting — follow-up questions are reformulated using chat history context. |
| **Source Attribution** | Every answer shows the relevant document chunks and their relevance scores for transparency and trust. |
| **One-Command Deployment** | Fully Dockerized with Docker Compose (multi-container) + CI/CD pipeline for AWS EC2 via GitHub Actions. |

---

## 🏗️ System Architecture

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                     FRONTEND (Streamlit)                  │
                          │  • Chat UI with session state   • PDF upload sidebar      │
                          │  • Source attribution display    • Custom CSS styling      │
                          └─────────────────────────┬────────────────────────────────┘
                                                    │ HTTP (REST API)
                          ┌─────────────────────────▼────────────────────────────────┐
                          │                 BACKEND (FastAPI + Uvicorn)                │
                          │                                                           │
                          │  ┌─────────────┐    ┌──────────────────────────────────┐  │
                          │  │  /api/upload │    │          /api/query              │  │
                          │  └──────┬──────┘    └──────────────┬───────────────────┘  │
                          │         │                          │                      │
                          │         ▼                          ▼                      │
                          │  ┌─────────────┐    ┌──────────────────────────────────┐  │
                          │  │  Ingestion   │    │       Query Rewriting            │  │
                          │  │  Pipeline    │    │  (Llama-3.3-70B via Groq)        │  │
                          │  │             │    │  Reformulates follow-ups using    │  │
                          │  │  PDF Parse   │    │  chat history context             │  │
                          │  │  → Chunk     │    └──────────────┬───────────────────┘  │
                          │  │  → Embed     │                   │                      │
                          │  │  → Store     │                   ▼                      │
                          │  └──────┬──────┘    ┌──────────────────────────────────┐  │
                          │         │           │       Hybrid Retrieval            │  │
                          │         │           │  ┌────────────┐ ┌──────────────┐  │  │
                          │         │           │  │   BM25      │ │   Vector     │  │  │
                          │         ▼           │  │  (keyword)  │ │  (semantic)  │  │  │
                          │  ┌─────────────┐    │  └──────┬─────┘ └──────┬───────┘  │  │
                          │  │  ChromaDB    │◄───│        └──────┬───────┘          │  │
                          │  │ (Persistent) │    │       EnsembleRetriever          │  │
                          │  └─────────────┘    │       (weighted fusion)           │  │
                          │                     └──────────────┬───────────────────┘  │
                          │                                    │                      │
                          │                                    ▼                      │
                          │                     ┌──────────────────────────────────┐  │
                          │                     │       Cohere Reranking            │  │
                          │                     │    (rerank-v3.5 cross-encoder)    │  │
                          │                     │     Contextual Compression        │  │
                          │                     └──────────────┬───────────────────┘  │
                          │                                    │                      │
                          │                                    ▼                      │
                          │                     ┌──────────────────────────────────┐  │
                          │                     │       Answer Generation           │  │
                          │                     │    (Llama-3.3-70B via Groq)       │  │
                          │                     │    Grounded in retrieved context  │  │
                          │                     └──────────────────────────────────┘  │
                          └───────────────────────────────────────────────────────────┘
```

---

## 🧠 Technical Skills & Concepts Demonstrated

### Natural Language Processing & Information Retrieval

| Skill | How It's Applied |
|---|---|
| **Retrieval-Augmented Generation (RAG)** | End-to-end RAG pipeline: ingest → retrieve → rerank → generate. The LLM is grounded in retrieved document context rather than relying on parametric knowledge alone. |
| **Dense Vector Embeddings** | Documents are embedded using [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) (768-dim) via the Sentence Transformers library, enabling semantic similarity search. |
| **Sparse Retrieval (BM25)** | BM25 (Okapi BM25) keyword retrieval via `rank-bm25` provides lexically-grounded search that captures exact-match and term-frequency signals that dense embeddings can miss. |
| **Hybrid Search (Ensemble Retrieval)** | LangChain's `EnsembleRetriever` fuses BM25 + vector retrieval results with configurable weights (default 0.5/0.5), combining the strengths of sparse and dense methods via reciprocal rank fusion. |
| **Neural Reranking (Cross-Encoder)** | Cohere's `rerank-v3.5` cross-encoder reranks the fused candidate set, dramatically improving precision by jointly attending to query-document pairs. Implemented via LangChain's `ContextualCompressionRetriever`. |
| **Recursive Text Chunking** | Documents are split using `RecursiveCharacterTextSplitter` with hierarchical separators (`\n\n`, `\n`, `.`, `?`, `!`, ` `, `""`), preserving semantic boundaries while maintaining consistent chunk sizes. |
| **Query Rewriting** | An LLM-based query reformulation step rewrites follow-up questions into standalone queries using chat history, enabling coherent multi-turn conversations. |
| **Prompt Engineering** | Hand-crafted prompt templates with system/human message separation, explicit behavioral instructions, and context injection for both answer generation and query rewriting. |

### Large Language Models & AI Infrastructure

| Skill | How It's Applied |
|---|---|
| **LLM Integration (Groq API)** | Llama-3.3-70B-Versatile served via Groq's inference API for ultra-low latency generation. Configured with temperature=0.1 for deterministic, factual outputs. |
| **Sentence Transformers** | Local embedding model (`BAAI/bge-base-en-v1.5`) loaded via `SentenceTransformer` for offline, privacy-preserving embedding generation — no data leaves the server. |
| **Cohere Rerank API** | Integration with Cohere's reranking endpoint via `langchain-cohere` for production-grade neural reranking without managing a local cross-encoder model. |
| **LangChain Orchestration** | Modular use of LangChain's core abstractions: `BaseRetriever` (custom retriever subclass), `Document`, `PromptTemplate`, `EnsembleRetriever`, `ContextualCompressionRetriever`, and `CohereRerank`. |

### Backend Engineering

| Skill | How It's Applied |
|---|---|
| **FastAPI** | Async REST API with automatic OpenAPI/Swagger docs, Pydantic request validation, modular router registration, and a lifespan context manager for startup/shutdown resource management. |
| **Async Context Manager (Lifespan)** | Application lifecycle management via `@asynccontextmanager` — embedding models, vector store, and LLM client are initialized at startup and shared across requests via `app.state`. |
| **Pydantic Data Validation** | Request schemas defined with Pydantic `BaseModel` for automatic type validation, serialization, and API documentation generation. |
| **Modular API Design** | Routes separated into domain-specific routers (`upload.py`, `query.py`) with clean dependency injection through FastAPI's `Request` object. |
| **Dataclass Configuration** | All pipeline hyperparameters centralized in `config.py` using Python `@dataclass` — single source of truth for chunk sizes, model names, thresholds, and weights. |
| **Custom Logging** | Structured, timestamped file-based logging with configurable paths, using Python's `logging` module with a custom formatter. |
| **File I/O & Deduplication** | Server-side PDF storage with deduplication — files are only saved and re-embedded if not already present, preventing redundant vector store entries. |

### Vector Database & Data Layer

| Skill | How It's Applied |
|---|---|
| **ChromaDB (Persistent)** | Persistent vector store using ChromaDB's `PersistentClient` — embeddings survive container restarts via Docker named volumes. |
| **Cosine Similarity Filtering** | Retrieved documents are filtered by a configurable similarity threshold (default 0.5) after converting ChromaDB's cosine distance to similarity score (`1 - distance`). |
| **UUID-Based Chunk Tracking** | Each text chunk is assigned a UUID during ingestion, enabling ensemble retrieval deduplication via LangChain's `id_key` mechanism. |
| **Metadata Enrichment** | Documents are enriched with metadata (source filename, file type, content length, chunk index) at both the loading and indexing stages for downstream traceability. |

### Frontend Engineering

| Skill | How It's Applied |
|---|---|
| **Streamlit** | Interactive chat UI with `st.chat_input`, session state management for conversation persistence, sidebar file uploader, and expandable source attribution panels. |
| **Custom CSS Injection** | Styled chat bubbles (user vs. bot), dark theme, and responsive layout via injected `<style>` blocks for a polished user experience. |
| **Session State Management** | Streamlit's `st.session_state` manages conversation history across re-runs, enabling a seamless multi-turn chat experience. |
| **Environment-Based Configuration** | Frontend dynamically reads `BACKEND_URL` from environment variables, supporting both local development and containerized deployment without code changes. |

### DevOps, CI/CD & Cloud Deployment

| Skill | How It's Applied |
|---|---|
| **Docker** | Separate Dockerfiles for backend and frontend with `python:3.12-slim` base images, optimized layer caching, and minimal image sizes. |
| **Docker Compose** | Multi-service orchestration with health checks, named volumes for data persistence, inter-service networking, and environment variable injection. |
| **GitHub Actions CI/CD** | Automated pipeline: build Docker images → push to Amazon ECR → deploy to EC2 via self-hosted runner. Triggered on push to `cicd` branch. |
| **Amazon ECR** | Container images stored in Amazon Elastic Container Registry with separate repositories for backend and frontend services. |
| **Amazon EC2** | Production deployment on EC2 with a self-hosted GitHub Actions runner for continuous deployment, Docker networking, and secure secret management. |
| **Health Checks** | Docker Compose health check verifies backend readiness before starting the frontend, ensuring proper startup ordering. |
| **Secrets Management** | API keys and cloud credentials managed via GitHub Secrets (CI/CD) and `.env` files (local development) — never committed to source. |

### Software Engineering Practices

| Skill | How It's Applied |
|---|---|
| **Modular Architecture** | Clean separation of concerns: `rag/` (core pipeline), `app/` (API layer), `utils/` (configuration), `logger/` (observability), `frontend/` (presentation). |
| **Custom BaseRetriever** | `RAGRetriver` subclasses LangChain's `BaseRetriever` with a custom `_get_relevant_documents` implementation for ChromaDB-specific querying. |
| **Dependency Management** | Dual dependency management: `requirements.txt` for Docker builds, `pyproject.toml` + `uv.lock` for reproducible local development with the `uv` package manager. |
| **Environment Configuration** | `.env.example` template, `python-dotenv` for local loading, and Docker Compose `env_file` for containerized deployment. |
| **Error Handling** | Comprehensive `try/except` blocks with structured logging at every pipeline stage, plus FastAPI `HTTPException` for meaningful API error responses. |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.12 | Core application language |
| **Backend Framework** | FastAPI + Uvicorn | Async REST API with auto-generated docs |
| **Frontend Framework** | Streamlit | Interactive chat UI |
| **LLM** | Llama-3.3-70B-Versatile | Answer generation & query rewriting |
| **LLM Inference** | Groq API | Ultra-low latency LLM inference |
| **Embeddings** | Sentence Transformers (`BAAI/bge-base-en-v1.5`) | 768-dim dense vector embeddings |
| **Vector Database** | ChromaDB (Persistent) | Embedding storage & similarity search |
| **Keyword Search** | BM25 (`rank-bm25`) | Sparse lexical retrieval |
| **Reranking** | Cohere Rerank v3.5 | Neural cross-encoder reranking |
| **Orchestration** | LangChain | RAG pipeline orchestration & abstractions |
| **PDF Parsing** | PyMuPDF | Fast, reliable PDF text extraction |
| **Data Validation** | Pydantic | Request/response schema validation |
| **Containerization** | Docker + Docker Compose | Multi-service containerization |
| **CI/CD** | GitHub Actions | Automated build, push & deploy |
| **Cloud** | AWS EC2 + ECR | Production hosting & container registry |
| **Package Manager** | uv | Fast, reproducible dependency resolution |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [Groq API Key](https://console.groq.com/keys) (free tier available)
- [Cohere API Key](https://dashboard.cohere.com/api-keys) (free tier available)

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/divyansh-iitk/PDF-ChatBot.git
cd PDF-ChatBot

# Create .env file
cp .env.example .env
# Edit .env and add your API keys

# Build and run
docker compose up --build
```

| Service | URL |
|---|---|
| **Frontend (Chat UI)** | [http://localhost:8501](http://localhost:8501) |
| **Backend API** | [http://localhost:8000](http://localhost:8000) |
| **API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) |

### Option 2: Local Setup

1. **Clone & enter the project:**
    ```bash
    git clone https://github.com/divyansh-iitk/PDF-ChatBot.git
    cd PDF-ChatBot
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    ```

5. **Start the FastAPI backend:**
    ```bash
    cd backend
    uvicorn app.main:app --reload
    ```
    The API will be live at `http://127.0.0.1:8000` (interactive docs at `/docs`).

6. **Start the Streamlit frontend** (new terminal):
    ```bash
    streamlit run frontend/app.py
    ```
    The frontend will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
PDF-ChatBot/
├── backend/                        # FastAPI backend service
│   ├── app/
│   │   ├── main.py                 # FastAPI app initialization & lifespan manager
│   │   ├── routes/
│   │   │   ├── upload.py           # POST /api/upload — PDF upload & ingestion
│   │   │   └── query.py            # POST /api/query — Retrieval + reranking + generation
│   │   └── schemas/
│   │       └── query.py            # Pydantic request models
│   ├── rag/                        # Core RAG pipeline modules
│   │   ├── loader.py               # PDF parsing with PyMuPDF + metadata enrichment
│   │   ├── splitter.py             # Recursive text chunking with UUID assignment
│   │   ├── embeddings.py           # Sentence Transformers embedding manager
│   │   ├── vectorstore.py          # ChromaDB persistent vector store wrapper
│   │   ├── BM25.py                 # BM25 keyword retriever with text normalization
│   │   ├── retriever.py            # Custom BaseRetriever subclass for vector search
│   │   ├── cohere_reranker.py      # Cohere reranking via ContextualCompressionRetriever
│   │   ├── ingest.py               # End-to-end PDF processing orchestrator
│   │   └── llm.py                  # Groq LLM client (query rewriting + answer gen)
│   ├── utils/
│   │   └── config.py               # Centralized pipeline configuration (dataclasses)
│   ├── logger/
│   │   └── __init__.py             # Custom file-based logging setup
│   ├── Dockerfile                  # Backend container image
│   └── requirements.txt            # Backend Python dependencies
├── frontend/                       # Streamlit frontend service
│   ├── app.py                      # Chat UI with session state & source display
│   ├── Dockerfile                  # Frontend container image
│   └── requirements.txt            # Frontend Python dependencies
├── .github/
│   └── workflows/
│       └── aws.yaml                # CI/CD: Build → ECR → EC2 deploy
├── compose.yaml                    # Docker Compose multi-service orchestration
├── pyproject.toml                  # Python project metadata (uv package manager)
├── uv.lock                         # Lockfile for reproducible dependency resolution
├── requirements.txt                # Root-level Python dependencies
├── .env.example                    # Required environment variables template
└── LICENSE                         # MIT License
```

---

## ⚙️ Configuration

All RAG pipeline hyperparameters are centralized in [`backend/utils/config.py`](backend/utils/config.py):

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 800 | Characters per text chunk |
| `chunk_overlap` | 100 | Overlap between consecutive chunks |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | Sentence Transformers model (768-dim) |
| `top_k` (vector retriever) | 10 | Documents retrieved from vector store |
| `score_threshold` | 0.5 | Minimum cosine similarity for inclusion |
| `top_k` (BM25) | 10 | Documents retrieved from BM25 |
| `BM25 weight` | 0.5 | Weight for keyword search in ensemble |
| `Vector weight` | 0.5 | Weight for semantic search in ensemble |
| `Reranker top_n` | 4 | Final documents retained after reranking |
| `LLM model` | `llama-3.3-70b-versatile` | Groq-hosted LLM for generation |
| `temperature` | 0.1 | LLM sampling temperature |
| `max_tokens` | 1024 | Maximum tokens in LLM response |
| `last_n_chats` | 3 | Chat history window for query rewriting |

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — confirms API is running |
| `POST` | `/api/upload` | Upload a PDF file for parsing, chunking, embedding, and indexing |
| `POST` | `/api/query` | Query ingested documents — returns answer, sources, scores, and rewritten query |

### Example: Query Request

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

### Example: Query Response

```json
{
  "answer": "The document primarily discusses...",
  "sources": [
    {
      "content": "Relevant text chunk from the PDF...",
      "metadata": {
        "source_file": "document.pdf",
        "relevance_score": 0.95
      }
    }
  ],
  "num_sources": 4,
  "rewritten_query": "What is the main topic of the document?"
}
```

---

## 🔄 CI/CD Pipeline

The project includes a fully automated CI/CD pipeline via GitHub Actions:

```
Push to `cicd` branch
        │
        ▼
┌──────────────────────────┐
│   Continuous Integration  │
│   (GitHub-hosted runner)  │
│                           │
│  1. Checkout code         │
│  2. Configure AWS creds   │
│  3. Login to Amazon ECR   │
│  4. Build Docker images   │
│  5. Push to ECR           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Continuous Deployment    │
│  (Self-hosted EC2 runner) │
│                           │
│  1. Pull latest images    │
│  2. Stop old containers   │
│  3. Start new containers  │
│  4. Configure networking  │
└──────────────────────────┘
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/divyansh-iitk">Divyansh Yadav</a>
</p>