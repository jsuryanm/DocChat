# DocChat — Agentic RAG System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1.1+-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.55+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5+-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square" />
</p>

<p align="center">
  <strong>Agentic RAG pipeline with self-correction, grounding verification, web search fallback, and multi-agent orchestration via A2A protocol.</strong>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Running the UI](#running-the-ui)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Extending the System](#extending-the-system)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

DocChat is a self-correcting agent pipeline that:

1. **Rewrites** the user query for better retrieval
2. **Retrieves** documents using hybrid BM25 + vector search
3. **Reranks** results with a cross-encoder model
4. **Checks relevance** before spending tokens on generation
5. **Generates** a grounded answer using only retrieved context
6. **Falls back to web search** (Tavily MCP) when local documents cannot answer
7. **Verifies** the answer against source documents to prevent hallucination
8. **Grades** answer quality and **retries** with a rewritten query if quality is insufficient
9. **Delegates** to remote specialist agents via the A2A protocol when no local documents exist

The system is designed around the principle: **Correct > Fast. Grounded > Creative. Deterministic > Fancy.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface Layer                         │
│              FastAPI REST API  ·  Streamlit UI              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      Agent Layer                            │
│                                                             │
│   QueryRewriter → ResearchAgent → VerificationAgent         │
│   AnswerGrader  → ReflexionAgent → RelevanceChecker         │
│                                                             │
│               Orchestrated by LangGraph StateGraph          │
└───────────┬────────────────────┬───────────────────────────┘
            │                    │
┌───────────▼──────┐   ┌─────────▼──────────────────────────┐
│  Retrieval Layer │   │          Tool Layer                 │
│                  │   │                                     │
│  BM25 + Chroma   │   │  MCP Tools (Tavily web search)      │
│  Cross-encoder   │   │  A2A Remote Agent Delegation        │
│  Reranker        │   │                                     │
└───────────┬──────┘   └────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│                      Data Layer                          │
│           ChromaDB vector store  ·  Document cache      │
└──────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

The full workflow is a stateful LangGraph `StateGraph` with conditional routing and retry loops:

```
START
  │
  ▼
[Query Rewrite]          Improve query for retrieval
  │
  ▼
[Retrieve]               Hybrid BM25 + vector search
  │
  ▼
[Rerank]                 Cross-encoder reranking (ms-marco-MiniLM)
  │
  ├─── no docs + REMOTE_AGENT_URL set ──► [Delegate → A2A Agent]
  │                                                │
  ▼                                                ▼
[Relevance Check]                             [Grade]
  │                                                │
  ├─── NO_MATCH + Tavily available ──► [Web Search (Tavily MCP)]
  │                                                │
  ├─── NO_MATCH (no Tavily) ──► [Finalize]         │
  │                                                │
  ▼                                                │
[Research Agent]  ◄──────────────────────────────-┘
  │
  ├─── tool_calls present ──► [MCP ToolNode] ──► [Research Agent]
  │
  ▼
[Answer Grader]          HIGH / MEDIUM / LOW quality label
  │
  ▼
[Verification Agent]     Checks grounding against source docs
  │
  ├─── grounded=True  + quality=HIGH ──► [Finalize]
  ├─── verification_failed=True ──────► [Finalize]   (tool failure, not content failure)
  ├─── delegated=True ────────────────► [Finalize]
  ├─── retries >= MAX_RETRIES ────────► [Finalize]
  │
  └─── retry ──► [Reflect] ──► [Query Rewrite]   (max 2 retries)

END ◄── [Finalize]
```

### Routing rules summary

| Condition | Route |
|---|---|
| No local docs + `REMOTE_AGENT_URL` set | → A2A delegate |
| Relevance = `NO_MATCH` + Tavily available | → Web search |
| Relevance = `NO_MATCH` + no Tavily | → Finalize (fallback message) |
| Relevance = `CAN_ANSWER` or `PARTIAL` | → Research agent |
| Research returns tool calls | → MCP ToolNode → Research |
| Grounded + HIGH quality | → Accept |
| Verification tool failed (`UNKNOWN`) | → Accept (don't retry infra failures) |
| Quality < HIGH or not grounded | → Retry (up to `MAX_RETRIES = 2`) |

---

## Key Features

**Self-correcting pipeline** — the system retries with a rewritten query when answers are low quality or ungrounded. Retry decisions are made by `ReflexionAgent` using grounding and quality signals, not random back-off.

**Hallucination prevention** — `VerificationAgent` checks every generated answer against source documents before accepting it. Unsupported claims and contradictions are explicitly detected.

**Hybrid retrieval** — BM25 keyword search combined with dense vector search in an ensemble retriever, followed by cross-encoder reranking. Outperforms either approach alone on domain-specific documents.

**Tavily web search fallback** — when uploaded documents cannot answer the question (`NO_MATCH`), the pipeline automatically falls back to Tavily web search via MCP rather than returning an empty response.

**A2A remote agent delegation** — when no documents are loaded at all, questions are routed to a remote specialist agent using the [A2A protocol](https://a2aprotocol.ai). The remote agent's answer goes through the same grading and verification pipeline.

**MCP tool integration** — the Research Agent can call any MCP-compatible tool mid-generation. New tools can be registered without changing agent code.

**Async throughout** — every agent, retrieval step, and tool call is fully async. Blocking operations (Chroma, cross-encoder, file I/O) are wrapped in `asyncio.to_thread`.

**Document caching** — processed document chunks are cached by SHA-256 file hash with a configurable TTL. Re-uploading the same file is instant.

**Dual interface** — FastAPI REST API for production integration, Streamlit UI for rapid prototyping and demos.

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| LLM | OpenAI GPT (configurable per agent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB |
| Keyword search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Document parsing | Docling (PDF, DOCX, TXT, Markdown) |
| Web search | Tavily via MCP (`langchain-mcp-adapters`) |
| Agent protocol | A2A SDK (`a2a-sdk`) |
| API framework | FastAPI + Uvicorn |
| UI | Streamlit |
| Logging | Loguru |
| Settings | Pydantic Settings |
| Testing | Pytest + pytest-asyncio |

---

## Project Structure

```
docchat/
├── main.py                          # FastAPI entrypoint (production)
├── app.py                           # Streamlit entrypoint (UI)
├── pyproject.toml
├── requirements.txt
├── .env                             # secrets (not committed)
├── CLAUDE.md                        # AI assistant instructions
│
├── src/
│   ├── agents/
│   │   ├── state.py                 # AgentState TypedDict (single source of truth)
│   │   ├── workflow.py              # LangGraph StateGraph + all routing logic
│   │   ├── query_rewriter.py        # Improves user query for retrieval
│   │   ├── research_agent.py        # Generates grounded answer from context
│   │   ├── verfication_agent.py     # Checks answer grounding vs. source docs
│   │   ├── answer_grader.py         # Scores answer quality (HIGH/MEDIUM/LOW)
│   │   ├── reflexion_agent.py       # Decides: accept / retry / stop
│   │   ├── relevance_checker.py     # Labels docs: CAN_ANSWER / PARTIAL / NO_MATCH
│   │   └── web_search_agent.py      # Tavily MCP fallback when NO_MATCH
│   │
│   ├── retriever/
│   │   └── builder.py               # Hybrid retriever + cross-encoder reranker
│   │
│   ├── document_processor/
│   │   └── file_handler.py          # Docling parsing, chunking, SHA-256 cache
│   │
│   ├── tools/
│   │   ├── mcp_tools.py             # MCP server registry (Tavily, filesystem, ...)
│   │   └── mcp_client.py            # Startup/shutdown, tool loading, health check
│   │
│   ├── a2a/
│   │   ├── client.py                # Calls remote A2A agents with retry
│   │   └── server.py                # Exposes this agent as an A2A endpoint
│   │
│   ├── api/
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── dependencies.py          # Singleton workflow + builder injection
│   │   └── router.py                # /health, /documents/upload, /chat/invoke
│   │
│   ├── config/
│   │   ├── settings.py              # Pydantic Settings (reads .env)
│   │   └── constants.py             # File size limits, allowed types
│   │
│   └── custom_logger/
│       └── logger.py                # Loguru configuration
│
├── tests/
│   ├── test_builder.py
│   ├── test_research_agent.py
│   ├── test_verification_agent.py
│   ├── test_relevance_checker.py
│   ├── test_query_rewriter.py
│   ├── test_answer_grader.py
│   ├── test_reflexion_agent.py
│   ├── test_workflow.py
│   ├── test_mcp_tools.py
│   ├── test_a2a_client.py
│   └── test_a2a_server.py
│
└── docs/
    ├── architecture.md
    ├── workflow.md
    ├── state.md
    └── skills.md
```

---

## Quick Start

### Requirements

- Python - 3.12.12
- uv package manager 
- An OpenAI API key
- A Tavily API key (for web search fallback; optional but recommended)

### 1. Clone and install

```bash
1. Setup uv package manager

pip install uv 
uv venv --python 3.12.12
uv init

2. Clone the repository

git clone https://github.com/your-org/docchat.git
cd docchat

3. Install requirements.txt
Activate virtual environemnt

.venv/bin/activate          
# Windows: .venv\Scripts\activate

uv add -r requirements.txt
uv pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
OPENAI_API_KEY=sk-...

# Required for web search fallback
TAVILY_API_KEY=tvly-...

# Optional: remote A2A specialist agent
REMOTE_AGENT_URL=http://localhost:9001

# Optional overrides (defaults shown)
RESEARCH_MODEL=gpt-4o-mini
VERIFY_MODEL=gpt-4o-mini
RELEVANCY_MODEL=gpt-4o-mini
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_SEARCH_K=8
RERANKER_TOP_N=3
```

### 3a. Run the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 3b. Run the Streamlit UI

```bash
streamlit run app.py
```

UI available at [http://localhost:8501](http://localhost:8501)

---

## Configuration

All settings are managed via `src/config/settings.py` using Pydantic Settings. Every value can be overridden by an environment variable or `.env` file entry.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. OpenAI API key. |
| `TAVILY_API_KEY` | — | Required for web search fallback. |
| `REMOTE_AGENT_URL` | `""` | A2A agent URL. Empty disables delegation. |
| `RESEARCH_MODEL` | `gpt-4o-mini` | LLM used by ResearchAgent. |
| `VERIFY_MODEL` | `gpt-4o-mini` | LLM used by VerificationAgent. |
| `RELEVANCY_MODEL` | `gpt-4o-mini` | LLM used by RelevanceChecker and QueryRewriter. |
| `EMBEDDINGS_MODEL` | `text-embedding-3-small` | OpenAI embeddings model. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder. |
| `VECTOR_SEARCH_K` | `8` | Number of docs retrieved per search. |
| `HYBRID_RETRIEVER_WEIGHTS` | `[0.5, 0.5]` | BM25 vs. vector search blend. |
| `RERANKER_TOP_N` | `3` | Docs kept after reranking. |
| `RESEARCH_MAX_TOKENS` | `1600` | Max tokens for Research LLM call. |
| `VERIFY_MAX_TOKENS` | `1500` | Max tokens for Verification LLM call. |
| `RELEVANCE_MAX_TOKENS` | `500` | Max tokens for Relevance LLM call. |
| `CHROMA_DB_PATH` | `./chroma_db` | Persistent ChromaDB directory. |
| `CACHE_DIR` | `document_cache` | Processed document chunk cache. |
| `CACHE_EXPIRE_DAYS` | `7` | Cache TTL in days. |
| `MCP_ENABLED_SERVERS` | `*` | Comma-separated MCP server names, or `*` for all. |
| `A2A_HOST` | `0.0.0.0` | Host for this agent's A2A server. |
| `A2A_PORT` | `9000` | Port for this agent's A2A server. |
| `LOG_LEVEL` | `INFO` | Loguru log level. |

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

---

### `GET /health`

Returns system health and readiness of MCP and retriever components.

**Response `200`:**

```json
{
  "status": "ok",
  "mcp_ready": true,
  "retriever_ready": true
}
```

---

### `POST /documents/upload`

Upload one or more documents to be indexed. Supported formats: `.pdf`, `.txt`, `.docx`, `.md`.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `files` | `File[]` | One or more document files |

**Response `200`:**

```json
{
  "message": "Documents indexed successfully",
  "chunks_indexed": 142,
  "files": ["report.pdf", "notes.md"]
}
```

**Error responses:**

| Code | Reason |
|---|---|
| `400` | No files provided, or unsupported file type |
| `500` | Document processing or indexing failure |

---

### `POST /chat/invoke`

Ask a question. The full agentic pipeline runs synchronously and returns when a final answer is produced.

**Request body:**

```json
{
  "question": "What were the key findings in the Q3 report?",
  "session_id": "optional-client-session-id"
}
```

**Response `200`:**

```json
{
  "answer": "The Q3 report identified three key findings...",
  "reasoning_steps": [
    "Query rewritten",
    "Retrieved 8 documents",
    "Documents reranked",
    "Relevance check: CAN_ANSWER",
    "Research agent generated an answer",
    "Answer quality = HIGH",
    "Verification grounded = true",
    "Final answer accepted"
  ],
  "retries": 0,
  "delegated": false,
  "web_used": false,
  "session_id": "optional-client-session-id"
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | `string` | The final grounded answer |
| `reasoning_steps` | `string[]` | Ordered log of agent decisions |
| `retries` | `int` | Number of retry loops executed (0–2) |
| `delegated` | `bool` | `true` if answer came from a remote A2A agent |
| `web_used` | `bool` | `true` if Tavily web search was used as fallback |

**Error responses:**

| Code | Reason |
|---|---|
| `400` | Empty question |
| `500` | Internal pipeline failure |

---

### A2A server endpoint

DocChat exposes itself as an A2A-compatible agent. Remote orchestrators can discover it via the agent card endpoint:

```
GET http://localhost:9000/.well-known/agent.json
```

---

## Running the UI

The Streamlit interface provides a quick way to upload documents and ask questions without the API.

```bash
streamlit run app.py
```

**Features:**

- Drag-and-drop multi-file upload (PDF, DOCX, TXT, Markdown)
- Real-time spinner during processing and agent execution
- Final answer display with remote-agent attribution badge
- Collapsible reasoning steps panel showing every agent decision
- Collapsible draft history panel showing all retried answers

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run a specific module

```bash
pytest tests/test_workflow.py -v
pytest tests/test_builder.py -v
```

### Run with coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Test design

Unit tests use injected fakes — no real LLM or network calls are made. `FakeChain`, `FakeRetriever`, `FakeResearch`, and `FakeReranker` fixtures keep the suite fast, deterministic, and free to run in CI.

Tests that make real LLM calls are marked `@pytest.mark.integration` and require valid API keys:

```bash
pytest tests/ -v -m integration
```

---

## Design Decisions

**Why LangGraph instead of a simple chain?**

The pipeline has conditional branching (relevance check, retry loop, tool routing, delegation) and multiple failure modes that each require different handling. LangGraph's `StateGraph` makes all routing explicit, inspectable, and testable. A linear chain would require deeply nested try/except blocks with implicit control flow that is difficult to reason about or extend.

**Why hybrid retrieval (BM25 + vector)?**

BM25 handles exact keyword matches and rare technical terms well. Dense vector search handles semantic similarity and paraphrasing. The ensemble consistently outperforms either alone on domain-specific documents with precise terminology. The cross-encoder reranker then re-scores the merged candidate set using full attention between query and document text.

**Why verify *after* grading?**

Grading (quality assessment) and verification (grounding check) are separate signals. An answer can be high quality but partially hallucinated, or low quality but technically grounded. Both signals feed the reflexion decision independently. Separating them allows future tuning of each threshold without affecting the other.

**Why is `verification_failed` a separate flag from `grounded`?**

A LLM or network failure inside `VerificationAgent` produces `supported=UNKNOWN` — this is an infrastructure failure, not evidence that the answer is ungrounded. Without the flag, the retry loop would re-run the full pipeline on a perfectly valid answer, wasting tokens and latency. The flag short-circuits directly to `accept` for infrastructure failures only, leaving content-quality retries unaffected.

**Why does web search connect after the relevance check, not as a fallback after research?**

Web search is a fallback for when local documents genuinely cannot answer the question (`NO_MATCH`). Mixing web results with local document context mid-research would make grounding verification ambiguous — the verifier would need to know which source to check claims against. Keeping web search on its own branch, going directly to grading, preserves clean verification semantics.

**Why MCP for Tavily instead of direct LangChain tool integration?**

MCP allows tool registration, discovery, and lifecycle management independent of agent code. New tools (filesystem access, SQL databases, external APIs) can be added by registering an MCP server in `mcp_tools.py` without modifying any agent. The `ResearchAgent` and `WebSearchAgent` automatically pick up all registered tools at startup.

---

## Extending the System

### Add a new agent

1. Create `src/agents/your_agent.py` with a single async method returning a result dict
2. Add a node in `AgentWorkflow._build()`: `g.add_node("your_node", self._your_node)`
3. Add a handler method `async def _your_node(self, state) -> dict` returning a state patch
4. Wire edges with `g.add_edge` or `g.add_conditional_edges`
5. Add the new state field(s) to `AgentState` in `state.py`
6. Initialize the new field in `AgentWorkflow.run()` initial state dict

### Add a new MCP tool

1. Add the server config to `MCP_SERVERS` in `src/tools/mcp_tools.py`
2. Add the server name to `MCP_ENABLED_SERVERS` in `.env`
3. The tool is automatically available to all agents at next startup — no other changes needed

### Add a new document type

1. Add the file extension to `ALLOWED_TYPES` in `src/config/constants.py`
2. Add a handler branch in `DocumentProcessor._process_file()` using any Docling-compatible converter

### Connect a remote specialist agent

Set `REMOTE_AGENT_URL` in `.env` to the base URL of any A2A-compatible agent. DocChat will automatically delegate queries when no local documents are available. The remote answer passes through the standard grading and verification pipeline before being returned to the caller.

---

## Known Limitations

**Stateless across requests** — the FastAPI server does not maintain conversation history between `/chat/invoke` calls. Each request is a fresh pipeline run. Session-aware multi-turn conversation requires an external store (Redis, PostgreSQL) keyed on `session_id`.

**Single retriever instance** — `AgentWorkflow` holds one retriever built from one document set. Multiple users uploading different documents will overwrite each other's retriever. Production deployments with multiple users require per-session or per-user retriever isolation.

**Synchronous `/chat/invoke`** — the endpoint blocks until the full pipeline completes (typically 5–30 seconds depending on retries and LLM latency). For high-concurrency production use, consider a task queue (Celery, ARQ) with a status-polling endpoint, or SSE streaming.

**Reranker downloads on first run** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB) is downloaded from HuggingFace Hub on first use. Pre-download it in your Docker build step:

```dockerfile
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

**No document deletion API** — once documents are indexed into ChromaDB, there is no endpoint to remove them. A collection reset requires deleting the `chroma_db/` directory and restarting the server.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with LangGraph · LangChain · FastAPI · ChromaDB · Docling
</p>