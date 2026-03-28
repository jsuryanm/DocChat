# System Architecture

DocChat uses Agentic RAG architecture.

Instead of:

Retrieve → Answer

We use:

Reason → Retrieve → Verify → Retry

---

# Layers

Interface Layer

Streamlit UI
A2A server

---

Agent Layer

LangGraph workflow.

Agents:

Rewrite
Research
Verify
Grade
Reflect

---

Retrieval Layer

Hybrid search:

BM25
Vector search

Reranking layer.

---

Tool Layer

MCP tools.

Optional enhancement.

---

Delegation Layer

Remote A2A agents.

---

Data Layer

Chroma vector DB.

Document cache.

---

# Reliability Design

Failures handled by:

Retries
Fallback answers
Delegation
Verification

---

# Execution Model

Async pipeline.

Concurrent retrieval.

Parallel reranking.

---

# Observability

Logging via loguru.

Future:

LangSmith
Tracing
Metrics

---

# Scalability Strategy

Future improvements:

Distributed agents
Queue workers
Vector sharding
Caching layer

---

# Design Principles

Modular agents
Typed state
Deterministic routing
Async execution
Structured outputs