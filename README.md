# DocChat 🐥

A document Q&A application powered by **Docling**, **LangChain**, **LangGraph**, and **Groq** — with a **Streamlit** UI.

---

## Architecture

```
app.py  (Streamlit UI)
│
├── document_processor/
│   └── file_handler.py      # Docling → Markdown → chunk splits (with disk cache)
│
├── retriever/
│   └── builder.py           # BM25 + Chroma hybrid retriever (local HF embeddings)
│
├── agents/
│   ├── relevance_checker.py # Groq llama-3.1-8b-instant  → CAN_ANSWER / PARTIAL / NO_MATCH
│   ├── research_agent.py    # Groq llama-3.3-70b-versatile → draft answer (+ streaming)
│   ├── verification_agent.py# Groq llama-3.1-8b-instant  → fact-check report
│   └── workflow.py          # LangGraph StateGraph orchestrator
│
├── config/
│   ├── constants.py         # File size / type limits
│   └── settings.py          # Pydantic settings (reads .env)
│
└── utils/
    └── logging.py           # Loguru rotating log
```

### LangGraph workflow

```
START
  │
  ▼
check_relevance ──── NO_MATCH ──► END  (short-circuit, no LLM calls wasted)
  │
  │ CAN_ANSWER / PARTIAL
  ▼
research
  │
  ▼
verify ──── Supported:NO (retry < 1) ──► research  (loop once)
  │
  │ Supported:YES  OR  retry limit reached
  ▼
END
```
## Setup

### 1. Install dependencies

```bash
uv add -r requirements.txt
```

### 2. Add your Groq API key

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your key>
# Free keys: https://console.groq.com/keys
```

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Upload** one or more PDF / DOCX / TXT / MD files in the sidebar.
2. **Type** your question in the text area (or load an example from the dropdown).
3. Hit **Submit 🚀**.

The pipeline will:
- Check whether your documents are relevant to the question.
- Generate a detailed answer grounded in the document content.
- Verify the answer for factual support and flag any unsupported claims.

Previous questions are saved in the **Session History** section at the bottom.

---

## Models used

```
| Agent | Model | Why |
|---|---|---|
| Relevance checker | `llama-3.1-8b-instant` | Tiny prompt, binary output — speed matters most |
| Research agent | `llama-3.3-70b-versatile` | Best reasoning for answer quality |
| Verification agent | `llama-3.1-8b-instant` | Structured output, fast |
```
All models run on **Groq** hardware — typically 10–30× faster than hosted IBM/OpenAI endpoints.

---

## Configuration

All tuneable settings live in `config/settings.py` and can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required** |
| `RELEVANCE_MODEL` | `llama-3.1-8b-instant` | Groq model for relevance check |
| `RESEARCH_MODEL` | `llama-3.3-70b-versatile` | Groq model for answer generation |
| `VERIFY_MODEL` | `llama-3.1-8b-instant` | Groq model for verification |
| `VECTOR_SEARCH_K` | `8` | Top-k chunks for vector retrieval |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `CACHE_DIR` | `document_cache` | Docling parse cache directory |
| `CACHE_EXPIRE_DAYS` | `7` | Cache TTL |

---

## Project structure

```
docchat/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── config/
│   ├── __init__.py
│   ├── constants.py
│   └── settings.py
├── document_processor/
│   ├── __init__.py
│   └── file_handler.py
├── retriever/
│   ├── __init__.py
│   └── builder.py
├── agents/
│   ├── __init__.py
│   ├── relevance_checker.py
│   ├── research_agent.py
│   ├── verification_agent.py
│   └── workflow.py
└── utils/
    ├── __init__.py
    └── logging.py
```
