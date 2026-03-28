# DocChat – Production Agentic RAG System

A production-grade **Agentic Retrieval Augmented Generation (RAG)** system built with:

• LangChain v1  
• LangGraph orchestration  
• Async Python architecture  
• MCP tool integration  
• A2A agent delegation  
• Hybrid retrieval (BM25 + Vector)  
• Verification + Reflexion retry loop  

This project implements a **self-correcting AI document assistant** that reasons about answers instead of just generating them.

Unlike basic RAG:

Retrieve → Answer ❌

This system performs:

Rewrite → Retrieve → Rerank → Reason → Verify → Retry → Finalize ✅

---

# Features

## Agentic Workflow
LangGraph controlled reasoning pipeline.

## Hybrid Retrieval
Combines:

• BM25 keyword search  
• Vector similarity search  

## Cross Encoder Reranking
Improves document ordering accuracy.

## Relevance Classification
Avoids wasting tokens on unrelated context.

## Grounded Answer Generation
Research agent only answers from provided documents.

## Hallucination Prevention
Verification agent ensures answer grounding.

## Reflexion Retry Loop
Agent retries when:

• Answer not grounded  
• Quality low  
• Missing information  

## MCP Tool Integration
External tools dynamically loaded.

## A2A Delegation
If documents insufficient:

System calls remote specialist agent.

## Async Execution
Entire pipeline async.

## Structured Agent Outputs
All agents use Pydantic schemas.

---

# Architecture Overview

## High Level Pipeline

```
User Question
      |
      v
Query Rewriter
      |
      v
Hybrid Retriever
      |
      v
Cross Encoder Reranker
      |
      v
Relevance Classifier
      |
      v
Research Agent
      |
      v
MCP Tool Use (optional)
      |
      v
Answer Grader
      |
      v
Verification Agent
      |
      v
Reflexion Loop
      |
      v
Final Answer
```

Fallback path:

```
No documents found
        |
        v
A2A Remote Agent
        |
        v
Verification
        |
        v
Finalize
```

---

# Installation

## Setup

```
pip install uv 
uv venv --python 3.12.12
uv init
```

## Clone project

```bash
git clone https://github.com/yourrepo/docchat

cd docchat
```

## Install dependencies

```bash
uv pip install -e .
```

---

# Running The System

## Run Streamlit UI

```bash
streamlit run app.py
```

## Run Tests

```bash
pytest -v
```

---


