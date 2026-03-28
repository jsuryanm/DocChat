# CLAUDE.md

## Project Purpose

This project implements an Agentic RAG system using LangGraph.

Focus areas:

Reliability
Grounded answers
Verification
Retry logic
Agent orchestration

This is NOT a simple RAG.
This is a production agent pipeline.

---

# Key Concepts

## Agent Workflow

Main orchestrator:

AgentWorkflow

Controls all routing.

## State

Typed state defined in:

agents/state.py

State is passed between nodes.

## Agents

QueryRewriter
ResearchAgent
VerificationAgent
AnswerGrader
RelevanceChecker
ReflexionAgent

Each has single responsibility.

---

# Important Design Rules

Agents must:

Return structured output
Be async
Avoid hallucination
Use settings config

---

# Workflow Graph

Defined in:

workflow.py

Uses:

StateGraph
Conditional edges
Retry routing

---

# Tool System

MCP tools optional.

Loaded at startup.

Injected into ResearchAgent.

---

# Delegation

Remote agents called via A2A.

File:

a2a/client.py

---

# Coding Rules

Never use deprecated LangChain patterns.

Always prefer:

Runnable
Structured outputs
Async execution

---

# Extension Points

Add new agent:

Add node
Add routing
Update state

Add tool:

Register MCP server

Add verification:

Extend VerificationAgent

---

# Testing Strategy

Unit test:

Agents
Retriever
Workflow routing

Integration test:

Full pipeline.

---

# Philosophy

Correct > Fast

Grounded > Creative

Deterministic > Fancy