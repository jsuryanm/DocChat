# Project Overview

This project implements an Agentic RAG system using LangGraph orchestration.

The system uses multiple specialized agents to:
- Retrieve information
- Generate answers
- Verify correctness
- Improve responses
- Produce final answers

Priority order:
Correctness > Grounding > Reliability > Speed

---

# Architecture

Workflow graph:

retrieve → research → draft → verify → reflexion → finalize

Agents must NOT bypass verification.

ReflexionAgent should only run when verification fails or confidence is low.

---

# Main Components

## Agents

### ResearchAgent
Responsibilities:
- Retrieve documents
- Perform hybrid retrieval (BM25 + vector)
- Use reranker if available

Output:
draft_answer
documents

### VerifierAgent
Responsibilities:
- Check hallucinations
- Verify grounding
- Check source support

Output:
verification_status
confidence_score
missing_facts

### ReflexionAgent
Responsibilities:
- Improve weak answers
- Fix hallucinations
- Add missing context
- Rewrite unclear answers

Must NOT retrieve new documents.

Only improve existing answer.

### FinalAnswerAgent
Responsibilities:
- Produce clean final response
- Ensure formatting
- Remove internal reasoning text

---

# State Management Rules

State must always contain:

question: str
documents: List[Document]
draft_answer: str
verification_status: str
final_answer: str

Rules:

Never delete fields from state.

Agents may only modify their owned fields.

ResearchAgent:
may update documents and draft_answer.

VerifierAgent:
may update verification_status and confidence_score.

ReflexionAgent:
may update draft_answer only.

FinalAnswerAgent:
sets final_answer.

---

# LangGraph Rules

Do not change graph structure unless necessary.

Conditional edges must remain:

verify → reflexion (if failed)
verify → finalize (if passed)

Do not introduce cycles unless explicitly required.

Avoid duplicate routing logic.

Use reducer pattern for state updates if needed.

---

# MCP Integration Rules

MCP tools must be:
- Fail safe
- Timeout protected
- Logged

Never assume MCP server availability.

Always handle tool failures gracefully.

---

# A2A Rules

Remote agents must:
- Be called asynchronously
- Have retry logic
- Handle timeout

Never block workflow waiting indefinitely.

---

# Coding Standards

Python version:
3.11+

Required:
Type hints
Pydantic models
Structured logging

Avoid:
Global mutable state
Blocking calls inside async functions
Silent exception handling

Always log errors.

---

# Error Handling Rules

Always:

Catch exceptions.
Log failures.
Return safe fallback.

Never:

Crash workflow.
Return None unexpectedly.
Swallow exceptions silently.

---

# Testing Rules

When modifying code:

Do not break:
retriever builder
agent interfaces
workflow state schema

If state changes:
update related agents.

---

# Known Issues

Possible issues:

ReflexionAgent may not be triggered.
Final answer sometimes not set.
Verification routing duplication.
State overwrite conflicts.

Fix root causes instead of patching.

---

# How Claude Should Modify Code

When updating files:

Provide FULL updated file.

Do NOT provide partial patches unless requested.

Do NOT change:
architecture
agent responsibilities
state structure

Unless explicitly asked.

Explain changes briefly.

Minimize unnecessary refactors.

---

# Output Expectations

When fixing bugs:

1 Explain problem
2 Explain root cause
3 Provide fix
4 Provide updated file

When suggesting improvements:

List:
Issue
Risk
Improvement

---

# Anti Patterns To Avoid

Do NOT:

Duplicate agent logic.
Mix verification and generation.
Let agents modify unrelated state.
Add hidden dependencies.
Break async execution.

---

# Preferred Patterns

Prefer:

Small focused agents.
Clear state transitions.
Deterministic routing.
Observable failures.
Idempotent agent execution.

---

# Project Goal

Build a reliable agentic RAG system that:

Minimizes hallucinations.
Produces grounded answers.
Is easy to extend.
Is production safe.

---

# Claude Instructions

Assume:

User is building production grade AI system.

Always:

Preserve architecture.
Improve reliability.
Improve clarity.
Reduce hidden bugs.

Never:

Simplify into toy examples unless asked.
Remove important engineering patterns.