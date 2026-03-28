# Agent Skills

## Query Rewriter

Purpose:
Improve retrieval quality.

Input:
User question

Output:
Better search query

---

## Research Agent

Purpose:
Generate grounded answer.

Input:

Question
Documents

Output:

Answer
Confidence
Missing info

---

## Verification Agent

Purpose:
Prevent hallucinations.

Checks:

Unsupported claims
Contradictions
Grounding

---

## Relevance Checker

Purpose:
Avoid wasting tokens.

Labels:

CAN_ANSWER
PARTIAL
NO_MATCH

---

## Answer Grader

Purpose:
Quality scoring.

Labels:

HIGH
MEDIUM
LOW

---

## Reflexion Agent

Purpose:
Retry decisions.

Logic:

Grounded + HIGH → accept

Retries exceeded → stop

Else → retry

---

## MCP Tool Skills

External abilities:

Search
File access
APIs

---

## A2A Skill

Delegation ability.

Used when:

No documents found.

---

# System Meta Skills

Self correction
Grounding enforcement
Delegation
Tool usage
Retry reasoning