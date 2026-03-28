# Agent State Design

State defined as TypedDict.

Contains all pipeline data.

Key principle:

State = single source of truth.

---

# Core Fields

question

Original user query.

rewritten_question

Improved retrieval query.

documents

Retrieved docs.

reranked_docs

Top ranked docs.

---

# Answer Fields

draft_answer

Current answer.

final_answer

Accepted answer.

confidence

Research confidence.

answer_quality

Grader output.

grounded

Verification result.

---

# Retry Fields

retry_count

Number of retries.

draft_history

Previous answers.

reasoning_steps

Agent decisions.

---

# MCP Fields

tool_calls

Requested tools.

mcp_tool_call_results

Tool outputs.

---

# A2A Fields

delegated

True if remote answer used.

---

# Design Philosophy

State must:

Be minimal
Be explicit
Avoid hidden memory
Support retries
Enable debugging