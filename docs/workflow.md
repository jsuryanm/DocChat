# Workflow Documentation

Main orchestrator:

AgentWorkflow

Built using:

LangGraph StateGraph.

---

# Workflow Graph

START

Rewrite

Retrieve

Rerank

Relevance Check

Research

Grade

Verify

Reflect (retry loop)

Finalize

END

---

# Routing Logic

After rerank:

If no docs:

Delegate to A2A

Else:

Check relevance

---

After relevance:

NO_MATCH:

Finalize

Else:

Research

---

After research:

If tools requested:

ToolNode

Else:

Grade

---

After verify:

If grounded AND HIGH:

Finalize

If retries exceeded:

Finalize

Else:

Reflect

---

# Retry Loop

Reflect:

Increment retry counter.

Return to rewrite.

Max retries = 2.

---

# Delegation Flow

If:

No documents

Call remote agent.

Then:

Grade
Verify
Finalize

---

# Finalization Rules

If answer empty:

Return fallback message.

If NO_MATCH:

Explain missing docs.

Else:

Accept answer.

---

# Failure Modes Covered

No docs
Low quality
Hallucination
Tool failure
Remote failure

---

# Key Strength

Self correcting pipeline.

Not single pass RAG.