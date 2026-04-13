from typing import List, Annotated, Dict, Any
from typing_extensions import TypedDict
import operator

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    rewritten_question: str

    documents: List[Document]
    reranked_docs: List[Document]

    relevance_label: str

    draft_answer: str
    final_answer: str
    confidence: str
    answer_quality: str
    grounded: bool
    retry_count: int
    failure_reason: str

    verification_failed: bool

    draft_history: Annotated[List[str], operator.add]
    reasoning_steps: Annotated[List[str], operator.add]

    # MCP
    tool_calls: List[Dict[str, Any]]
    mcp_tool_results: Annotated[List[str], operator.add]
    web_used: bool  # True when Tavily web search was used
    # A2A
    # True when draft_answer originated from a remote A2A agent
    delegated: bool
