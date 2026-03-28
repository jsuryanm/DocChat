from typing import List,Annotated,Dict,Any
from typing_extensions import TypedDict
import operator

from langchain_core.documents import Document 

class AgentState(TypedDict):
    question: str 
    rewritten_question: str 

    documents: List[Document]
    reranked_docs: List[Document]
    
    draft_answer: str 
    final_answer: str 
    confidence: str
    answer_quality: str
    grounded: bool
    retry_count: int
    failure_reason: str
    
    draft_history: Annotated[List[str],operator.add]
    reasoning_steps: Annotated[List[str],operator.add] 

    # MCP
    tool_calls: List[Dict[str,Any]]
    mcp_tool_call_results: Annotated[List[str],operator.add]

    # A2A
    # True whenn the draft_answer originated from remote to A2A agent
    delegated: bool