from pydantic import BaseModel,Field
from typing import List,Optional 

class ChatRequest(BaseModel):
    question: str = Field(...,min_length=1,max_length=2000)
    session_id: Optional[str] = None 

class ChatResponse(BaseModel):
    answer: str 
    reasoning_steps: List[str]
    retries: int 
    delegated: bool 
    web_used: bool
    session_id: Optional[str] = None 

class UploadResponse(BaseModel):
    message: str 
    chunks_indexed: int 
    files: List[str]

class HealthResponse(BaseModel):
    status: str 
    mcp_ready: bool
    retriever_ready: bool