from typing import Optional 
from src.retriever.builder import RetrieverBuilder,EmptyRetriever
from src.agents.workflow import AgentWorkflow

_builder: Optional[RetrieverBuilder] = None 
_workflow: Optional[AgentWorkflow] =  None 

def get_builder() -> RetrieverBuilder:
    if _builder is None:
        raise RuntimeError("RetrieverBuilder not initialized. Call init_singleton() at startup")
    return _builder

def get_workflow() -> AgentWorkflow:
    if _workflow is None: 
        raise RuntimeError("AgentWorkflow not initialized. Call init_singleton() at startup")
    return _workflow

async def init_singletons():
    global _workflow,_builder 
    _builder = RetrieverBuilder()
    retriever = EmptyRetriever()
    _workflow = AgentWorkflow(retriever=retriever,builder=_builder)

async def rebuild_workflow(docs):
    """Called aftter document upload to swap in real retriever"""
    global _workflow
    builder = get_builder()
    retriever = await builder.build_hybrid_retriever(docs)
    _workflow = AgentWorkflow(retriever=retriever, builder=builder)