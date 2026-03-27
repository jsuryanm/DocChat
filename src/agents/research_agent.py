from typing import List,Dict,AsyncGenerator,Optional
from pydantic import BaseModel, Field 
import asyncio
from tenacity import retry,stop_after_attempt,wait_exponential

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config.settings import settings 
from src.custom_logger.logger import logger 

_SYSTEM_PROMPT = """You are a research agent in a RAG system.

Answer ONLY using provided context.

Rules:

Be factual
Be concise
Do not hallucinate
If unsure → say insufficient context
List missing information if relevant

Return structured output.
"""

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([

    ("system", _SYSTEM_PROMPT),

    ("human",
    """
    Question:
    {question}

    Context:
    {context}

    Produce grounded answer.
    """)

])



class ResearchResult(BaseModel):
    answer: str = Field(description="Final grounded answer")
    confidence: str = Field(description="HIGH, MEDIUM or LOW")
    missing_information: str = Field(description="What context is missing if incomplete")

class ResearchAgent:
    MAX_DOCS = 4
    MAX_CHARS = 500
    TIMEOUT = 45

    def __init__(self,llm: Optional[ChatOpenAI] = None):

        
        self.llm = ChatOpenAI(model=settings.RESEARCH_MODEL,
                                    temperature=0.1,
                                    max_completion_tokens=settings.RESEARCH_MAX_TOKENS,
                                    api_key=settings.OPENAI_API_KEY)
                
        self.chain = (
            RESEARCH_PROMPT 
            | self.llm.with_structured_output(ResearchResult)    
        )
    
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(min=1,max=8))
    async def _call_llm(self,payload: Dict) -> ResearchResult:
        return await asyncio.wait_for(self.chain.ainvoke(payload),
                                      timeout=self.TIMEOUT)

    async def generate(self,
                       question: str,
                       documents: List[Document],
                       relevance: Optional[str] = None) -> Dict:
        
        if relevance == "NO_MATCH":
            logger.info("Skipping research due to NO_MATCH")

            return {"draft_answer":"Insufficient relevant context.",
                    "confidence":"LOW",
                    "context_used":"",
                    "doc_count":0}

        try:
            context = self._compress_documents(documents)

            payload = {"question":question,
                       "context":context}
            
            result = await self._call_llm(payload)
        
        except Exception as e:
            logger.error(f"ResearchAgent error: {e}")
            raise RuntimeError("ResearchAgent failed") from e 
        
        logger.info(f"Research completed | docs = {len(documents)} | confidence = {result.confidence}")
        
        return {"draft_answer":result.answer,
                "confidence":result.confidence,
                "missing_information":result.missing_information,
                "context_used":context,
                "doc_count":len(documents),
                "model":settings.RESEARCH_MODEL}
    
    async def stream(self,
                     question: str,
                     documents: List[Document]) -> AsyncGenerator[str,None]:
        
        context = self._compress_documents(documents)
        
        payload = {"question":question,
                   "context":context}
        
        try:
            async for chunk in self.chain.astream(payload):
                if chunk:
                    yield str(chunk)
        
        except asyncio.CancelledError:  
            logger.warning(f"Research stream cancelled")
            raise 

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise RuntimeError("Streaming Failed") from e 
        
    def _compress_documents(self,docs: List[Document]) -> str:

        if not docs:
            return ""
        
        docs = sorted(docs,
                      key=lambda d:d.metadata.get("score",0),
                      reverse=True)
        
        trimmed = []
        total_chars = 0

        for doc in docs[:self.MAX_DOCS]:
            text = doc.page_content.strip()

            if len(text) > self.MAX_CHARS:
                text =  text[:self.MAX_CHARS] + "..."
            
            total_chars += len(text)
            trimmed.append(text)
        
        logger.info(f"Context built | chunks = {len(trimmed)} | chars = {total_chars}")

        return "\n\n".join(trimmed)