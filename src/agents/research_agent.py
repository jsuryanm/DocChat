from typing import List,Dict,Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config.settings import settings 
from src.custom_logger.logger import logger 

_SYSTEM_PROMPT = """You are an AI assistant that answers questions ONLY from the provided context.

Rules:

Be concise
Be factual
Include relevant data points

If context insufficient -> say so.
"""


class ResearchAgent:
    MAX_DOCS = 4
    MAX_CHARS = 500

    def __init__(self):

        self.llm = ChatOpenAI(model=settings.RESEARCH_MODEL,
                              temperature=0.2,
                              max_completion_tokens=settings.RESEARCH_MAX_TOKENS,
                              api_key=settings.OPENAI_API_KEY)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",_SYSTEM_PROMPT),
            ("human","""
                Question:
                {question}

                Context:
                {context}
                
                Answer:
            """)
        ])

        self.chain = (
            prompt 
            | self.llm 
            | StrOutputParser()
        )

    def generate(self,
                 question: str,
                 documents: List[Document]) -> Dict[str,str]:
        try:
            context = self._compress_documents(documents)
            draft = self.chain.invoke({"question":question,
                                       "context":context}).strip()
        
        except Exception as e:
            logger.error(f"ResearchAgent error: {e}")
            raise RuntimeError("ResearchAgent failed") from e 

        logger.info(f"Draft length: {len(draft)}")
        
        return {"draft_answer":draft,
                "context_used":context}
    
    def stream(self,
               question: str,
               documents: Generator[str,None,None]):
        
        context = self._compress_documents(documents)
        
        for chunk in self.chain.stream({"question":question,
                                        "context":context}):
            if chunk:
                yield chunk

    def _compress_documents(self,docs: List[Document]) -> str:
        trimmed = []

        for doc in docs[:self.MAX_DOCS]:
            text = doc.page_content.strip()

            if len(text) > self.MAX_CHARS:
                text =  text[:self.MAX_CHARS] + "..."
            
            trimmed.append(text)

        return "\n\n".join(trimmed)