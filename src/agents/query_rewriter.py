from pydantic import BaseModel,Field 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import settings 
from src.custom_logger.logger import logger 

class RewriteResult(BaseModel):
    rewritten_question: str = Field(description="Improved search query")

_SYSTEM = """Rewrite query for better retrieval.
            Keep meaning identical."""

class QueryRewriter: 

    def __init__(self):
        llm = ChatOpenAI(model=settings.RELEVANCY_MODEL,
                         temperature=0,
                         api_key=settings.OPENAI_API_KEY)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",_SYSTEM),
            ("human","{question}")
        ])

        self.chain = (
            prompt 
            | llm.with_structured_output(RewriteResult)
        )

    async def rewrite(self,question):
        try:
            result = await self.chain.ainvoke({"question":question})
            return result.rewritten_question
        
        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return question