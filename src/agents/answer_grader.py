from pydantic import BaseModel,Field 

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import settings

class AnswerGrade(BaseModel):
    quality: str = Field()
    reason: str = Field()


_SYSTEM = """Grade answer quality:
            
            HIGH
            MEDIUM
            LOW
            
            Based on completeness"""

class AnswerGrader:
    
    def __init__(self):
        llm = ChatOpenAI(model=settings.VERIFY_MODEL,
                         temperature=0,
                         api_key=settings.OPENAI_API_KEY)
        
        prompt =  ChatPromptTemplate.from_messages([
            ("system",_SYSTEM),
            
            ("human",
             """Question: 
             {question}
             
             Answer:
             {answer}""")
        ])

        self.chain = (
            prompt 
            | llm.with_structured_output(AnswerGrade)
        )
    
    async def grade(self,question,answer):

        result = await self.chain.ainvoke({"question":question,
                                           "answer":answer})
        
        return {"quality":result.quality,
                "reason":result.reason}