from pydantic import BaseModel,Field 

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import settings

class AnswerGrade(BaseModel):
    quality: str = Field()
    reason: str = Field()

_SYSTEM = """You are an answer quality grader.
 
Given a question and an answer, assign one of three quality labels:
 
HIGH   — The answer is complete, directly addresses the question,
          contains no unsupported claims, and requires no follow-up.
 
MEDIUM — The answer partially addresses the question, may lack detail,
          or contains minor gaps that don't make it wrong.
 
LOW    — The answer is incomplete, off-topic, contradicts the question,
          or states it cannot answer due to missing context.
 
Return ONLY the label (HIGH / MEDIUM / LOW) and a one-sentence reason.
Do not add any other text.
"""

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
             {answer}
             
             Grade the answer.""")
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