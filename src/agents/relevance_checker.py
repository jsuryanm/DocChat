import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever

from src.config.settings import settings 
from src.custom_logger.logger import logger 


class RelevanceChecker:

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict relevance classifier.\n"
        "Return ONLY one word:\n"
        "CAN_ANSWER or PARTIAL or NO_MATCH"   
    ),
    (
        "human",
        "Question: {question}\n\nPassages:\n{passages}\n\nLabel:"
    )])

    def __init__(self):
        
        llm = ChatOpenAI(model=settings.RELEVANCY_MODEL,
                       temperature=0,
                       max_tokens=settings.RELEVANCE_MAX_TOKENS,
                       api_key=settings.OPENAI_API_KEY)
        
        self.chain = self.PROMPT_TEMPLATE | llm | StrOutputParser()

    async def check(self,question: str,
                    retriever: EnsembleRetriever,
                    k: int = 3) -> str:
        """Retrieve top k chunks and classify relevance
        Returns: CAN_ANSWER,PARTIAL,NO_MATCH"""
        
        try:
            top_docs = await asyncio.to_thread(retriever.invoke,
                                               question)
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            return "NO_MATCH"
        
        if not top_docs:
            logger.warning("No documents retrieved")
            return "NO_MATCH"
        
        passages = "\n\n".join(doc.page_content[:500] 
                               for doc in top_docs[:k])
        
        try:
            response = await self.chain.ainvoke({"question":question,
                                                 "passages":passages})
        except Exception as e:
            logger.warning(f"RelevnaceChecker chain error: {e}")
            return "NO_MATCH"
        
        label = (
            response
            .strip()
            .upper()
            .replace(".","")
            .replace(":","")
        )

        if label.startswith("CAN_ANSWER"):
            return "CAN_ANSWER"

        if label.startswith("PARTIAL"):
            return "PARTIAL"

        if label.startswith("NO_MATCH"):
            return "NO_MATCH"
        
        logger.warning(f"Unrecognized label: {label}")
        return "NO_MATCH"