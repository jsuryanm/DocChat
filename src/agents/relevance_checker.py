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
        "You are a strict relevance classifier. Your only job is to decide "
        "whether the provided passages contain information to answer the question.\n\n"
        "Reply with ONLY one label:\n"
        "CAN_ANSWER: the passages contain direct, explicit information that answers the question\n"
        "PARTIAL: the passages mention the topic but lack enough detail to fully answer\n"
        "NO_MATCH: the passages do not mention the specific subject of the question at all\n\n"
        "IMPORTANT: If the question asks about a specific product, person, or entity "
        "that is NOT mentioned by name in the passages, reply NO_MATCH."
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
        
        label = response.strip().upper()

        valid = {"CAN_ANSWER","PARTIAL","NO_MATCH"}

        for v in valid:
        
            if v in label:
                logger.info(f"Relevance classification: {v}")
                return v
        
        logger.warning(f"Unrecognized label: {label}")
        return "NO_MATCH"