from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever

from src.config.settings import settings 
from src.custom_logger.logger import logger 


class RelevanceChecker:

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     """You are a relevance classifier. Your job is to decide whether the provided passages
     contain enough information to answer the user's questions.\n\n
     
     Reply with ONLY one of these labels - no explanation, no punctuation:\n
     
     CAN_ANSWER - passages fully answer the question\n
     PARTIAL - passages touch on the topic but are incomplete\n
     NO_MATCH -passages are completely unrelated"""),
     ("human","Question:{question}\n\nPassages:\n{passages}\n\nLabel:")
    ])

    def __init__(self):
        
        llm = ChatGroq(model=settings.RELEVANCY_MODEL,
                       temperature=0,
                       max_tokens=settings.RELEVANCE_MAX_TOKENS,
                       api_key=settings.GROQ_API_KEY)
        
        self.chain = self.PROMPT_TEMPLATE | llm | StrOutputParser()

    def check(self,
              question: str,
              retriever: EnsembleRetriever,
              k: int = 5) -> str:
        """Retrieve top k chunks and classify relevance
        Returns: CAN_ANSWER,PARTIAL,NO_MATCH"""

        top_docs = retriever.invoke(question)
        
        if not top_docs:
            logger.warning("No documents retrieved - returning NO_MATCH")
            return "NO_MATCH"
        
        passages  = "\n\n".join(doc.page_content for doc in top_docs[:k])

        try:
            response = self.chain.invoke({"question":question,
                                          "passages":passages})
        
        except Exception as e:
            logger.error(f"RelevanceChecker chain error: {e}")
            return "NO_MATCH"
        
        label = response.strip().upper()
        valid = {"CAN_ANSWER","PARTIAL","NO_MATCH"}
        
        for v in valid: 
            if v in label:
                logger.info(f"Relevance classification: {v}")
                return v 
    
        logger.warning(f"Unrecognized label '{label}' - defaulting NO_MATCH")
        return "NO_MATCH"