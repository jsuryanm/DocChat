from typing import List 

from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_core.documents import Document 

from src.config.settings import settings 
from src.custom_logger.logger import logger 

class RetrieverBuilder: 
    def __init__(self):
        logger.info("Loading HuggingFaceEmbeddings model all-MiniLM-L6-v2")

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                                model_kwargs={"device":"cpu"},
                                                encode_kwargs={"normalize_embeddings":True})
        
        logger.info("Embeddings model ready")

    
    def build_hybrid_retrievers(self,docs: List[Document]) -> EnsembleRetriever:
        """Build a BM25 + Chroma Ensemble Retriever"""
        pass 
