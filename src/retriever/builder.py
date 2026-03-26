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

    
    def build_hybrid_retriever(self,docs: List[Document]) -> EnsembleRetriever:
        """Build a BM25 + Chroma Ensemble Retriever"""
        try:
            vector_store = Chroma.from_documents(documents=docs,
                                                 embedding=self.embeddings,
                                                 persist_directory=settings.CHROMA_DB_PATH,
                                                 collection_name=settings.CHROMA_COLLECTION_NAME)
            logger.info("Chroma vector store created")

            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = settings.VECTOR_SEARCH_K
            logger.info("BM-25 retriever created")

            vector_retriever = vector_store.as_retriever(search_kwargs={"k":settings.VECTOR_SEARCH_K})
            hybrid = EnsembleRetriever(retrievers=[bm25,vector_retriever],
                                       weights=settings.HYBRID_RETRIEVER_WEIGHTS)
            logger.info("Hybrid retriever ready")
            return hybrid

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever:{e}")
            raise
