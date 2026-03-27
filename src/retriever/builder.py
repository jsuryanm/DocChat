from typing import List
from pathlib import Path
import asyncio

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder # (reranker)
from langchain_core.documents import Document

from src.config.settings import settings
from src.custom_logger.logger import logger


class EmptyRetriever:
    """Safe fallback retriever"""

    def invoke(self,query: str):
        return []

class RetrieverBuilder: 
    
    def __init__(self,
                 embeddings=None,
                 reranker=None):
        
        self._embeddings = embeddings
        self._reranker = reranker
        
        logger.info("RetrieverBuilder initialized")

    async def _get_embeddings(self):
        
        if self._embeddings is None:
            logger.info("Loading embeddings model")        
            self._embeddings = OpenAIEmbeddings(model=settings.EMBEDDINGS_MODEL,
                                                api_key=settings.OPENAI_API_KEY,
                                                chunk_size=settings.EMBEDDINGS_BATCH_SIZE)
        return self._embeddings
     
    async def _get_reranker(self):
        if self._reranker is None:
            logger.info("Loading reranker model")
            
            self._reranker = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)  
        return self._reranker

    
    async def build_hybrid_retriever(self,
                                     docs: List[Document],
                                     persist: bool = True):
        """Build a BM25 + Chroma Ensemble Retriever"""

        if not docs:
            logger.warning("No documents provided using Empty Retriever")
            return EmptyRetriever()

        try:
            embeddings = await self._get_embeddings() 
            
            if persist and Path(settings.CHROMA_DB_PATH).exists():
                logger.info("Loading existing CHROMADB")
                vector_store = await asyncio.to_thread(Chroma.from_documents,
                                                       documents=docs,
                                                       embedding_function=embeddings,
                                                       persist_directory=settings.CHROMA_DB_PATH,
                                                       collection_name=settings.CHROMA_COLLECTION_NAME)
                
            
            else:
                logger.info("Creating new CHROMADB")
                vector_store = await asyncio.to_thread(Chroma.from_documents,
                                                       documents=docs,
                                                       embedding_function=embeddings,
                                                       persist_directory=settings.CHROMA_DB_PATH,
                                                       collection_name=settings.CHROMA_COLLECTION_NAME)


            bm25 = await asyncio.to_thread(BM25Retriever.from_documents,docs)
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
    
    async def rerank(self,
                     query: str,
                     docs: List[Document]) -> List[Document]:
        
        if not docs:
            return []
        try:
            reranker = await self._get_reranker()

            logger.info(f"BM25 docs: {len(docs)}")
            logger.info(f"Vector search k:{settings.VECTOR_SEARCH_K}")

            batch_size = settings.RERANK_BATCH_SIZE

            doc_texts = [d.page_content for d in docs]

            batches = [
                doc_texts[i:i+batch_size]
                for i in range(0,len(doc_texts),batch_size)
            ]

            tasks = [
                asyncio.to_thread(reranker.score,query,batch)
                for batch in batches
            ]

            results = await asyncio.gather(*tasks)

            scores = [s for batch in results for s in batch]
                        
            if len(scores) != len(docs):
                logger.warning("Rerank score mismatch")
            
                return docs[:settings.RERANKER_TOP_N]

            ranked = sorted(zip(docs,scores),
                            key=lambda x:x[1],
                            reverse=True)
        
            return [d for d,_ in ranked[:settings.RERANKER_TOP_N]]
        
        except Exception as e:
            logger.error(f"Rerank failed: {e}")

            return docs[:settings.RERANKER_TOP_N]