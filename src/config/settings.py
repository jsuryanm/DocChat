from pydantic_settings import BaseSettings,SettingsConfigDict
from src.config.constants import MAX_FILE_SIZE,MAX_TOTAL_SIZE,ALLOWED_TYPES

class Settings(BaseSettings):
    OPENAI_API_KEY: str 

    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    VECTOR_SEARCH_K: int = 8 
    HYBRID_RETRIEVER_WEIGHTS: list = [0.5,0.5]

    RELEVANCY_MODEL: str = "gpt-5-nano"
    RESEARCH_MODEL: str = "gpt-5-mini"
    VERIFY_MODEL: str = "gpt-5-nano"

    EMBEDDINGS_MODEL: str = "text-embeddings-3-small"
    EMBEDDINGS_BATCH_SIZE: int = 100

    RESEARCH_MAX_TOKENS: int = 512 
    VERIFY_MAX_TOKENS: int = 256 
    RELEVANCE_MAX_TOKENS: int = 10

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_BATCH_SIZE: int = 16

    RERANKER_TOP_N: int = 3 
    VECTOR_SEARCH_K: int = 20

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8")
    
settings = Settings()