from pydantic_settings import BaseSettings
from src.config.constants import MAX_FILE_SIZE,MAX_TOTAL_SIZE,ALLOWED_TYPES

class Settings(BaseSettings):
    GROQ_API_KEY: str 

    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    VECTOR_SEARCH_K: int = 8 
    HYBRID_RETRIEVER_WEIGHTS: list = [0.5,0.5]

    RELEVANCY_MODEL: str = "llama-3.1-8b-instant"
    RESEARCH_MODEL: str = "llama-3.3-70b-versatile"
    VERIFY_MODEL: str = "llama-3.1-8b-instant"

    RESEARCH_MAX_TOKENS: int = 512 
    VERIFY_MAX_TOKENS: int = 256 
    RELEVANCE_MAX_TOKENS: int = 10

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    LOG_LEVEL: str = "INFO"

    class Config: 
        env_file = ".env"
        env_file_encoding = "utf-8"
    
settings = Settings()