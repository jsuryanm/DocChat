from pydantic_settings import BaseSettings, SettingsConfigDict
from src.config.constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str

    MCP_ENABLED_SERVERS: str = "*"

    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    VECTOR_SEARCH_K: int = 8
    HYBRID_RETRIEVER_WEIGHTS: list = [0.5, 0.5]

    RELEVANCY_MODEL: str = "gpt-5-nano"
    RESEARCH_MODEL: str = "gpt-5-mini"
    VERIFY_MODEL: str = "gpt-5-nano"

    EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    EMBEDDINGS_BATCH_SIZE: int = 100

    RESEARCH_MAX_TOKENS: int = 1600

    # FIX: raised from 400 → 1000.
    # gpt-5-nano (and any reasoning-model variant) allocates reasoning tokens
    # against max_tokens BEFORE producing the structured output field.
    # 400 tokens was reliably exhausted before the JSON was emitted, causing
    # LengthFinishReasonError on every verify call and triggering a spurious
    # retry loop that discarded a valid answer.
    VERIFY_MAX_TOKENS: int = 1500

    # FIX: raised from 200 → 500.
    # Same root cause as VERIFY_MAX_TOKENS: the reasoning model consumed all
    # 200 tokens for chain-of-thought and had none left for the label field,
    # which caused the retry-pass relevance check to return NO_MATCH and
    # finalize with the "no relevant docs" fallback message.
    RELEVANCE_MAX_TOKENS: int = 500

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_BATCH_SIZE: int = 16

    RERANKER_TOP_N: int = 3

    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    REMOTE_AGENT_URL: str = ""
    A2A_HOST: str = "0.0.0.0"
    A2A_PORT: int = 9000

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()