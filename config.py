from functools import lru_cache
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    # Anthropic — used by the LLM (chatbot.py)
    anthropic_api_key: str

    # Voyage AI — used for embeddings (vector_store.py)
    voyage_api_key: str

    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.2
    max_tokens: int = 1024

    embedding_model: str = "voyage-4-large"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    history_window_size: int = 10

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> _Settings:
    return _Settings()
