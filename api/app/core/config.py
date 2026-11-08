from pathlib import Path
from functools import  lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

#api/env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix = "RAG+",
        env_file=ENV_FILE,
        extra = "ignore",
        env_file_encoding="utf-8"
    )

    #APP
    app_name: str = "Clause"
    data_path: str = "data/shop_handbook.pdf"

    #Chat
    chat_model: str = "gemma4:31b-cloud"
    chat_provider: str = "ollama"
    chat_temperature: float = 0.1

    #Embeding
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    #Indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    #Retrieval
    top_k: int = 3
    threshold_score: float = 0.5
    
    #Qdrant
    collection: str = "shop_handbook"  #table
    qdrant_data: str = "./qdrant_data" #vectorStore
    qdrant_url: str | None = None

    #Prompts
    system: str = (
        "You are helpful Clause shop assisstant."
        "Answer only from the context provided."
    )

@lru_cache
def get_settings():
    return Settings()

def get_embeddings(settings: Settings) -> Embeddings:
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider
    )

def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        path=settings.qdrant_data
    )

    