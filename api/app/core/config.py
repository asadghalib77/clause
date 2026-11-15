from pathlib import Path
from functools import  lru_cache
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

#api/env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE)  # Load environment variables from .env file
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix = "RAG_",
        env_file=ENV_FILE,
        extra = "ignore",
        env_file_encoding="utf-8"
    )

    #APP
    app_name: str = "Clause"
    data_path: str = "data/shop_handbook.pdf"



    # Chat
    chat_model: str = "gemini-3.1-flash-lite"
    chat_provider: str = "google_genai"
    chat_temperature: float = 0

    chat_api_key: SecretStr = Field(
        validation_alias="GOOGLE_API_KEY"
    )

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
    system_prompt: str = (
        "You are the Clause shop customer assistant. "
        "Answer using ONLY the handbook text you are given. "
        "Find the one sentence that answers the question and copy its numbers exactly. "
        "Answer in one or two short sentences. Never mention 'the context' or 'the handbook text'. "
        "If the handbook text does not answer the question, reply exactly: "
        "\"I don't know.\""
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

    