#Indexing
#1: Load, 2: Chunks, 3: Embed, 4: Store

# from pypdf import PdfReader
from langchain_core.documents import Document

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings, get_embeddings, get_qdrant_client

def load(path: Path) -> list[Document]:
    return PyPDFLoader(file_path=path).load()     #PyPDF List of documents bnata hai

def ingest() -> int:
    settings = get_settings()

    #1: Load
    file = Path(settings.data_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    docs = load(path=file)

    #2: Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
        separators = ["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    #3: Embed
    embed = get_embeddings(settings)

    #4: Store
    client = get_qdrant_client(settings)
        # path= settings.qdrant_path
        # )

    try:
        if client.collection_exists(collection_name=settings.collection):
            client.delete_collection(collection_name=settings.collection)

        client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(
                size=settings.embed_dim,
                distance=Distance.COSINE
            )
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection,
            embedding=embed
        )

        vector_store.add_documents(chunks)
        print(f"Successfully indexed {len(chunks)} chunks.")

        count = client.count(
            collection_name = settings.collection,
            exact = True
        ).count
        
    # except:
    #     pass
    finally: 
        client.close()

    return len(chunks)

if __name__ == "__main__":
    print(ingest(), "chunk indexed!")