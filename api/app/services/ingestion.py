#Indexing
#1: Load, 2: Chunks, 3: Embed, 4: Store

# from pypdf import PdfReader
from langchain_core.documents import Document

import re   #add this in v6

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

from app.core.config import get_settings, get_embeddings, get_qdrant_client

STAFF_ONLY = re.compile(
    r"^\s*12\.\s*Internal:.*?\(staff only\).*?(?=^\s*Clause Shop\s*\||\Z)",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
)

def load(path: Path) -> list[Document]:
    docs = PyPDFLoader(file_path=path).load()

    for doc in docs:
        doc.page_content = STAFF_ONLY.sub("", doc.page_content)

    return docs
    

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
            vectors_config={"dense": VectorParams(
                size=settings.embed_dim,
                distance=Distance.COSINE
            )},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection,
            embedding=embed,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode = RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse"
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