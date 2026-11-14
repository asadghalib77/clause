from app.services.ingestion import ingest
from app.services.chat import RAGService

def main() -> None:
    rag_service = RAGService()
    try:
        query = input("You:").strip()
        result = rag_service.ask(query)
        print(f"AI Bot: {result.answer}")
        if result.sources:
            pass
        
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rag_service.close()
    pass

if __name__ == "__main__":
    print(f"Ingested {ingest()} chunks")
    main()
    
    
    
    
    
    
    
    
    
    
    
    
# from pathlib import Path

# #from langchain.chat_models import init_chat_model
# #from langchain.embeddings import init_embeddings

# #from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.documents import Document, PyPDFLoader

# #from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
# #from langchain_core.vectorstores import InMemoryVectorStore # vs

# DATA_PATH = Path(__file__).resolve().parent.parent / "data" /"shop_handbook.md"
# text = DATA_PATH.read_text(encoding="utf-8")

# doc = Document(page_content=text, metadata={"source": "data/shop_handbook.md"})