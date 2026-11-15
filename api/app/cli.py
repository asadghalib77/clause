from app.services.ingestion import ingest
from app.services.chat import RAGService

def main() -> None:
    rag_service = RAGService()
    try:
        while True:
            query = input("You:").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "bye"]:
                break
            result = rag_service.ask(query)
            print(f"AI Bot: {result.answer}")
            if result.sources:
                print("Sources:")
                for source in result.sources:
                    print(f"- {source.source} (page: {source.page}, score: {source.score})")
        
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