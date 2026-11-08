from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

#from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document # .txt, .md

from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain_core.vectorstores import InMemoryVectorStore # vs

DATA_PATH = Path(__file__).resolve().parent.parent / "data" /"shop_handbook.md"
text = DATA_PATH.read_text(encoding="utf-8")

doc = Document(page_content=text, metadata={"source": "data/shop_handbook.md"})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, 
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
    )

chunks = text_splitter.split_documents([doc])

embedd = init_embeddings(
    model="nomic-embed-text:latest", 
    provider="ollama"
    )

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedd
    )

llm = init_chat_model(
    model="gemma4:31b-cloud", 
    model_provider="ollama",
    temperature = 0.1
)

query = "What is the handbook about?"

retrieved_docs = vector_store.similarity_search(query, k=3)     #similarity search me symantic search
context = "\n".join([doc.page_content for doc in retrieved_docs])       #Generator expression

messages = [
    {"role": "system", "content": "You are helpful assistan called Clause."},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
]

print(
    llm.invoke(f"Answer only from this context:\n{context}\n\nQuestion: {query}").text
    )


print("YAYYYYYYYYY")