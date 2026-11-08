from fastapi import FastAPI, Request
from pydantic import BaseModel, StringConstraints
from typing import Annotated, AsyncIterator
from contextlib import asynccontextmanager
from app.services.chat import RAGService, Answer, Source

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.rag = RAGService()
    yield
    app.state.rag.close()

app = FastAPI(title = "Clause APIs", lifespan=lifespan)


class AskQuestion(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]

@app.get(path="/")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post(path="/ask", response_model=Answer)
def ask_query(body: AskQuestion, req: Request) -> None:
    return req.app.state.rag.ask(body.question)