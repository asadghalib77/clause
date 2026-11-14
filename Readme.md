# Clause

Clause is a local, PDF-based question-answering project. Its Python backend uses retrieval-augmented generation (RAG) to find relevant passages in a shop handbook and ask an Ollama model to answer from those passages.

The repository also contains a Next.js frontend scaffold. It currently displays the starter page and is not connected to the question-answering API. Use the API documentation or the command-line interface to ask questions.

## How it works

1. Load the handbook PDF with PyPDFLoader.
2. Split its pages into overlapping text chunks.
3. Generate embeddings using Ollama's `nomic-embed-text:latest` model.
4. Store the vectors in a local Qdrant collection.
5. Retrieve relevant chunks for a question and pass them to `gemma3:1b` through Ollama.
6. Return an answer alongside source passages, page metadata, and similarity scores.

By default, retrieval returns up to three chunks with a similarity threshold of `0.5`. If no chunks meet that threshold, the API returns a fixed out-of-context response. The model is instructed to answer only from the supplied context; generated answers should still be checked against the sources.

## Project structure

```text
clause/
├── api/
│   ├── app/
│   │   ├── core/config.py       # Settings, embeddings, and Qdrant client
│   │   ├── services/ingestion.py # PDF loading and indexing
│   │   ├── services/chat.py     # Retrieval and answer generation
│   │   ├── main.py             # FastAPI application
│   │   ├── cli.py              # Index PDF and ask one question
│   │   └── rag.py              # Earlier standalone experiment
│   ├── evals/                  # Sample questions and evaluation script
│   ├── files/shop_handbook.pdf # Included handbook
│   ├── vector_stores/          # Existing stored vector data
│   └── pyproject.toml          # Backend dependencies
├── web/                       # Next.js, React, TypeScript, and Tailwind scaffold
├── pyproject.toml             # Separate root Python project
└── uv.lock
```

## Prerequisites

- Git
- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for Python environments and dependencies
- [Ollama](https://ollama.com/) installed and running locally
- Node.js and npm if you want to run the frontend scaffold

The backend uses embedded Qdrant storage, so a separate Qdrant server is not required.

## Clone from GitHub

```sh
git clone https://github.com/asadghalib77/clause.git
cd clause
```

## Run the backend

### 1. Download the Ollama models

```sh
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

Keep Ollama running while indexing documents and asking questions. If it is not already running, start it in another terminal with `ollama serve`.

### 2. Install the backend dependencies

Run these commands from the repository root:

```sh
cd api
uv sync
uv pip install pydantic-settings sqlalchemy
```

The current backend imports `pydantic-settings` and `sqlalchemy` without declaring them directly in `api/pyproject.toml`. The additional install covers those imports without editing the project manifest. The commands below use `--no-sync` to retain these environment additions.

Install from `api/`, since the root Python project does not declare the full backend dependencies.

### 3. Configure the handbook and storage

Create a file named `.env` inside `api/` with:

```dotenv
RAG_DATA_PATH=files/shop_handbook.pdf
RAG_CHAT_PROVIDER=ollama
RAG_CHAT_MODEL=gemma3:1b
RAG_CHAT_TEMPERATURE=0.1
RAG_EMBED_PROVIDER=ollama
RAG_EMBED_MODEL=nomic-embed-text:latest
RAG_EMBED_DIM=768
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=100
RAG_TOP_K=3
RAG_THRESHOLD_SCORE=0.5
RAG_COLLECTION=shop_handbook
RAG_QDRANT_DATA=./qdrant_data
```

The PDF path setting is required for the included handbook: the code's default points to `data/shop_handbook.pdf`, while the repository stores it under `files/`. Relative PDF and storage paths resolve from the working directory, so run backend commands from `api/`.

Settings are loaded from `api/.env` and use the `RAG_` prefix. The current Qdrant client uses local storage; `RAG_QDRANT_URL` is not used by that client. The existing `api/vector_stores/` directory is separate from the default `api/qdrant_data/` location.

### 4. Index the handbook

From `api/`, run:

```sh
uv run --no-sync python -m app.services.ingestion
```

This loads the PDF, generates embeddings, and creates the `shop_handbook` collection. Re-running ingestion deletes and rebuilds the configured collection. Stop the API or other processes using the same local Qdrant directory before indexing.

### 5. Start the API

```sh
uv run --no-sync uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the API interactively. Indexing must finish before API startup because the application requires the collection to exist.

## Ask a question

In the interactive API docs, expand `POST /ask`, select **Try it out**, and submit:

```json
{
  "question": "What is the handbook about?"
}
```

The response contains `answer` and `sources`. Each source includes its file path, page metadata, similarity score, text content, and other document metadata. Page indices from the PDF loader are zero-based.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Returns `{"status":"ok"}` once the application is running |
| `POST /ask` | Answers a question using the indexed handbook |
| `GET /docs` | Interactive API documentation |

Questions are trimmed and must contain between 1 and 2,000 characters.

### Command-line alternative

With the API stopped, run this from `api/`:

```sh
uv run --no-sync python -m app.cli
```

This rebuilds the index, prompts for one question, prints the answer, and exits. It does not run a continuous chat session or print source details.

## Run the frontend scaffold

In a separate terminal, start from the repository root:

```sh
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The current page is the Next.js starter screen; a handbook chat interface and API integration still need to be implemented.

Other frontend commands, run from `web/`:

```sh
npm run lint
npm run build
npm start
```

Run `npm run build` before `npm start`.

## Evaluate answers

With the handbook indexed, Ollama running, and the API stopped, run from `api/`:

```sh
uv run --no-sync python -m evals.run
```

The script reads `evals/golden.jsonl` and reports retrieval hit rate, whether required facts appear in retrieved chunks, answer correctness, refusal rate, and staff-only information leakage. These checks use keyword matching rather than a full semantic assessment. Its refusal phrases also do not include the service's fixed out-of-context response, so inspect individual failures when interpreting the results.

## Troubleshooting and current limits

- **PDF not found:** Set `RAG_DATA_PATH=files/shop_handbook.pdf` in `api/.env` and run from `api/`.
- **Collection not found:** Run ingestion before starting the API, with the same collection and storage settings.
- **Ollama connection or model errors:** Confirm Ollama is running and both configured models have been downloaded.
- **Missing Python modules:** Install dependencies from `api/`, including the extra imports described above.
- **Qdrant storage already in use:** Stop other processes accessing the same local storage directory, including the API before running ingestion, CLI, or evaluation commands.
- **Changed embedding model:** Match `RAG_EMBED_DIM` to the model's vector size and rebuild the index.
- **Current functionality:** Responses are single-turn. Streaming, conversation history, and staff-only content filtering are not implemented. The evaluation includes leakage checks, but those checks do not enforce access restrictions.

Use `app.main`, `app.cli`, and `app.services.ingestion` for the documented workflows. `app/rag.py` is an earlier experiment that references a Markdown file absent from this repository.
