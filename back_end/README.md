# Backend RAG Service

This folder hosts the Retrieval-Augmented Generation (RAG) backend for the AI Business Analyst Assistant. The service ingests user documents, indexes them with FAISS, and exposes FastAPI endpoints for two agents:

- **Ideation Agent** – generates new solution ideas grounded in project briefs, reports, and meeting notes.
- **Progress Agent** – analyses logs, timelines, and task updates to surface progress insights, deadline risks, and recovery suggestions.

## Capabilities
- Per-user document segregation under `data/<user_id>` with metadata tracked in `doc_index.json`.
- LangChain-based chunking (512 token chunks, 128 token overlap).
- Sentence-transformers embeddings stored in FAISS indices (`vector_store/<user_id>`).
- Retrieval layer with optional tag filters and ranked similarity scores.
- LLM generation layer wired to `AI.llm.client` with graceful stub fallback.
- REST API for document ingestion, ideation, and progress analysis.

## Requirements

Install dependencies from `back_end/requirements.txt` (Python 3.10+ recommended):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r back_end/requirements.txt
```

> The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. The LLM layer defaults to the Ollama client defined in `AI.llm.client`, but safely falls back to a stub when the model is unavailable.

## Running the API

```powershell
uvicorn back_end.app:app --reload
```

Key endpoints:

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Service heartbeat |
| `GET`  | `/users/{user_id}/documents` | List stored documents for a user |
| `POST` | `/users/{user_id}/documents` | Upload and index a document (`multipart/form-data` with `file` and optional `tags`) |
| `POST` | `/users/{user_id}/ideation` | Generate 5-10 ideas for a topic |
| `POST` | `/users/{user_id}/progress` | Summarise progress status as of a date |

### Sample requests

**Upload a document**

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://localhost:8000/users/user1/documents `
	-Form @{ file = Get-Item .\docs\project_brief.pdf; tags = "ideation,brief" }
```

**Run Ideation Agent**

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://localhost:8000/users/user1/ideation `
	-Body (@{ topic = "New fintech revenue streams" } | ConvertTo-Json) `
	-ContentType "application/json"
```

**Run Progress Agent**

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://localhost:8000/users/user1/progress `
	-Body (@{ reference_date = "2025-03-31" } | ConvertTo-Json) `
	-ContentType "application/json"
```

## Implementation Notes

- Vector search uses cosine similarity via a `faiss.IndexFlatIP` index; embeddings are pre-normalised.
- Tags help scope retrieval (e.g., `ideation`, `progress`, `timeline`). When omitted, each agent fallbacks to sensible defaults.
- The generator enforces the rule “Only answer based on the documents.” If context is insufficient, it responds with `I could not find enough information in the documents to answer.`
- Document duplication is prevented via SHA-256 checksums.
- All directories are auto-created: `data/`, `vector_store/`, and metadata files.

## Extending

- Add new agent logic under `rag/agents/` and register it in `back_end/app.py`.
- Swap the embedding model by updating `rag/config.py` (ensure compatible vector dimensions).
- Connect to alternative LLM providers by injecting a custom client into `ResponseGenerator`.

The system is designed for composability—every stage (store, splitter, embeddings, vector DB, retriever, generator, ranker) is modular and easily replaceable.
