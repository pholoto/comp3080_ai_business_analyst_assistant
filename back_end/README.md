# Backend RAG Service

This folder hosts the Retrieval-Augmented Generation (RAG) backend for the AI Business Analyst Assistant. The implementation now covers the complete reference pipeline: document ingestion → chunking → embeddings → FAISS index → retrieval & ranking → prompt templating → LLM answer generation → `rag_pipeline` orchestration → FastAPI endpoints and tests.

## Architecture overview

| Stage | Description | Key files |
| --- | --- | --- |
| Ingestion & metadata | Per-user storage, checksum dedupe, txt/pdf/docx/md parsing | `rag/document_store.py`, `back_end/app.py` (`POST /documents`), `rag/pipeline.py` (`ingest_document`)
| Chunking | Recursive splitter with overlap, char ranges, provenance fields | `rag/text_splitter.py`
| Embeddings | Provider abstraction (SentenceTransformers default, mockable) | `rag/embedding_generator.py`
| Vector index | Persistent FAISS per user with metadata blob | `rag/vector_store.py`
| Retrieval & ranking | Score thresholding, optional recency boost, tag filters, similarity ranking previews | `rag/retrieval.py`, `rag/ranking.py`
| Prompting | Context budget enforcement, reusable templates, guardrails | `rag/prompts.py`
| LLM generation | `ResponseGenerator` (wrapping `AI.llm.client`), citations in answer | `rag/generation.py`, `rag/pipeline.py`
| RAG orchestration | `RagPipeline`, CLI demo, `/users/{id}/rag` endpoint | `rag/pipeline.py`, `back_end/app.py`
| Agents | Ideation + Progress flows built on retriever/generator | `rag/agents/*.py`
| Tests | Unit + integration coverage for chunking, embeddings, vector store, retriever, prompts, pipeline | `back_end/tests/`

This mapping aligns with the provided architecture diagram: ingestion (DocumentStore) feeds chunking (TextSplitter), embeddings (EmbeddingGenerator) populate the vector store (FaissVectorStore), retrieval (ContextRetriever) fuels prompting (`rag/prompts.py`) and LLM generation, all orchestrated by `RagPipeline` and exposed through FastAPI/CLI.

## Requirements

Install dependencies from `back_end/requirements.txt` (Python 3.10+ recommended):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r back_end/requirements.txt
```

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` (set `RAG_EMBED_MODEL` to override). The LLM layer defers to `AI.llm.client` (MLVoca API with DeepSeek R1 by default) and falls back to a deterministic stub for local testing.

## Running the FastAPI service

```powershell
uvicorn back_end.app:app --reload
```

Key endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service heartbeat |
| `GET` | `/users/{user_id}/documents` | List stored documents + metadata |
| `POST` | `/users/{user_id}/documents` | Upload + index a document (`multipart/form-data`) |
| `POST` | `/users/{user_id}/ideation` | Run the Ideation agent |
| `POST` | `/users/{user_id}/progress` | Run the Progress agent |
| `POST` | `/users/{user_id}/rag` | General-purpose RAG answer with citations (JSON output) |

### Sample PowerShell calls

Upload + index:

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://localhost:8000/users/demo/documents `
	-Form @{ file = Get-Item .\docs\project_brief.pdf; tags = "brief,ideation" }
```

Ask a question via the unified pipeline:

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://localhost:8000/users/demo/rag `
	-Body (@{
			question = "What risks were recorded last sprint?"
			top_k = 6
			score_threshold = 0.2
		} | ConvertTo-Json) `
	-ContentType "application/json"
```

Response schema:

```json
{
	"answer": "...",
	"citations": [
		{"source": "status_report.docx", "chunk_id": "doc_0003", "chunk_index": 3, "score": 0.84}
	],
	"used_context": [...],
	"prompt_template": "...full prompt...",
	"timings_ms": {"retrieval": 43.1, "total": 312.8}
}
```

## Running the CLI demo / smoke test

The pipeline exposes a CLI wrapper that can ingest a folder and answer a query in one shot (useful for demos or benchmarking latency).

```powershell
python -m back_end.rag.pipeline "List current scope risks" --user-id demo --ingest-folder AI/sample_documents --top-k 6
```

Output is a JSON blob matching the `/rag` endpoint, including citations and timings.

## Rebuilding / extending the pipeline

- **Rebuild the vector index**: delete the per-user folder under `back_end/vector_store/<user_id>` and re-run ingestion, or call the CLI with `--ingest-folder` to rehydrate everything.
- **Swap embedding providers**: implement `EmbeddingProvider` (see `rag/embedding_generator.py`) and pass it into `RagPipeline` or `EmbeddingGenerator`. Tests stub this interface, so new providers should add coverage.
- **Add new retrieval heuristics**: extend `ContextRetriever` to adjust thresholds, or surface additional score controls via the `/rag` payload.
- **New agents**: add a class under `rag/agents/`, inject the shared retriever/generator via `RagDependencies`, and expose the endpoint in `back_end/app.py`.

## Tests & validation

Run the unit + integration suite (covers chunking, embeddings, vector store, retrieval, prompts, and the full pipeline):

```powershell
python -m pytest back_end/tests -q
```

The pipeline integration test ingests synthetic data, executes `RagPipeline.run`, and asserts citations plus latency metrics—serving as the requested smoke test.

## Developer tips

- Adjust `max_context_chars`, `score_threshold`, or `recency_boost_*` directly in `rag/config.py` (or pass a custom `RagConfig`) to tune retrieval behaviour for different deployments.
- Logs around ingestion, retrieval, and generation can be enabled by configuring the app’s logging level (FastAPI/uvicorn). Add additional context logging where needed.
- When FAISS or sentence-transformers are unavailable (e.g., CI), the code raises explicit TODO-style errors so dependencies are clear.

With these pieces the backend now delivers a complete, test-backed RAG pipeline that mirrors the architecture requirements and exposes a turnkey `/rag` endpoint for the AI Business Analyst Assistant.
