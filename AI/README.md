## AIBA – AI Business Analyst (AI module)

The `AI/` package exposes the AIBA assistant as a FastAPI service plus a set of
local testing tools. It bundles six expert features that share the same session
memory:

- `requirement_clarifier`
- `use_case_generator`
- `feature_prioritization`
- `market_fit_analyzer`
- `stakeholder_insights`
- `ba_report_export`

Use the guide below if you are new to Git or Python.

## 1. Get the project onto your machine

1. Install Git for Windows from https://git-scm.com/download/win.
2. Open PowerShell and choose a folder for the project, e.g. `Documents`.
3. Clone your repository:

   ```powershell
   git clone https://github.com/pholoto/comp3080_ai_business_analyst_assistant.git
   cd comp3080_ai_business_analyst_assistant
   ```

4. If you already have the project, update it with `git pull` while inside the folder.

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of your PowerShell prompt.

## 3. Install Python dependencies

```powershell
pip install -r AI/requirements.txt
```

Re-run the command after pulling new changes so you stay in sync with the team.

## 4. (Optional) Connect to a real LLM

The assistant falls back to a deterministic stub when it cannot reach Ollama,
so you can prototype without any external setup. For real answers:

1. Download and install Ollama from https://ollama.com/download.
2. Start the service and pull a model:

   ```powershell
   OLLAMA_KEEP_ALIVE=1h ollama serve
   ollama run llama3.1
   ```

3. Point the app to another host or model if needed:

   ```powershell
   $Env:OLLAMA_BASE_URL = "http://localhost:11434"
   $Env:OLLAMA_MODEL = "llama3.1"
   ```

Hardware guidance for llama3.1: minimum 8-core CPU and 16 GB RAM, recommended
32 GB RAM or a GPU with at least 12 GB VRAM. Use the stub mode or a remote host
if your laptop cannot meet these numbers.

## 5. Run the FastAPI server

```powershell
python -m AI
```

The app listens on `http://127.0.0.1:8000`. Visit `http://127.0.0.1:8000/docs`
for the auto-generated Swagger UI. Stop the server with `Ctrl+C`.

### Key endpoints

- `POST /sessions` creates a new chat session.
- `POST /sessions/{id}/chat` runs one of the six features.
- `GET /sessions/{id}/transcript` returns the conversation history.
- `GET /sessions/{id}/state` shows accumulated artefacts and attachments.
- `POST /sessions/{id}/attachments` uploads PDF, DOCX, or TXT files.
- `GET /strategies` lists chunking/indexing options you can switch to with the
  `/chunking` and `/indexing` endpoints.
- `POST /sessions/{id}/search` performs retrieval over the current index.
- `POST /sessions/{id}/evaluation` computes Precision@k, Recall@k, MRR, and NDCG@k.

The `ba_report_export` feature writes a DOCX report using the VinUni template in
`back_end/templates/`, saving the output under `reports/`.

## 6. Benchmark chunking and indexing offline

1. Drop sample PDFs, DOCXs, or TXTs into `AI/sample_documents/`.
2. From the project root run:

   ```powershell
   python -m AI.evaluate_chunking_indexing --documents AI/sample_documents
   ```

   The CLI prints Precision@k, Recall@k, MRR, NDCG@k, and latency for every
   chunking/indexing combination.

3. Add flags to focus your test:
   - `--queries my_queries.json` to provide custom evaluation prompts.
   - `--chunkers fixed semantic` or `--indexers faiss` to narrow the matrix.
   - `--top-k 8` to change the retrieval depth.
   - `--save-json results.json` to capture the metrics.

## 7. Try the assistant without HTTP

Use the console harness to script or manually drive the assistant features:

```powershell
python -m AI.test_assistant_cli --attachments AI/sample_documents
```

- Enter `list` to see feature keys, then choose one (for example `stakeholder_insights`).
- Provide the message when prompted and review the JSON payload the feature returns.
- Type `state` to inspect stored requirements, assumptions, and attachment info.
- Supply `--script tests/conversation.json` to replay a canned dialogue where the
  JSON file contains `{"turns": [{"feature": "requirement_clarifier", "message": "..."}]}`.

The CLI reuses the same session manager as the API, so every feature shares
context, attachments, and state changes.

## Retrieval strategies at a glance

- Chunking:
  - `all_in_one` keeps each document as a single chunk (baseline).
  - `fixed` slices documents into 1,200 character windows with 200 character overlap.
  - `semantic` aligns chunks with detected headings and paragraphs.
- Indexing:
  - `none` runs a linear substring search only.
  - `faiss` simulates cosine similarity over lightweight embeddings in memory.
  - `llama_index` builds a three-level (document, section, chunk) scorer.

## Sample documents

Store any fake artefacts you want to experiment with under `AI/sample_documents/`.
Both the FastAPI uploads and the local CLIs read from the same folder structure.
