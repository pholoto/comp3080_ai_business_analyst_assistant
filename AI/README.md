Perfect — let’s simplify the README so it assumes the user has enough disk space and memory, and avoids the whole E: drive workaround. Here’s the streamlined version with clear step‑by‑step instructions:

---

# AIBA – AI Business Analyst (AI module)

The `AI/` package exposes the AIBA assistant as a FastAPI service plus a set of local testing tools. It bundles six expert features that share the same session memory:

- `requirement_clarifier`  
- `use_case_generator`  
- `feature_prioritization`  
- `market_fit_analyzer`  
- `stakeholder_insights`  
- `ba_report_export`

---

## 1. Get the project onto your machine

1. **Install Git**: Download Git for Windows from [git-scm.com](https://git-scm.com/download/win).  
2. **Clone the repository**:
   ```powershell
   cd $HOME\Documents
   git clone https://github.com/pholoto/comp3080_ai_business_analyst_assistant.git
   cd comp3080_ai_business_analyst_assistant
   ```
3. **Update if already cloned**:
   ```powershell
   git pull
   ```

---

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your PowerShell prompt.

---

## 3. Install Python dependencies

```powershell
pip install -r AI/requirements.txt
```

Re-run this after pulling new changes to stay in sync.

---

## 4. Connect to a real LLM (optional)

By default, the assistant uses a deterministic stub so you can prototype without setup. For real answers, connect to Ollama:

1. **Install Ollama**: Download from [ollama.com/download](https://ollama.com/download).  
2. **Start the service and pull a model**:
   ```powershell
   OLLAMA_KEEP_ALIVE=1h ollama serve
   ollama pull llama3
   ollama run llama3
   ```
   > Tip: `gemma:2b` or `phi` are lightweight models that run well on most laptops.  
3. **Point the app to Ollama**:
   ```powershell
   $Env:OLLAMA_BASE_URL = "http://localhost:11434"
   $Env:OLLAMA_MODEL = "llama3"
   ```

---

## 5. Run the FastAPI server

```powershell
python -m AI
```

- The app listens on `http://127.0.0.1:8000`.  
- Visit `http://127.0.0.1:8000/docs` for Swagger UI.  
- Stop the server with `Ctrl+C`.

### Key endpoints

- `POST /sessions` → create a new chat session  
# AIBA – AI Business Analyst (AI module)

The `AI/` package exposes the AIBA assistant as a FastAPI service plus a set of local testing tools. It bundles six expert features that share the same session memory:

- `requirement_clarifier`
- `use_case_generator`
- `feature_prioritization`
- `market_fit_analyzer`
- `stakeholder_insights`
- `ba_report_export`

This README focuses on a minimal setup and how to test chunking and the AI output. The assistant uses the public MLVoca text-generation API by default, so no local LLM install is required.

## 1. Quick setup

1. Install Git (if needed) and clone the repo:

```powershell
cd $HOME\Documents
git clone https://github.com/pholoto/comp3080_ai_business_analyst_assistant.git
cd comp3080_ai_business_analyst_assistant
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install Python dependencies:

```powershell
pip install -r AI/requirements.txt
```

Notes:
- By default the app falls back to a deterministic stub client when an external LLM is not reachable. The repository is configured to use the public MLVoca API (`https://mlvoca.com/api/generate`) so you do not need to run or configure a local LLM server.
- If you need to override the default, set `MLVOCA_BASE_URL` and/or `MLVOCA_MODEL` in your environment.

## 2. Run the FastAPI server

```powershell
python -m AI
```

- The app listens on `http://127.0.0.1:8000` by default.
- Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

Key endpoints:

- `POST /sessions` → create a new chat session
- `POST /sessions/{id}/chat` → run one of the features (supply `feature` key in payload)
- `GET /sessions/{id}/transcript` → get conversation history
- `GET /sessions/{id}/state` → view accumulated artefacts and attachments
- `POST /sessions/{id}/attachments` → upload PDF, DOCX, or TXT files
- `GET /strategies` → list chunking/indexing options
- `POST /sessions/{id}/search` → retrieval over the current index
- `POST /sessions/{id}/evaluation` → compute Precision@k, Recall@k, MRR, NDCG@k

The `ba_report_export` feature writes a DOCX report using the VinUni template in `back_end/templates/` and saves output under `reports/`.

## 3. Test chunking and indexing (offline benchmark)

1. Place sample PDFs, DOCXs or TXTs into `AI/sample_documents/`.
2. Run the evaluation harness:

```powershell
python -m AI.evaluate_chunking_indexing --documents AI/sample_documents
```

This prints Precision@k, Recall@k, MRR, NDCG@k and latency for each chunking/indexing combination. Useful flags:

- `--queries my_queries.json` → use custom evaluation prompts
- `--chunkers fixed semantic` → evaluate a subset of chunkers
- `--indexers faiss` → evaluate only the FAISS indexer
- `--top-k 8` → change retrieval depth
- `--save-json results.json` → save metrics to a file

## 4. Test AI output and features (local CLI)

You can exercise features without running the HTTP server using the console harness. This is useful for quickly checking how features format their JSON output and ensuring the assistant returns clean structured data.

```powershell
python -m AI.test_assistant_cli --attachments AI/sample_documents
```

- Enter `list` to show available feature keys, then type a feature key (for example `requirement_clarifier`) and provide a message when prompted.
- The CLI returns the feature output. The assistant code attempts to strip internal thinking traces and recover structured JSON where requested.
- Use `state` to inspect stored requirements, assumptions and attachments in the session.

For scripted runs, pass `--script path/to/script.json` where `script.json` contains a `turns` list of `{"feature":"...","message":"..."}` objects.

## 5. Retrieval strategies at a glance

- Chunking:
  - `all_in_one` → single chunk per document (baseline)
  - `fixed` → fixed-size windows with overlap
  - `semantic` → chunk according to headings and paragraph boundaries

- Indexing:
  - `none` → linear substring search
  - `faiss` → approximate nearest neighbours over embeddings
  - `llama_index` → document/section/chunk scoring layers

## 6. Sample documents

Store example artefacts under `AI/sample_documents/`. Both the FastAPI endpoints and the local CLI read from the same folder.

---

If you want, I can also add a one-line quick-start checklist or an example `curl` invocation to call a feature via the HTTP API.