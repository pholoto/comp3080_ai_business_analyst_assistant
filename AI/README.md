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
- `POST /sessions/{id}/chat` → run one of the six features  
- `GET /sessions/{id}/transcript` → get conversation history  
- `GET /sessions/{id}/state` → view accumulated artefacts and attachments  
- `POST /sessions/{id}/attachments` → upload PDF, DOCX, or TXT files  
- `GET /strategies` → list chunking/indexing options  
- `POST /sessions/{id}/search` → retrieval over the current index  
- `POST /sessions/{id}/evaluation` → compute Precision@k, Recall@k, MRR, NDCG@k  

The `ba_report_export` feature writes a DOCX report using the VinUni template in `back_end/templates/`, saving the output under `reports/`.

---

## 6. Benchmark chunking and indexing offline

1. Place sample PDFs, DOCXs, or TXTs into `AI/sample_documents/`.  
2. Run:
   ```powershell
   python -m AI.evaluate_chunking_indexing --documents AI/sample_documents
   ```
   Prints Precision@k, Recall@k, MRR, NDCG@k, and latency for each chunking/indexing combination.  
3. Add flags to customize:
   - `--queries my_queries.json` → custom evaluation prompts  
   - `--chunkers fixed semantic` or `--indexers faiss` → narrow the matrix  
   - `--top-k 8` → change retrieval depth  
   - `--save-json results.json` → save metrics  

---

## 7. Try the assistant without HTTP

Run the console harness:

```powershell
python -m AI.test_assistant_cli --attachments AI/sample_documents
```

- Enter `list` to see feature keys, then choose one (e.g. `stakeholder_insights`).  
- Provide a message and review the JSON payload returned.  
- Type `state` to inspect stored requirements, assumptions, and attachments.  
- Replay a canned dialogue with:
  ```powershell
  python -m AI.test_assistant_cli --script tests/conversation.json
  ```
  where the JSON file contains:
  ```json
  {"turns": [{"feature": "requirement_clarifier", "message": "..."}]}
  ```

---

## Retrieval strategies at a glance

- **Chunking**  
  - `all_in_one` → single chunk per document (baseline)  
  - `fixed` → 1,200-character windows with 200-character overlap  
  - `semantic` → align chunks with headings and paragraphs  

- **Indexing**  
  - `none` → linear substring search  
  - `faiss` → cosine similarity over lightweight embeddings  
  - `llama_index` → three-level scorer (document, section, chunk)  

---

## Sample documents

Store any fake artefacts under `AI/sample_documents/`. Both FastAPI uploads and local CLIs read from the same folder.

---

This version assumes the user has enough disk space and memory, so the Ollama setup is straightforward: install, pull a model, run it, and point the app to it.  

I can also prepare a **quick-start checklist** version of this README if you’d like something even shorter and more action-oriented.