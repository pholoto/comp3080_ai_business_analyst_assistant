# Direct Feature API – AI Business Analyst Assistant

This guide is for front‑end developers who just want to call the AI feature functions and show the LLM’s answer. No server required — you can invoke the Python modules directly. Keep it simple: send a user message in, get a structured answer out.

---
## 1. What You Call (Plain Words)

- Use the feature registry to create a feature by name.
- Provide a message (the user’s question).
- Get back a result with `title`, `summary`, and `data` (structured details you can render).

Available feature names:
`problem_definition`, `requirements_analysis`, `solution_design`, `prototype_development`, `testing_validation`, `market_analysis`, `documentation`.

---
## 2. Minimal Python Example (Single Turn)

Put this in a small Python file (e.g. `examples/run_feature.py`) and run it. It calls one feature and prints the LLM answer.

```python
from AI.features import FeatureContext, build_default_registry
from AI.llm import get_default_client
from AI.memory import SessionManager

# 1) Create a session (stores conversation state)
session = SessionManager().create_session()

# 2) Get an LLM client and feature registry
llm = get_default_client()
registry = build_default_registry()

# 3) Choose a feature and your user message
feature_key = "solution_design"  # pick one from the list above
message = "Propose a high-level architecture for our mobile app."

# 4) Create the feature and run it
context = FeatureContext(session=session, llm=llm)
feature = registry.create(feature_key, context)
result = feature.run(message, context=context)

# 5) Show the output (what you render in front-end)
print("Title:", result.title)
print("Summary:\n", result.summary)
print("Data:")
print(result.data)  # dict with structured fields (e.g., components, steps, matrices)
```

What you’ll get:
- `result.title`: short title for UI headers.
- `result.summary`: main text answer from the LLM.
- `result.data`: JSON‑like dict of structured info to render in cards/tables.

---
## 3. Keeping History (Optional)

If you want multi‑turn behavior, append messages to the session. The features already record turns, but you can also log your own:

```python
session.memory.append("user", message, feature=feature_key)
session.memory.append("assistant", result.summary, feature=feature_key)

# Inspect history later
for m in session.memory.messages:
		print(m.role, m.feature, m.content)
```

---
## 4. Using Your Documents (Optional)

If you want the LLM to cite and use your documents without any server:

```python
from back_end.rag.document_store import DocumentStore
from pathlib import Path

store = DocumentStore()
user_id = "alice"
payload = Path("C:/path/to/YourDoc.pdf").read_bytes()
record = store.ingest_file(user_id, "YourDoc.pdf", payload)

# Attach the file bytes to the session so features can reference it
session.add_attachment(filename=record.metadata.original_name,
											content_type="application/pdf",
											data=payload)

# Now run features as before; they can use/cite the attached doc.
```

Tip: Supported extensions include `.pdf`, `.docx`, `.txt`, `.md`.

---
## 5. Conversation Simulator (Scripted Demo)

Run a deterministic multi‑turn scenario with sample documents to see the typical outputs:

```powershell
python -m back_end.conversation_simulator
```

It will:
- Load sample PDFs from `AI/sample_documents/`.
- Attach them to a test session.
- Run several feature turns (problem → requirements → design → prototype → testing → market → documentation).
- Print human summaries and structured `data` for each turn.

Customize it by editing `DEFAULT_UPLOAD_SOURCES` and `SCENARIO` in `back_end/conversation_simulator.py`.

---
## 6. Error Handling (Quick Notes)

- If modules are missing, install requirements:
	`pip install -r AI/requirements.txt` and `pip install -r back_end/requirements.txt`.
- Document ingestion errors (duplicates) may raise `DuplicateDocumentError` — handle gracefully or skip.
- If you later use FAISS for chunk retrieval, install `faiss-cpu`.

---
## 7. Front‑End Integration Ideas

- Call a small Python script (like the example) from your tooling and capture `stdout`/JSON.
- Or wrap the example in a tiny local Python service of your choice if you prefer HTTP — but it’s optional.
- Render `result.summary` as the main answer; map fields in `result.data` to tables/cards.

---
## 8. Quick Feature Reference

| Feature | Purpose |
|---------|---------|
| `problem_definition` | Clarify problem space & pain points |
| `requirements_analysis` | Functional & non‑functional requirements |
| `solution_design` | Architecture, patterns, tech choices |
| `prototype_development` | MVP scope & implementation tips |
| `testing_validation` | Test matrix & validation plan |
| `market_analysis` | Competitors & GTM ideas |
| `documentation` | Structured SRS / docs export |

Use these strings in `feature_key` when creating the feature via the registry.

---
## 9. Safety Note

Generated content may contain mistakes. Review before using in reports or code.

---
Happy building!

---
## 2. Core Concepts (Plain Language)

- **User ID**: A simple string you choose (e.g. `"alice"`). Used to keep documents and conversation state separate per user.
- **Documents**: Files you upload (PDF, DOCX, TXT, MD) so the assistant can cite and use them when answering.
- **RAG**: “Retrieval Augmented Generation” – the system pulls relevant document chunks before asking the AI model.
- **Features**: Specialized analysis modes (problem definition, requirements analysis, solution design, prototype development, testing & validation, market analysis, documentation). You invoke them via `/chat` with a `feature` name.
- **Session State**: The backend remembers previous turns and uploaded files for the same user ID.

---
## 3. Available Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Basic status check |
| GET | `/users/{user_id}/documents` | List uploaded documents for that user |
| POST | `/users/{user_id}/documents` | Upload & index a document (multipart) |
| POST | `/users/{user_id}/ideation` | Generate idea list for a topic |
| POST | `/users/{user_id}/progress` | Analyze progress/status using stored docs |
| POST | `/users/{user_id}/rag` | Ask a direct question with retrieval & citations |
| POST | `/users/{user_id}/chat` | Use an AI feature with optional retrieval |

### 3.1 Upload a Document

Multipart form fields:
- `file`: the file to upload
- `tags`: (optional) comma‑separated labels (e.g. `project,baseline`)

Example (PowerShell `curl`):
```powershell
curl -F "file=@C:/path/to/YourDoc.pdf" -F "tags=project,baseline" http://127.0.0.1:8000/users/alice/documents
```
Response (simplified):
```json
{
	"metadata": {
		"document_id": "...",
		"original_name": "YourDoc.pdf",
		"tags": ["project", "baseline"],
		"created_at": "2025-12-01T12:34:56Z"
	},
	"chunks_indexed": 42
}
```

### 3.2 List Documents
```powershell
curl http://127.0.0.1:8000/users/alice/documents
```

### 3.3 Feature Names for `/chat`
Use one of: `problem_definition`, `requirements_analysis`, `solution_design`, `prototype_development`, `testing_validation`, `market_analysis`, `documentation`.

### 3.4 Run a Feature with Retrieval
`POST /users/{user_id}/chat`
Body JSON:
```json
{
	"feature": "solution_design",
	"message": "Propose a high level architecture for our mobile app.",
	"use_rag": true,
	"rag_top_k": 6,
	"rag_tags": ["project"],
	"rag_score_threshold": 0.3
}
```
Response (simplified):
```json
{
	"title": "Solution Architecture Overview",
	"summary": "High level components ...",
	"citations": [ {"source": "YourDoc.pdf", "score": 0.78 } ],
	"context_snippets": [ {"source_name": "YourDoc.pdf", "preview": "..."} ],
	"session_state": {"last_rag_query": "Propose a high level architecture..."}
}
```

### 3.5 Direct RAG Question (No feature extras)
`POST /users/{user_id}/rag`
```json
{
	"question": "Summarize the baseline performance metrics in our uploaded docs",
	"top_k": 5,
	"tags": ["baseline"]
}
```

### 3.6 Ideation Example
`POST /users/{user_id}/ideation`
```json
{
	"topic": "Ways to improve onboarding experience",
	"desired_ideas": 6,
	"top_k": 8,
	"tags": ["project"]
}
```

### 3.7 Progress Analysis Example
`POST /users/{user_id}/progress`
```json
{
	"reference_date": "2025-12-01",
	"top_k": 8,
	"tags": ["baseline"]
}
```

---
## 4. Calling from Front‑End JavaScript (Fetch Examples)

Assuming server at `http://127.0.0.1:8000`.

### 4.1 Upload File (browser)
```javascript
async function uploadDoc(file) {
	const form = new FormData();
	form.append('file', file);
	form.append('tags', 'project,baseline');
	const res = await fetch('http://127.0.0.1:8000/users/alice/documents', {
		method: 'POST',
		body: form
	});
	return res.json();
}
```

### 4.2 Run Feature Chat
```javascript
async function runFeature(message) {
	const payload = {
		feature: 'solution_design',
		message,
		use_rag: true,
		rag_top_k: 6
	};
	const res = await fetch('http://127.0.0.1:8000/users/alice/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload)
	});
	return res.json();
}
```

### 4.3 Plain RAG Question
```javascript
async function askRag(question) {
	const payload = { question, top_k: 5 };
	const res = await fetch('http://127.0.0.1:8000/users/alice/rag', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(payload)
	});
	return res.json();
}
```

Handle errors simply:
```javascript
try { const data = await runFeature('Design our system'); } catch (e) { console.error(e); }
```

---
## 5. Conversation Simulator (Deterministic Demo)

Purpose: Quickly see a multi‑turn scripted scenario (problem definition → requirements → design → prototype → testing → market → documentation) with sample uploaded research papers.

### 5.1 Run It
From project root (after installing dependencies):
```powershell
python -m back_end.conversation_simulator
```

### 5.2 What It Does
1. Creates an in‑memory session for user `simulated_user`.
2. Uploads sample PDF documents found in `AI/sample_documents/` (adjust list in `conversation_simulator.py` if paths change).
3. Attaches those documents to the session so retrieval can cite them.
4. Executes each scripted turn (feature + message) with a short pause.
5. Prints:
	 - Each feature’s human‑readable summary.
	 - Structured data fields (e.g. architecture blueprint, requirement lists, test matrix).
	 - Final session state and conversation history.

### 5.3 Customizing
Edit arrays in `conversation_simulator.py`:
- `DEFAULT_UPLOAD_SOURCES`: change or add file paths.
- `SCENARIO`: add/remove turns; each dictionary has `feature`, `message`, and optional `label`.

### 5.4 Troubleshooting
- If documents are not found: check the file paths exist on disk; adjust extensions to supported types (.pdf, .docx, .txt, .md).
- If FAISS error appears when using the API for document uploads: ensure `pip install faiss-cpu` succeeded.
- Long pauses: adjust `time.sleep(10)` near the end of each loop iteration.

---
## 6. Common Errors & Fixes

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | Re‑run the two `pip install` commands in section 1C. Make sure virtual env is activated. |
| `RuntimeError: faiss-cpu is required` | Install `faiss-cpu` then restart `uvicorn`. |
| Empty citations list | Upload relevant documents first; set `use_rag: true`. |
| 409 on upload | Duplicate file already ingested (same checksum). Rename or modify content if truly new. |

---
## 7. Minimal Testing Checklist (Manual)

1. Start server (`uvicorn ...`).
2. `GET /health` returns `{ "status": "ok" }`.
3. Upload a PDF → response shows `chunks_indexed > 0`.
4. `GET /users/{user}/documents` lists the PDF.
5. Call `/chat` with a feature + `use_rag: true` → response includes `citations`.
6. Run simulator → observe multi‑turn output.

If all pass, front‑end integration can proceed.

---
## 8. Next Steps for Front‑End Integration

- Create a simple settings panel for choosing `user_id`.
- Show streaming or loading state while awaiting `/chat` response.
- Display `citations` with source names; optionally fetch `/users/{user}/documents` to map names.
- Offer a “Upload Docs” button using the file upload example.

---
## 9. License / Attribution

Internal educational project. Do not distribute externally without approval.

---
## 10. FAQ (Short)

**Do I need a database?** Not yet; storage uses local file paths inside `back_end/data/`.

**Can I reset a user’s session?** Change to a new `user_id` or delete their stored files under `back_end/data/<user_id>` while server is stopped.

**How do I add a new feature?** Define it under `AI/features/` (see existing modules), register via `build_default_registry()`.

---
## 11. Quick Feature Reference

| Feature | Purpose |
|---------|---------|
| `problem_definition` | Clarify problem space & pain points |
| `requirements_analysis` | Functional & non‑functional requirements |
| `solution_design` | Architecture, patterns, tech choices |
| `prototype_development` | MVP scope & implementation tips |
| `testing_validation` | Test matrix & validation plan |
| `market_analysis` | Competitors & GTM ideas |
| `documentation` | Structured SRS / docs export |

Use these strings in the `feature` field of `/chat` requests.

---
## 12. Safety Note

Generated content may contain mistakes. Always review before using in reports or code.

---
Happy building!

