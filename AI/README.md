# comp3080_ai_business_analyst_assistant

## AI Module Quickstart

The `AI/` package exposes the AIBA (AI Business Analyst) assistant as a FastAPI
service. It provides six conversation-driven expert modes that share the same
session memory:

- `requirement_clarifier`
- `use_case_generator`
- `feature_prioritization`
- `market_fit_analyzer`
- `stakeholder_insights`
- `ba_report_export`

### Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r AI/requirements.txt
```

### Choosing an LLM Backend

The API expects a running Ollama instance for real language-model responses,
but automatically falls back to an offline stub that returns placeholder text
when no backend is reachable. This means you can start developing immediately
without installing any model, albeit with mock answers.

#### Option A — Install Ollama locally

1. Download the latest Ollama release for Windows from the official website or
   run the provided installer (for example `E:\ollamasetup.exe`).
2. Launch the Ollama service and pull the llama model:

    ```powershell
    ollama serve
    ollama run llama3.1 --keep-alive
    ```

3. (Optional) Point the backend to a different host or model:

    ```powershell
    $Env:OLLAMA_BASE_URL = "http://localhost:11434"
    $Env:OLLAMA_MODEL = "llama3.1"
    ```

#### Option B — Remote / Shared Ollama

If your machine cannot run Ollama comfortably, deploy Ollama on a workstation
or server that meets the requirements below, then set `OLLAMA_BASE_URL` to that
host before starting the API.

### Hardware Requirements for Ollama (llama3.1)

- Minimum: 8 physical CPU cores, 16 GB RAM (model loads roughly 10–12 GB).
- Recommended: discrete GPU with ≥12 GB VRAM or CPU with AVX2 support, 32 GB RAM
  for smoother inference.
- Disk space: ~8 GB to store the quantised llama3.1 model.

If these requirements are not met, expect slow responses or loading failures.
In that case, rely on the stubbed replies or use a remote host.

### Running the API locally

```powershell
python -m AI
```

The FastAPI server listens on `http://127.0.0.1:8000` by default.

### Core Endpoints

- `POST /sessions` → create a new shared-memory chat session.
- `POST /sessions/{session_id}/chat` → invoke one of the six features while
  preserving the conversation history.
- `GET /sessions/{session_id}/transcript` → retrieve the full conversation log.
- `GET /sessions/{session_id}/state` → fetch the aggregated artefacts (requirements,
  stories, stakeholders, etc.).
- `GET /features` → list available feature keys and descriptions.

The `ba_report_export` feature compiles every artefact into the official VinUni
CECS DOCX template found under `back_end/templates/`. Reports are saved in
`reports/` at the repository root.
