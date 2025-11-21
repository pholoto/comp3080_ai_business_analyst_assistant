"""Deterministic conversation harness for exercising the backend stack.

python -m back_end.conversation_simulator

This mirrors the interactive CLI in AI/test_assistant_cli.py but runs a fixed
scenario so backend contributors can quickly sanity check RAG attachments and
feature flows without manual input.
"""
from __future__ import annotations

import json
import mimetypes
import time
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Sequence

from AI.features import FeatureContext, build_default_registry
from AI.llm import get_default_client
from AI.memory import Session, SessionManager
from back_end.rag.document_store import DocumentStore, DuplicateDocumentError

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pdf"}
ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT_DIR / "AI" / "sample_documents"
DEFAULT_UPLOAD_SOURCES = [
    SAMPLE_DIR / "FEDAVG Communication-Efficient Learning of Deep Networks.pdf",
    SAMPLE_DIR / "Federated Domain Generalization A Survey.pdf",
]
SIMULATED_USER_ID = "simulated_user"

SCENARIO = [
    {
        "feature": "requirement_clarifier",
        "message": "Could you summarize what federated learning is and why organizations use it?",
        "label": "Foundational knowledge",
    },
    {
        "feature": "use_case_generator",
        "message": (
            "I need to design a federated learning system that improves keyboard word suggestions for "
            "Vietnamese users. Highlight potential components."),
        "label": "Solution ideation",
    },
    {
        "feature": "requirement_clarifier",
        "message": "Please clarify the specific requirements we should gather for that Vietnamese federated keyboard system.",
        "label": "Requirement deep-dive",
    },
]


def simulate_conversation(
    *,
    upload_sources: Sequence[Path] = DEFAULT_UPLOAD_SOURCES,
    scenario: Sequence[dict] = SCENARIO,
) -> None:
    manager = SessionManager()
    session = manager.create_session()
    session.set_state("user_id", SIMULATED_USER_ID)
    llm = get_default_client()
    registry = build_default_registry()
    store = DocumentStore()

    stored_copies = _simulate_user_uploads(store, SIMULATED_USER_ID, upload_sources)
    if stored_copies:
        print(f"Stored {len(stored_copies)} document(s) under data/{SIMULATED_USER_ID}.")
    else:
        print("No documents were stored. Adjust DEFAULT_UPLOAD_SOURCES if needed.")

    loaded = _attach_files(session, stored_copies)
    if loaded:
        print(f"Attached {len(loaded)} supporting document(s).")
        for path in loaded:
            print(f" - {path.name}")
    else:
        print("No attachments were loaded. Adjust DEFAULT_UPLOAD_SOURCES if needed.")

    for idx, turn in enumerate(scenario, start=1):
        feature_key = turn["feature"].lower()
        message = turn["message"].strip()
        label = turn.get("label") or f"Turn {idx}"
        print(f"\n=== Turn {idx}: {label} ({feature_key}) ===")
        result = _invoke_feature(session, registry, llm, feature_key, message)
        if result is None:
            print(f"Feature '{feature_key}' is not registered; skipping.")
            continue
        _print_result(result)
        # Pause between requests to simulate a user thinking/typing delay.
        # Do not sleep after the final turn.
        try:
            total = len(scenario)
        except Exception:
            total = None
        if total is None or (isinstance(total, int) and idx < total):
            time.sleep(10)

    _print_state_snapshot(session)
    _print_history(session)


def _attach_files(session: Session, inputs: Sequence[Path]) -> List[Path]:
    files = list(_discover_files(inputs))
    loaded: List[Path] = []
    for path in files:
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"[warn] Cannot read {path}: {exc}")
            continue
        content_type, _ = mimetypes.guess_type(path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        try:
            session.add_attachment(
                filename=path.name,
                content_type=content_type,
                data=data,
            )
        except Exception as exc:  # pragma: no cover - safety during manual runs
            print(f"[warn] Failed to attach {path.name}: {exc}")
            continue
        loaded.append(path)
    return loaded


def _simulate_user_uploads(
    store: DocumentStore,
    user_id: str,
    sources: Sequence[Path],
) -> List[Path]:
    stored: List[Path] = []
    for path in _discover_files(sources):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            print(f"[warn] Cannot read {path}: {exc}")
            continue
        checksum = sha256(payload).hexdigest()
        existing = store.get_document_by_checksum(user_id, checksum)
        if existing:
            stored.append(Path(existing.stored_path))
            continue
        try:
            record = store.ingest_file(user_id, path.name, payload)
        except DuplicateDocumentError as exc:
            print(f"[warn] Duplicate detected but missing checksum match for {path.name}: {exc}")
            continue
        stored.append(Path(record.metadata.stored_path))
    return stored


def _discover_files(inputs: Sequence[Path]) -> Iterable[Path]:
    for entry in inputs:
        if entry.is_dir():
            for candidate in sorted(entry.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield entry


def _invoke_feature(session: Session, registry, llm, feature_key: str, message: str):
    context = FeatureContext(session=session, llm=llm)
    try:
        feature = registry.create(feature_key, context)
    except KeyError:
        return None
    session.memory.append("user", message, feature=feature_key)
    try:
        result = feature.run(message, context=context)
    except Exception as exc:  # pragma: no cover - defensive feedback
        print(f"[error] Feature '{feature_key}' failed: {exc}")
        session.memory.append("assistant", f"Encountered error: {exc}", feature=feature_key)
        return None
    # Record the assistant's output in the conversation memory with full detail.
    try:
        payload = result.data if getattr(result, "data", None) is not None else None
        if payload is None:
            assistant_text = str(getattr(result, "summary", ""))
        else:
            try:
                structured = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception:
                structured = str(payload)
            assistant_text = f"{getattr(result, 'summary', '')}\n\n{structured}"
    except Exception:
        assistant_text = str(getattr(result, "summary", ""))

    session.memory.append("assistant", assistant_text, feature=feature_key)
    return result


def _print_result(result) -> None:
    print(f"Assistant ({result.title}):")
    # Primary human-readable summary
    try:
        print(result.summary)
    except Exception:
        print(str(result.summary))

    # Show every field the feature returned. Iterate top-level keys so
    # callers can easily see user_stories, alternate_paths, requirement_backlog, etc.
    data = result.data if result is not None else None
    if data is None:
        print("<no structured data returned>")
        return

    if not isinstance(data, dict):
        # If data is a primitive or list, just print it.
        try:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            print(str(data))
        return

    if not data:
        print("<structured data is an empty object>")
        return

    print("\nStructured data:")
    for key in sorted(data.keys()):
        value = data.get(key)
        print(f" - {key}:")
        try:
            formatted = json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            formatted = str(value)
        # Indent the printed JSON block for readability
        for line in formatted.splitlines():
            print(f"     {line}")


def _print_state_snapshot(session: Session) -> None:
    state = dict(session.state)
    attachments = state.pop("attachments", [])
    print("\nSession state:")
    for key, value in state.items():
        print(f" - {key}: {value}")
    if attachments:
        print(" - attachments:")
        for meta in attachments:
            name = meta.get("filename") or meta.get("document_label") or meta.get("chunk_id")
            chunk_count = meta.get("chunk_count", "?")
            print(f"   - {name} (chunks: {chunk_count})")


def _print_history(session: Session) -> None:
    print("\nConversation history:")
    if not session.memory.messages:
        print(" - <empty>")
        return
    for message in session.memory.messages:
        feature = f" [{message.feature}]" if message.feature else ""
        print(f" - {message.role}{feature}: {message.content}")


if __name__ == "__main__":
    simulate_conversation()
