"""Console harness for exercising the AI Business Analyst assistant features.

This module mirrors the FastAPI workflow without requiring an HTTP server.
It lets you attach sample documents, switch retrieval strategies, and invoke
features directly from the terminal. Run with:

    python -m AI.test_assistant_cli --attachments AI/sample_documents

Optionally load a scripted conversation by passing ``--script path/to/turns.json``.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from .chunking import available_chunkers
from .features import FeatureContext, build_default_registry
from .indexing import available_indexers
from .llm import get_default_client
from .memory import Session, SessionManager

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try the assistant without running the FastAPI server."
    )
    parser.add_argument(
        "--attachments",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional files or folders to attach before the first turn. "
            "Use AI/sample_documents for the bundled examples."
        ),
    )
    parser.add_argument(
        "--chunking",
        choices=sorted(available_chunkers()),
        default=None,
        help="Override the default chunking strategy for the session.",
    )
    parser.add_argument(
        "--indexing",
        choices=sorted(available_indexers()),
        default=None,
        help="Override the default indexing strategy for search results.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing scripted turns. "
            "If omitted, the tool drops into interactive mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = SessionManager()
    session = manager.create_session()
    if args.chunking:
        session.set_chunking_strategy(args.chunking)
    if args.indexing:
        session.set_indexing_strategy(args.indexing)

    llm = get_default_client()
    registry = build_default_registry()

    if args.attachments:
        loaded = _attach_files(session, args.attachments)
        if loaded:
            print(f"Attached {len(loaded)} document(s).")
        else:
            print("No attachments matched the supported formats; continuing without them.")
    else:
        print("No attachments provided. Use --attachments AI/sample_documents to preload examples.")

    if args.script:
        turns = _load_scripted_turns(args.script)
        _run_scripted_turns(session, registry, llm, turns)
    else:
        _interactive_loop(session, registry, llm)


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
        except Exception as exc:  # pragma: no cover - defensive during manual runs
            print(f"[warn] Failed to attach {path.name}: {exc}")
            continue
        loaded.append(path)
    return loaded


def _discover_files(inputs: Sequence[Path]) -> Iterator[Path]:
    for entry in inputs:
        if entry.is_dir():
            for candidate in sorted(entry.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield entry


def _load_scripted_turns(path: Path) -> List[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read scripted turns: {exc}") from exc
    raw_turns = payload.get("turns", payload)
    if not isinstance(raw_turns, Iterable):
        raise SystemExit("Script format must be a list or an object with a 'turns' list.")
    turns: List[dict] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            print("[warn] Skipping malformed turn entry; expected an object.")
            continue
        feature = str(item.get("feature", "")).strip()
        message = str(item.get("message", "")).strip()
        if not feature or not message:
            print("[warn] Scripted turn is missing 'feature' or 'message'; skipping.")
            continue
        turns.append({"feature": feature, "message": message})
    if not turns:
        raise SystemExit("No valid turns found in the script.")
    return turns


def _run_scripted_turns(session: Session, registry, llm, turns: Sequence[dict]) -> None:
    print(f"Running scripted conversation with {len(turns)} turn(s)...")
    for idx, turn in enumerate(turns, start=1):
        feature_key = turn["feature"]
        message = turn["message"]
        print(f"\nTurn {idx}: {feature_key}\nUser: {message}")
        result = _invoke_feature(session, registry, llm, feature_key, message)
        if result is None:
            print("  ↳ skipped (unknown feature)")
            continue
        _print_result(result)
    _print_state_snapshot(session)


def _interactive_loop(session: Session, registry, llm) -> None:
    print("\nAvailable commands: 'list', 'state', 'history', 'quit'.")
    print("Type a feature key to run it (for example: requirement_clarifier).\n")
    while True:
        try:
            choice = input("feature> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not choice:
            continue
        lowered = choice.lower()
        if lowered in {"quit", "exit"}:
            print("Goodbye!")
            break
        if lowered == "list":
            _print_feature_catalog(session, registry, llm)
            continue
        if lowered == "state":
            _print_state_snapshot(session)
            continue
        if lowered == "history":
            _print_history(session)
            continue
        message = input("message> ").strip()
        if not message:
            print("[warn] Empty message; nothing sent.")
            continue
        result = _invoke_feature(session, registry, llm, lowered, message)
        if result is None:
            print(f"[warn] Unknown feature '{lowered}'. Type 'list' to see options.")
            continue
        _print_result(result)


def _invoke_feature(session: Session, registry, llm, feature_key: str, message: str):
    context = FeatureContext(session=session, llm=llm)
    try:
        feature = registry.create(feature_key, context)
    except KeyError:
        return None
    session.memory.append("user", message, feature=feature_key)
    try:
        result = feature.run(message, context=context)
    except Exception as exc:  # pragma: no cover - interactive error handling
        print(f"[error] Feature '{feature_key}' failed: {exc}")
        session.memory.append("assistant", f"Encountered error: {exc}", feature=feature_key)
        return None
    return result


def _print_feature_catalog(session: Session, registry, llm) -> None:
    context = FeatureContext(session=session, llm=llm)
    print("\nFeatures:")
    for key in sorted(registry.keys()):
        feature = registry.create(key, context)
        print(f" - {feature.name}: {feature.description}")
    print()


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
    print()


def _print_history(session: Session) -> None:
    print("\nConversation history:")
    if not session.memory.messages:
        print(" - <empty>")
    for message in session.memory.messages:
        feature = f" [{message.feature}]" if message.feature else ""
        print(f" - {message.role}{feature}: {message.content}")
    print()


def _print_result(result) -> None:
    print(f"\nAssistant ({result.title}):")
    print(result.summary)
    if result.data:
        try:
            formatted = json.dumps(result.data, indent=2, ensure_ascii=False)
        except TypeError:
            formatted = str(result.data)
        print(formatted)
    print()


if __name__ == "__main__":
    main()
