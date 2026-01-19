"""Run local experiments comparing chunking and indexing strategies."""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence

from .chunking import available_chunkers, describe_chunkers, get_chunker
from .evaluation import (mean_reciprocal_rank, ndcg_at_k, precision_at_k,
                         recall_at_k, summarise_latency)
from .indexing import available_indexers, describe_indexers, get_indexer
from .memory.attachments import extract_text_from_attachment

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pdf"}
DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_documents"


@dataclass
class Document:
    doc_id: str
    path: Path
    text: str

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class QuerySpec:
    query: str
    relevant_chunks: List[str]
    top_k: Optional[int] = None


def load_documents(paths: Sequence[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        raw = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        try:
            text, _ = extract_text_from_attachment(path.name, content_type, raw)
        except RuntimeError as exc:
            print(f"[warn] Skipping {path.name}: {exc}")
            continue
        if not text.strip():
            continue
        doc_id = path.stem
        documents.append(Document(doc_id=doc_id, path=path, text=text))
    return documents


def discover_documents(source: Path, limit: Optional[int]) -> List[Document]:
    if source.is_file():
        candidates = [source]
    else:
        candidates = sorted(source.rglob("*"))
    documents = load_documents(candidates)
    if limit is not None:
        documents = documents[:limit]
    return documents


def load_queries(path: Optional[Path], documents: Sequence[Document], default_top_k: int) -> List[QuerySpec]:
    if path is None:
        return build_default_queries(documents, default_top_k)
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("queries", []) if isinstance(raw, dict) else raw
    queries: List[QuerySpec] = []
    for item in items:
        query = item.get("query") if isinstance(item, dict) else None
        if not query:
            continue
        relevant = item.get("relevant_chunks", []) if isinstance(item, dict) else []
        top_k = item.get("top_k") if isinstance(item, dict) else None
        if not relevant:
            relevant = [query]
        queries.append(QuerySpec(query=query, relevant_chunks=list(relevant), top_k=top_k))
    if not queries:
        queries = build_default_queries(documents, default_top_k)
    return queries


def build_default_queries(documents: Sequence[Document], default_top_k: int) -> List[QuerySpec]:
    queries: List[QuerySpec] = []
    for doc in documents:
        snippet = _first_meaningful_snippet(doc.text)
        if not snippet:
            continue
        query = _build_query_from_snippet(snippet)
        queries.append(
            QuerySpec(
                query=query,
                relevant_chunks=[snippet],
                top_k=default_top_k,
            )
        )
    return queries


def _first_meaningful_snippet(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    return cleaned[:max_chars]


def _build_query_from_snippet(snippet: str, max_words: int = 8) -> str:
    words = [word for word in snippet.split() if len(word) > 2]
    return " ".join(words[:max_words]) if words else snippet


def evaluate_combinations(
    documents: Sequence[Document],
    queries: Sequence[QuerySpec],
    chunker_keys: Sequence[str],
    indexer_keys: Sequence[str],
    top_k: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for chunk_key in chunker_keys:
        chunker = get_chunker(chunk_key)
        print(f"\n[chunking] {chunker.name}: {chunker.description}")
        chunked_docs = _chunk_documents(chunker, documents)
        for index_key in indexer_keys:
            indexer = get_indexer(index_key)
            print(f"  [indexing] {indexer.name}: {indexer.description}")
            build_ms = _build_index(indexer, chunked_docs)
            evaluation = _evaluate_index(indexer, queries, top_k)
            evaluation["chunking"] = chunker.name
            evaluation["indexing"] = indexer.name
            evaluation["index_build_ms"] = build_ms
            results.append(evaluation)
            _print_summary(evaluation)
    return results


def _chunk_documents(chunker, documents: Sequence[Document]) -> Dict[str, List[Dict[str, object]]]:
    chunked: Dict[str, List[Dict[str, object]]] = {}
    for doc in documents:
        chunks = chunker.chunk(doc.text)
        chunked[doc.doc_id] = [
            {
                "chunk": chunk,
                "metadata": {
                    "document_id": doc.doc_id,
                    "filename": doc.name,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                },
            }
            for idx, chunk in enumerate(chunks)
        ]
    return chunked


def _build_index(indexer, chunked_docs: Dict[str, List[Dict[str, object]]]) -> float:
    docs: List[str] = []
    metadata: List[dict] = []
    for entries in chunked_docs.values():
        for entry in entries:
            docs.append(str(entry["chunk"]))
            meta = entry.get("metadata") if isinstance(entry, dict) else None
            if isinstance(meta, dict):
                metadata.append(dict(meta))
            else:
                metadata.append({})
    start = perf_counter()
    indexer.reset()
    if docs:
        indexer.add_documents(docs, metadata=metadata)
    elapsed = (perf_counter() - start) * 1000.0
    return elapsed


def _evaluate_index(indexer, queries: Sequence[QuerySpec], default_top_k: int) -> Dict[str, object]:
    if not queries:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "per_query": [],
        }
    latencies: List[float] = []
    per_query: List[Dict[str, object]] = []
    aggregate_precision: List[float] = []
    aggregate_recall: List[float] = []
    aggregate_mrr: List[float] = []
    aggregate_ndcg: List[float] = []
    for spec in queries:
        top_k = spec.top_k or default_top_k
        start = perf_counter()
        results = indexer.search(spec.query, top_k=top_k)
        latencies.append((perf_counter() - start) * 1000.0)
        retrieved_chunks = [result.chunk for result in results]
        relevance_flags = _compute_relevance_flags(retrieved_chunks, spec.relevant_chunks)
        precision = precision_at_k(relevance_flags, top_k)
        recall = recall_at_k(relevance_flags, len(spec.relevant_chunks), top_k)
        mrr_value = mean_reciprocal_rank(relevance_flags)
        ndcg_value = ndcg_at_k(relevance_flags, top_k)
        aggregate_precision.append(precision)
        aggregate_recall.append(recall)
        aggregate_mrr.append(mrr_value)
        aggregate_ndcg.append(ndcg_value)
        per_query.append(
            {
                "query": spec.query,
                "top_k": top_k,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "mrr": mrr_value,
                "ndcg_at_k": ndcg_value,
            }
        )
    median, p95 = summarise_latency(latencies)
    count = len(per_query) or 1
    return {
        "precision_at_k": sum(aggregate_precision) / count,
        "recall_at_k": sum(aggregate_recall) / count,
        "mrr": sum(aggregate_mrr) / count,
        "ndcg_at_k": sum(aggregate_ndcg) / count,
        "median_latency_ms": median,
        "p95_latency_ms": p95,
        "per_query": per_query,
    }


def _compute_relevance_flags(retrieved: Sequence[str], relevant: Sequence[str]) -> List[int]:
    if not retrieved:
        return []

    prepared = [
        {
            "lower": rel.lower(),
            "tokens": _tokenise(rel.lower()) if rel else set(),
            "matched": False,
        }
        for rel in relevant
    ]

    flags: List[int] = []
    for chunk in retrieved:
        chunk_lower = chunk.lower()
        chunk_tokens = _tokenise(chunk_lower)
        match_idx: Optional[int] = None
        for idx, candidate in enumerate(prepared):
            if candidate["matched"]:
                continue
            rel_lower = candidate["lower"]
            if not rel_lower:
                continue
            if rel_lower in chunk_lower:
                match_idx = idx
                break
            rel_tokens = candidate["tokens"]
            if rel_tokens:
                overlap = len(chunk_tokens & rel_tokens) / len(rel_tokens)
                if overlap >= 0.6:
                    match_idx = idx
                    break
        if match_idx is not None:
            prepared[match_idx]["matched"] = True
            flags.append(1)
        else:
            flags.append(0)
    return flags


_TOKEN_PATTERN = re.compile(r"[\w']+")


def _tokenise(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text))


def _print_summary(result: Dict[str, object]) -> None:
    print(
        "    ↳ metrics: P@k={precision:.2f}, R@k={recall:.2f}, MRR={mrr:.2f}, NDCG@k={ndcg:.2f}, "
        "median={median:.1f}ms, p95={p95:.1f}ms".format(
            precision=result["precision_at_k"],
            recall=result["recall_at_k"],
            mrr=result["mrr"],
            ndcg=result["ndcg_at_k"],
            median=result["median_latency_ms"],
            p95=result["p95_latency_ms"],
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare chunking and indexing strategies locally.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory or file containing sample documents (default: AI/sample_documents).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Optional path to JSON file with evaluation queries.",
    )
    parser.add_argument(
        "--chunkers",
        nargs="*",
        default=None,
        help="Subset of chunking strategy keys to evaluate.",
    )
    parser.add_argument(
        "--indexers",
        nargs="*",
        default=None,
        help="Subset of indexing strategy keys to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default number of results to retrieve per query.",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=None,
        help="Optional cap on the number of documents to load.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the aggregated results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = discover_documents(args.documents, args.limit_documents)
    if not documents:
        print("No documents found. Drop PDFs/DOCXs/TXTs into AI/sample_documents or specify --documents.")
        return
    chunker_keys = args.chunkers or list(available_chunkers())
    indexer_keys = args.indexers or list(available_indexers())
    print("Loaded documents:")
    for doc in documents:
        print(f" - {doc.name}")
    print("\nChunking strategies:")
    for entry in describe_chunkers():
        if entry["key"] in chunker_keys:
            print(f" - {entry['key']}: {entry['description']}")
    print("\nIndexing strategies:")
    for entry in describe_indexers():
        if entry["key"] in indexer_keys:
            print(f" - {entry['key']}: {entry['description']}")
    queries = load_queries(args.queries, documents, args.top_k)
    if not queries:
        print("No queries available; aborting evaluation.")
        return
    print(f"\nEvaluating {len(chunker_keys) * len(indexer_keys)} combinations across {len(queries)} queries...")
    results = evaluate_combinations(documents, queries, chunker_keys, indexer_keys, args.top_k)
    if args.save_json:
        payload = {
            "documents": [doc.name for doc in documents],
            "results": results,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved detailed metrics to {args.save_json}")


if __name__ == "__main__":
    main()
