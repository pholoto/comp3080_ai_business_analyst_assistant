"""RAG subsystem for the AI Business Analyst Assistant backend."""

from .config import DEFAULT_CONFIG, RagConfig, get_default_config
from .document_store import DocumentStore
from .embedding_generator import EmbeddingGenerator
from .generation import ResponseGenerator
from .pipeline import RagAnswer, RagPipeline, RagQuery, rag_pipeline
from .ranking import SimilarityRanker
from .retrieval import ContextRetriever, RetrievalResult
from .text_splitter import DocumentChunk, TextSplitter
from .vector_store import FaissVectorStore, InMemoryVectorStore

__all__ = [
    "RagConfig",
    "DocumentStore",
    "EmbeddingGenerator",
    "ResponseGenerator",
    "SimilarityRanker",
    "ContextRetriever",
    "RetrievalResult",
    "DocumentChunk",
    "TextSplitter",
    "FaissVectorStore",
    "InMemoryVectorStore",
    "RagPipeline",
    "RagQuery",
    "RagAnswer",
    "rag_pipeline",
    "DEFAULT_CONFIG",
    "get_default_config",
]
