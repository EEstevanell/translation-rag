"""Retrieval strategies for Translation RAG."""

from .base import RAGStrategy
from .jaccard import JaccardRAG
from .levenshtein import LevenshteinRAG

__all__ = ["RAGStrategy", "JaccardRAG", "LevenshteinRAG", "SemanticRAG"]


def __getattr__(name: str):
    if name == "SemanticRAG":
        from .semantic import SemanticRAG

        return SemanticRAG
    raise AttributeError(name)
