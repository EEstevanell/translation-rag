"""Levenshtein distance based retrieval strategy."""

from typing import Optional

from ..translation_memory import TranslationMemory
from .base import RAGStrategy
from ..logging_utils import get_logger


class LevenshteinRAG(RAGStrategy):
    """Retrieve context using edit distance similarity."""

    def __init__(self, memory: TranslationMemory):
        self.memory = memory
        self.logger = get_logger()

    def get_context(
        self, text: str, source_lang: str, target_lang: str, k: int = 4
    ) -> Optional[str]:
        results = self.memory.retrieve_levenshtein(
            text, source_lang, target_lang, k=k, return_scores=True
        )
        if not results:
            self.logger.info("No relevant documents retrieved")
            return None

        self.logger.info(
            f"Retrieved {len(results)} documents using Levenshtein search:"
        )
        context_parts = []
        for i, (score, entry) in enumerate(results):
            preview = entry.source_sentence.replace("\n", " ")[:60]
            self.logger.info(f"  [{i+1}] Similarity: {score:.3f} | {preview}...")
            context_parts.append(f"{entry.source_sentence} -> {entry.target_sentence}")

        return "\n".join(context_parts)
