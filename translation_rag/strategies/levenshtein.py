"""Levenshtein distance based retrieval strategy."""

from typing import Optional

from ..logging_utils import get_logger
from ..translation_memory import TranslationMemory
from .base import RAGStrategy


class LevenshteinRAG(RAGStrategy):
    """Retrieve context using edit distance similarity."""

    def __init__(self, memory: TranslationMemory):
        self.memory = memory
        self.logger = get_logger()

    def get_context(
        self, text: str, source_lang: str, target_lang: str, k: int = 4
    ) -> Optional[str]:
        entries_with_scores = self.memory.retrieve_levenshtein_with_scores(
            text, source_lang, target_lang, k=k
        )
        if not entries_with_scores:
            self.logger.info("No relevant documents retrieved")
            return None

        context_parts = []
        for i, (score, entry) in enumerate(entries_with_scores):
            preview = entry.source_sentence.replace("\n", " ")[:60]
            self.logger.info(f"  [{i+1}] Levenshtein similarity: {score:.3f} | {preview}...")
            context_parts.append(f"{entry.source_sentence} -> {entry.target_sentence}")
        
        return "\n".join(context_parts)
