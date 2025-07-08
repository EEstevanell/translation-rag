"""Levenshtein distance based retrieval strategy."""

from typing import Optional

from ..translation_memory import TranslationMemory
from .base import RAGStrategy


class LevenshteinRAG(RAGStrategy):
    """Retrieve context using edit distance similarity."""

    def __init__(self, memory: TranslationMemory):
        self.memory = memory

    def get_context(
        self, text: str, source_lang: str, target_lang: str, k: int = 4
    ) -> Optional[str]:
        entries = self.memory.retrieve_levenshtein(text, source_lang, target_lang, k=k)
        if not entries:
            return None
        return "\n".join(f"{e.source_sentence} -> {e.target_sentence}" for e in entries)
