"""Base interface for retrieval strategies."""

from abc import ABC, abstractmethod
from typing import Optional


class RAGStrategy(ABC):
    """Interface for building retrieval augmented strategies."""

    @abstractmethod
    def get_context(
        self, text: str, source_lang: str, target_lang: str, k: int = 4
    ) -> Optional[str]:
        """Return contextual examples for the given text."""
