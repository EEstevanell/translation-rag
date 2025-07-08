"""Vector similarity based retrieval strategy."""

from typing import Optional

from ..config import Config
from ..logging_utils import get_logger
from ..pipeline import RAGPipeline
from .base import RAGStrategy


class SemanticRAG(RAGStrategy):
    """Retrieve context using vector similarity search."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.logger = get_logger()

    def get_context(
        self, text: str, source_lang: str, target_lang: str, k: int = 4
    ) -> Optional[str]:
        if not self.pipeline.vectorstore:
            return None

        metadata_filter = None
        if source_lang != "unknown":
            metadata_filter = {"source_lang": source_lang, "target_lang": target_lang}

        filter_dict = (
            self.pipeline._build_filter(metadata_filter) if metadata_filter else None
        )
        results_with_scores = self.pipeline.vectorstore.similarity_search_with_score(
            text, k=k, filter=filter_dict
        )
        if not results_with_scores:
            self.logger.info("No relevant documents retrieved")
            return None

        filtered = []
        for i, (doc, score) in enumerate(results_with_scores):
            similarity = 1 - score if score <= 1 else 0
            preview = doc.page_content.replace("\n", " ")[:60]
            self.logger.info(f"  [{i+1}] Similarity: {similarity:.3f} | {preview}...")
            if similarity >= Config.SIMILARITY_THRESHOLD:
                filtered.append(doc)
            else:
                self.logger.info(
                    f"  ✗ Filtered out (below threshold {Config.SIMILARITY_THRESHOLD:.3f})"
                )

        if not filtered:
            self.logger.info(
                f"No documents above similarity threshold {Config.SIMILARITY_THRESHOLD:.3f}"
            )
            return None

        context_parts = []
        for doc in filtered:
            tgt = doc.metadata.get("target_sentence")
            if tgt:
                context_parts.append(f"{doc.page_content} -> {tgt}")
            else:
                context_parts.append(doc.page_content)
        return "\n".join(context_parts)
