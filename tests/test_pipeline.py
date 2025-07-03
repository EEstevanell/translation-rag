try:
    from translation_rag.pipeline import RAGPipeline, get_embeddings
    from langchain.llms.base import LLM
except Exception:  # pragma: no cover - skip if deps missing
    import pytest
    pytest.skip("LangChain not available", allow_module_level=True)

class EchoLLM(LLM):
    """Simple LLM that echoes prompts for testing."""
    def _call(self, prompt: str, stop=None):
        return f"ECHO: {prompt[:20]}"

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "echo"


def test_pipeline_basic(tmp_path):
    embeddings = get_embeddings("all-MiniLM-L6-v2")
    pipeline = RAGPipeline(EchoLLM(), embeddings, persist_directory=str(tmp_path))
    pipeline.add_documents(["Hello world"], metadatas=[{"lang": "en"}])
    result = pipeline.query("Hi", use_rag=True)
    assert "ECHO:" in result


def test_pipeline_with_memory(tmp_path):
    embeddings = get_embeddings("all-MiniLM-L6-v2")
    pipeline = RAGPipeline(EchoLLM(), embeddings, persist_directory=str(tmp_path))
    from translation_rag.translation_memory import load_fake_memory, memory_to_documents

    texts, metas = memory_to_documents(load_fake_memory())
    pipeline.add_documents(texts, metas)
    assert pipeline.vectorstore is not None


def test_build_filter_conversion():
    multi = RAGPipeline._build_filter({"source_lang": "en", "target_lang": "es"})
    assert multi == {
        "$and": [
            {"source_lang": {"$eq": "en"}},
            {"target_lang": {"$eq": "es"}},
        ]
    }
    single = RAGPipeline._build_filter({"lang_en": True})
    assert single == {"lang_en": {"$eq": True}}
