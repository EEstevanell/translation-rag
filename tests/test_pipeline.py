from pipeline import RAGPipeline, get_embeddings
from langchain.llms.base import LLM

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
