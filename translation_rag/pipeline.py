"""Utility pipeline for creating a RAG system with Chroma and LangChain."""

from __future__ import annotations

from typing import List, Optional
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_fireworks import ChatFireworks
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from .logging_utils import get_logger

# Embedding imports happen lazily in ``get_embeddings`` to avoid heavy
# dependencies during tests.


def get_embeddings(model_name: str):
    """Return an embedding instance with multiple fallbacks."""
    # Prefer Fireworks embeddings when API key is available
    try:
        from langchain_fireworks import FireworksEmbeddings

        if os.getenv("FIREWORKS_API_KEY"):
            return FireworksEmbeddings(model=model_name)
    except Exception:
        pass

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    except Exception:
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings

            return SentenceTransformerEmbeddings(model_name=model_name)
        except Exception:
            from langchain_community.embeddings import FakeEmbeddings

            return FakeEmbeddings(size=384)


def create_llm(model: str, api_key: str, base_url: str, max_tokens: int):
    """Create the Fireworks chat model."""
    return ChatFireworks(
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        fireworks_api_key=api_key,
        fireworks_api_base=base_url,
    )


class RAGPipeline:
    """Simple wrapper around a LangChain retrieval QA pipeline."""

    def __init__(
        self,
        llm,
        embeddings,
        persist_directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.logger = get_logger()
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        self.prompt_template = PromptTemplate(
            template=(
                "You are an expert translator. Use the following context to help "
                "with translation tasks.\n\nContext: {context}\n\nQuestion: {question}\n\n"
                "Provide accurate translations and explain any cultural nuances when relevant.\n\nAnswer:"
            ),
            input_variables=["context", "question"],
        )

    def load_vectorstore(self) -> bool:
        """Load an existing Chroma vector store if present."""
        if not os.path.isdir(self.persist_directory):
            return False
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                    persist_directory=self.persist_directory,
                ),
            )
            # Trigger collection to check that data exists
            count = self.vectorstore._collection.count()
            if count:
                self.logger.info(f"Loaded {count} documents from existing vector store")
                return True
        except Exception as exc:
            self.logger.debug(f"Could not load existing vector store: {exc}")
        self.vectorstore = None
        return False

    @staticmethod
    def _build_filter(metadata_filter: Optional[dict]) -> Optional[dict]:
        """Convert simple key/value filters to the Chroma structured format."""
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            key, value = next(iter(metadata_filter.items()))
            return {key: {"$eq": value}}
        return {
            "$and": [{k: {"$eq": v}} for k, v in metadata_filter.items()]
        }

    def add_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None, batch_size: int = 128
    ) -> None:
        """Add texts to the vector store.

        Fireworks embeddings have a batch limit of 256 rows, so we add
        documents in manageable batches to avoid errors when a large
        number of texts is supplied.
        """
        documents: List[Document] = []
        for i, text in enumerate(texts):
            for chunk in self.text_splitter.split_text(text):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=chunk, metadata=metadata))

        if not documents:
            self.logger.debug("No documents to add to the vector store")
            return

        first_batch = documents[:batch_size]
        self.vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            client_settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
                persist_directory=self.persist_directory,
            ),
        )
        for i in range(batch_size, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            self.vectorstore.add_documents(batch_docs)

        
        self.logger.info(f"Added {len(documents)} documents to vector store")
        # Since Chroma 0.4.x documents are automatically persisted
        # when using a persistent directory.

    def query(
        self,
        question: str,
        use_rag: bool = True,
        k: int = 3,
        metadata_filter: Optional[dict] = None,
    ) -> str:
        """Query the pipeline with optional metadata filtering."""
        self.logger.info(f"Query: {question} | use_rag={use_rag}")
        if use_rag and self.vectorstore:
            search_kwargs = {"k": k}
            if metadata_filter:
                search_kwargs["filter"] = self._build_filter(metadata_filter)
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

            # Retrieve and log documents for debugging
            retrieved = retriever.get_relevant_documents(question)
            if retrieved:
                for doc in retrieved:
                    preview = doc.page_content.replace("\n", " ")[:80]
                    self.logger.debug(f"Retrieved: {preview} | meta={doc.metadata}")
            else:
                self.logger.debug("No relevant documents retrieved")

            # Build context string using retrieved documents. If a target sentence
            # is present in metadata, include it so the LLM sees the full
            # translation pair, while embeddings remain based only on the source
            # sentence stored in the vector database.
            context_parts = []
            for doc in retrieved:
                tgt = doc.metadata.get("target_sentence")
                if tgt:
                    context_parts.append(f"{doc.page_content} -> {tgt}")
                else:
                    context_parts.append(doc.page_content)
            context = "\n".join(context_parts)

            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, "content") else str(response)
            self.logger.debug(f"RAG response: {result}")
            return result

        response = self.llm.invoke(question)
        result = response.content if hasattr(response, "content") else str(response)
        self.logger.debug(f"LLM response: {result}")
        return result

