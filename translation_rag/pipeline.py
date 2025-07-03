"""Utility pipeline for creating a RAG system with Chroma and LangChain."""

from __future__ import annotations

from typing import List, Optional
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_fireworks import ChatFireworks
from langchain_community.vectorstores import Chroma

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
            return

        first_batch = documents[:batch_size]
        self.vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        for i in range(batch_size, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            self.vectorstore.add_documents(batch_docs)

        self.vectorstore.persist()

    def query(
        self,
        question: str,
        use_rag: bool = True,
        k: int = 3,
        metadata_filter: Optional[dict] = None,
    ) -> str:
        """Query the pipeline with optional metadata filtering."""
        if use_rag and self.vectorstore:
            search_kwargs = {"k": k}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
            )
            return qa_chain.run(question)

        response = self.llm.invoke(question)
        return response.content if hasattr(response, "content") else str(response)

