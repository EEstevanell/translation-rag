"""Utility pipeline for creating a RAG system with Chroma and LangChain."""

from __future__ import annotations

from typing import List, Optional
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_fireworks import ChatFireworks
# Chroma vector store is provided by the separate ``langchain-chroma`` package
# as of LangChain 0.2.9. Importing from there avoids deprecation warnings.
from langchain_chroma import Chroma
from chromadb.config import Settings

from .logging_utils import get_logger
from .config import Config

# Embedding imports happen lazily in ``get_embeddings`` to avoid heavy
# dependencies during tests.


def get_embeddings(model_name: str):
    """Return a Fireworks embedding instance."""
    try:
        from langchain_fireworks import FireworksEmbeddings
        
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable is required for embeddings")
        
        print(f"✓ Using Fireworks embeddings with model: {model_name}")
        return FireworksEmbeddings(
            model=model_name,
            api_key=api_key
        )
    except Exception as e:
        print(f"⚠️  Failed to initialize Fireworks embeddings: {e}")
        raise ValueError(f"Could not initialize Fireworks embeddings: {e}")


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
        query_text: Optional[str] = None,
    ) -> str:
        """Query the pipeline with optional metadata filtering.

        Parameters
        ----------
        question: str
            The text or prompt that will be sent to the LLM.
        use_rag: bool, optional
            Whether to retrieve examples from the vector store.
        k: int, optional
            Number of documents to retrieve.
        metadata_filter: dict, optional
            Metadata filter passed to the retriever.
        query_text: str, optional
            Text used for similarity search. If omitted, ``question`` is used.
        """
        self.logger.info(f"Query: {question} | use_rag={use_rag}")
        if use_rag and self.vectorstore:
            # Use similarity search with scores to filter by threshold
            retrieval_query = query_text or question
            results_with_scores = self.vectorstore.similarity_search_with_score(
                retrieval_query, k=k, filter=self._build_filter(metadata_filter) if metadata_filter else None
            )
            
            # Filter results based on similarity threshold
            # Note: Chroma returns distance, where lower values mean higher similarity
            # We need to convert distance to similarity and filter
            filtered_results = []
            for doc, score in results_with_scores:
                # Convert distance to similarity (1 - normalized_distance)
                # For cosine distance, similarity = 1 - distance
                similarity = 1 - score if score <= 1 else 0
                
                self.logger.debug(f"Document similarity: {similarity:.3f} (distance: {score:.3f})")
                
                if similarity >= Config.SIMILARITY_THRESHOLD:
                    filtered_results.append(doc)
                    preview = doc.page_content.replace("\n", " ")[:80]
                    self.logger.debug(f"Retrieved: {preview} | similarity={similarity:.3f} | meta={doc.metadata}")
                else:
                    self.logger.debug(f"Filtered out document with similarity {similarity:.3f} below threshold {Config.SIMILARITY_THRESHOLD}")
            
            retrieved = filtered_results
            
            if not retrieved:
                self.logger.debug(f"No documents above similarity threshold {Config.SIMILARITY_THRESHOLD}")

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

