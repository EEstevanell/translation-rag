"""Unified Translation RAG system built on the reusable pipeline."""
import sys
import warnings
import json
from requests.exceptions import RequestsDependencyWarning

# Silence noisy warning when charset detection libs are missing
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
from typing import List, Optional

from .config import Config
from .utils import (
    get_supported_languages,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)
from .pipeline import RAGPipeline, create_llm, get_embeddings
from .logging_utils import get_logger
from langchain.prompts import PromptTemplate

class TranslationRAG:
    """Translation RAG system with ChromaDB integration."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Translation RAG system."""
        Config.validate()
        self.logger = get_logger()
        self.config = Config
        self.setup_pipeline()
        # Load existing ChromaDB state if available
        self.pipeline.load_vectorstore()
        self.vectorstore = self.pipeline.vectorstore
    
    def setup_pipeline(self) -> None:
        """Create the reusable RAG pipeline."""
        llm = create_llm(
            self.config.MODEL_NAME,
            self.config.FIREWORKS_API_KEY,
            self.config.FIREWORKS_BASE_URL,
            self.config.MAX_TOKENS,
        )
        self.llm = llm
        embeddings = get_embeddings(self.config.EMBEDDING_MODEL)
        self.pipeline = RAGPipeline(
            llm,
            embeddings,
            self.config.CHROMA_PERSIST_DIR,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )

        # Prompt template simply injects retrieved examples into the pre-rendered
        # translation prompt provided at query time.
        self.pipeline.prompt_template = PromptTemplate(
            template="{question}\n\nContext Examples:\n{context}",
            input_variables=["context", "question"],
        )
        print("✓ RAG pipeline configured")
    
    def load_translation_data(self):
        """Load translation data from seed memory only."""
        # Try loading an existing vector store to avoid reseeding each run
        if self.pipeline.load_vectorstore():
            self.vectorstore = self.pipeline.vectorstore
            print("✓ Loaded existing translation memory from ChromaDB")
            return

        # Load only from seed memory
        from .translation_memory import load_fake_memory, memory_to_documents

        seed_entries = load_fake_memory()
        if seed_entries:
            # Convert seed memory entries to documents for the vector store
            # This stores only the source sentences for semantic matching
            # while keeping the target sentences in metadata
            documents, metadatas = memory_to_documents(seed_entries)
            self.add_documents(documents, metadatas)
            print(f"✓ Added {len(documents)} translation examples from seed memory")
        else:
            print("⚠️  No seed memory found. Please check the seed_memory directory.")
            # Add basic examples as fallback
            basics, basic_meta = self.get_basic_examples()
            self.add_documents(basics, basic_meta)
            print("✓ Added basic translation examples as fallback")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to the underlying pipeline."""
        self.pipeline.add_documents(texts, metadatas)
        self.vectorstore = self.pipeline.vectorstore
    
    def get_basic_examples(self) -> tuple[List[str], List[dict]]:
        """Return a list of basic translation examples and metadata."""
        basic_examples = [
            """Context: Common Greetings (Formality: informal)
English: Hello, how are you?
Spanish: Hola, ¿cómo estás?
French: Bonjour, comment ça va?
German: Hallo, wie geht's?
Italian: Ciao, come stai?""",
            
            """Context: Expressing Gratitude (Formality: neutral)
English: Thank you very much
Spanish: Muchas gracias
French: Merci beaucoup
German: Vielen Dank
Italian: Grazie mille""",
            
            """Context: Asking for Directions (Formality: neutral)
English: Where is the bathroom?
Spanish: ¿Dónde está el baño?
French: Où sont les toilettes?
German: Wo ist die Toilette?
Italian: Dove si trova il bagno?""",
            
            """Context: Business Communication (Formality: formal)
English: I would like to schedule a meeting
Spanish: Me gustaría programar una reunión
French: J'aimerais programmer une réunion
German: Ich möchte einen Termin vereinbaren
Italian: Vorrei programmare una riunione"""
        ]
        
        metadatas = [
            {"type": "greeting", "context": "Common Greetings", "formality": "informal"},
            {"type": "courtesy", "context": "Expressing Gratitude", "formality": "neutral"},
            {"type": "question", "context": "Asking for Directions", "formality": "neutral"},
            {"type": "business", "context": "Business Communication", "formality": "formal"}
        ]
        return basic_examples, metadatas

    def add_basic_examples(self):
        """Add basic translation examples as fallback."""
        examples, metas = self.get_basic_examples()
        self.add_documents(examples, metas)
        print("✓ Added basic translation examples")
    
    def query(self, question: str, use_rag: bool = True, k: int = 4) -> str:
        """Query the system with enhanced response handling.

        Parameters
        ----------
        question: str
            User query or translation request.
        use_rag: bool, optional
            Whether to use the retrieval augmented generation pipeline.
        k: int, optional
            Number of documents to retrieve from the vector store.
        """
        if use_rag:
            from .utils import detect_language
            lang = detect_language(question)
            # No metadata filtering for general queries - let semantic search handle it
            metadata_filter = None
            response = self.pipeline.query(
                question, use_rag=True, k=k, metadata_filter=metadata_filter
            )
        else:
            enhanced_prompt = (
                f"You are an expert multilingual translator.\n\n"
                f"Supported Languages: {', '.join(get_supported_languages().values())}\n\n"
                f"Translation Request: {question}\n\n"
                "Please provide accurate translations and explain any cultural nuances when relevant."
            )
            response = self.pipeline.llm.invoke(enhanced_prompt)
            response = response.content if hasattr(response, "content") else str(response)
        return response

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        system_message: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        use_rag: bool = True,
        k: int = 4,
    ) -> str:
        """Translate text using the underlying pipeline.

        Parameters
        ----------
        text: str
            The text to translate.
        target_lang: str
            Target language code.
        source_lang: str, optional
            Source language code. If not provided, language will be detected.
        system_message: str, optional
            Custom system prompt template.
        use_rag: bool, optional
            Whether to use retrieval augmented generation.
        k: int, optional
            Number of documents to retrieve for RAG context.
        """
        from .utils import detect_language, render_translation_prompt

        src_lang = source_lang or detect_language(text)
        context = None
        
        # Retrieve examples if using RAG
        if use_rag and self.vectorstore:
            metadata_filter = None
            if src_lang != "unknown":
                # Filter by source and target language to get relevant examples
                metadata_filter = {"source_lang": src_lang, "target_lang": target_lang}
            
            # Use similarity search with scores to get similarity information
            filter_dict = self.pipeline._build_filter(metadata_filter) if metadata_filter else None
            results_with_scores = self.vectorstore.similarity_search_with_score(
                text, k=k, filter=filter_dict
            )
            
            if results_with_scores:
                self.logger.info(f"Retrieved {len(results_with_scores)} candidates from vector search:")
                
                # Apply similarity threshold filtering (same logic as in pipeline.py)
                filtered_results = []
                for i, (doc, score) in enumerate(results_with_scores):
                    # Convert distance to similarity (1 - normalized_distance)
                    similarity = 1 - score if score <= 1 else 0
                    preview = doc.page_content.replace("\n", " ")[:60]
                    
                    self.logger.info(f"  [{i+1}] Similarity: {similarity:.3f} | {preview}...")
                    
                    if similarity >= Config.SIMILARITY_THRESHOLD:
                        filtered_results.append(doc)
                        self.logger.debug(f"✓ Accepted (above threshold {Config.SIMILARITY_THRESHOLD:.3f})")
                    else:
                        self.logger.info(f"  ✗ Filtered out (below threshold {Config.SIMILARITY_THRESHOLD:.3f})")
                
                retrieved = filtered_results
                
                if not retrieved:
                    self.logger.info(f"No documents above similarity threshold {Config.SIMILARITY_THRESHOLD:.3f}")
                    context = None
                else:
                    self.logger.info(f"Using {len(retrieved)} documents above threshold for context")
                    # Build context string using retrieved documents
                    context_parts = []
                    for doc in retrieved:
                        tgt = doc.metadata.get("target_sentence")
                        if tgt:
                            context_parts.append(f"{doc.page_content} -> {tgt}")
                        else:
                            context_parts.append(doc.page_content)
                    context = "\n".join(context_parts)
            else:
                self.logger.info("No relevant documents retrieved")
                context = None

        system_msg = render_system_prompt(src_lang, target_lang, system_message)
        prompt = render_translation_prompt(text, src_lang, target_lang, system_msg, context)
        
        # Use the pipeline without RAG since we already have the context
        return self.pipeline.query(prompt, use_rag=False)
    
    def display_stats(self):
        """Display system statistics."""
        print("\n" + "="*50)
        print("Translation RAG System Stats")
        print("="*50)
        
        if self.pipeline.vectorstore:
            try:
                collection = self.pipeline.vectorstore._collection
                count = collection.count()
                print(f"Documents in ChromaDB: {count}")
            except Exception:
                print("ChromaDB: Active (stats unavailable)")
        else:
            print("No documents loaded")
        
        print(f"Supported Languages: {len(get_supported_languages())}")
        print(f"ChromaDB Directory: {self.pipeline.persist_directory}")
        
        self.config.display()
        print("="*50)


def main():
    """Command line interface for the Translation RAG system."""
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "--help":
            print("\nTranslation RAG System Help")
            print("===========================")
            print("This system uses RAG (Retrieval Augmented Generation) to provide")
            print("accurate translations with cultural context.")
            print("\nUsage:")
            print("  python -m translation_rag '<text>' --from SRC --to TGT [--system MESSAGE] [--k NUM]")
            print("       (MESSAGE can use {{ source_lang }} and {{ target_lang }} placeholders)")
            print("  python -m translation_rag '<translation_query>' --no-rag")
            print("  python -m translation_rag --stats")
            print("  python -m translation_rag --seed")
            print("  python -m translation_rag --help")
            sys.exit(0)

        if len(sys.argv) >= 2 and sys.argv[1] == "--seed":
            rag = TranslationRAG()
            rag.load_translation_data()
            print("\u2713 Seeded example translations into ChromaDB")
            sys.exit(0)

        rag = TranslationRAG()
        if not rag.vectorstore:
            print("⚠️  No translation memory found. Run 'python -m translation_rag --seed' to load sample data.")

        target_lang = None
        source_lang = None
        system_message = DEFAULT_SYSTEM_PROMPT_TEMPLATE
        rag_k = 4
        if "--to" in sys.argv:
            idx = sys.argv.index("--to")
            if idx + 1 < len(sys.argv):
                target_lang = sys.argv[idx + 1]
                del sys.argv[idx : idx + 2]
        if "--from" in sys.argv:
            idx = sys.argv.index("--from")
            if idx + 1 < len(sys.argv):
                source_lang = sys.argv[idx + 1]
                del sys.argv[idx : idx + 2]
        if "--system" in sys.argv:
            idx = sys.argv.index("--system")
            if idx + 1 < len(sys.argv):
                system_message = sys.argv[idx + 1]
                del sys.argv[idx : idx + 2]
        if "--k" in sys.argv:
            idx = sys.argv.index("--k")
            if idx + 1 < len(sys.argv):
                try:
                    rag_k = int(sys.argv[idx + 1])
                except ValueError:
                    raise ValueError("--k requires an integer value")
                del sys.argv[idx : idx + 2]

        if len(sys.argv) < 2:
            print("\nTranslation RAG System")
            print("=====================")
            print("\nUsage:")
            print("  python -m translation_rag '<text>' --from SRC --to TGT [--system MESSAGE] [--k NUM]")
            print("       (MESSAGE can use {{ source_lang }} and {{ target_lang }} placeholders)")
            print("  python -m translation_rag '<translation_query>' --no-rag")
            print("  python -m translation_rag --stats")
            print("  python -m translation_rag --seed")
            print("  python -m translation_rag --help")
            print("\nExamples:")
            print("  python -m translation_rag 'Hello' --from en --to es")
            print("  python -m translation_rag 'Guten Morgen' --from de --to en --system 'You are a polite translator.'")
            print("  python -m translation_rag 'Bonjour' --from fr --to it")
            sys.exit(1)

        if sys.argv[1] == "--stats":
            rag.display_stats()
            sys.exit(0)

        query = sys.argv[1]
        use_rag = "--no-rag" not in sys.argv

        print(f"\nTranslation Query: {query}")
        if target_lang:
            print(f"Target Language: {target_lang}")
        print(f"Using RAG: {use_rag}")
        if use_rag:
            print(f"Retrieval k: {rag_k}")
        print("-" * 60)

        if target_lang and source_lang:
            response = rag.translate(
                query,
                target_lang,
                source_lang=source_lang,
                system_message=system_message,
                use_rag=use_rag,
                k=rag_k,
            )
        else:
            raise ValueError("Both --from and --to must be provided for translation.")
        print(f"\nResponse:\n{response}")
        rag.logger.info(f"Final response: {response}")

        if "--stats" in sys.argv:
            rag.display_stats()
        
    except KeyboardInterrupt:
        print("\n\nTranslation session interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        if Config.LOG_LEVEL == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
