"""Unified Translation RAG system built on the reusable pipeline."""
import sys
import warnings
from requests.exceptions import RequestsDependencyWarning

# Silence noisy warning when charset detection libs are missing
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
from typing import List, Optional

from .config import Config
from .utils import (
    load_translation_data,
    format_translation_examples,
    setup_sample_data_file,
    get_supported_languages,
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
        self.load_translation_data()
    
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

        # Custom prompt for the pipeline
        self.pipeline.prompt_template = PromptTemplate(
            template="""You are an expert multilingual translator with deep cultural knowledge.
            Use the following translation examples and context to help with translation tasks.

            Context Examples:
            {context}

            Translation Request: {question}

            Instructions:
            1. Provide accurate translations
            2. Explain cultural nuances when relevant
            3. Suggest alternative translations if appropriate
            4. Consider formality levels (formal/informal)
            5. Note any regional variations

            Supported Languages: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Russian, Arabic, Hindi

            Response:""",
            input_variables=["context", "question"],
        )
        print("✓ RAG pipeline configured")
    
    def load_translation_data(self):
        """Load translation data from file or create sample data."""
        # Ensure sample data file exists
        setup_sample_data_file()
        
        # Load translation data
        data = load_translation_data("translation_data.json")
        documents: List[str] = []
        metadatas: List[dict] = []
        if data:
            documents.extend(format_translation_examples(data))
            for item in data:
                langs = list(item.get("translations", {}).keys())
                lang_flags = {f"lang_{code}": True for code in langs}
                metadatas.append(
                    {
                        "type": item.get("type", "general"),
                        "context": item.get("context", "General"),
                        "formality": item.get("formality", "neutral"),
                        **lang_flags,
                    }
                )
            print(f"✓ Loaded {len(data)} translation examples from file")
        else:
            basics, basic_meta = self.get_basic_examples()
            documents.extend(basics)
            metadatas.extend(basic_meta)

        from .translation_memory import load_fake_memory, memory_to_documents

        mem_texts, mem_meta = memory_to_documents(load_fake_memory())
        documents.extend(mem_texts)
        metadatas.extend(mem_meta)

        self.add_documents(documents, metadatas)
    
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
    
    def query(self, question: str, use_rag: bool = True) -> str:
        """Query the system with enhanced response handling."""
        if use_rag:
            from .utils import detect_language
            lang = detect_language(question)
            metadata_filter = {f"lang_{lang}": True} if lang != "unknown" else None
            response = self.pipeline.query(
                question, use_rag=True, k=4, metadata_filter=metadata_filter
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
        use_rag: bool = True,
    ) -> str:
        """Translate text using the underlying pipeline."""
        from .utils import detect_language, render_translation_prompt

        src_lang = source_lang or detect_language(text)
        metadata_filter = None
        if src_lang != "unknown":
            metadata_filter = {"source_lang": src_lang, "target_lang": target_lang}

        prompt = render_translation_prompt(text, src_lang, target_lang)
        return self.pipeline.query(
            prompt,
            use_rag=use_rag,
            k=4,
            metadata_filter=metadata_filter,
        )
    
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
            print("  python -m translation_rag '<translation_query>' [--to LANG]")
            print("  python -m translation_rag '<translation_query>' --no-rag")
            print("  python -m translation_rag --stats")
            print("  python -m translation_rag --seed")
            print("  python -m translation_rag --help")
            sys.exit(0)

        if len(sys.argv) >= 2 and sys.argv[1] == "--seed":
            TranslationRAG()
            print("\u2713 Seeded example translations into ChromaDB")
            sys.exit(0)

        rag = TranslationRAG()

        target_lang = None
        if "--to" in sys.argv:
            idx = sys.argv.index("--to")
            if idx + 1 < len(sys.argv):
                target_lang = sys.argv[idx + 1]
                del sys.argv[idx : idx + 2]

        if len(sys.argv) < 2:
            print("\nTranslation RAG System")
            print("=====================")
            print("\nUsage:")
            print("  python -m translation_rag '<translation_query>' [--to LANG]")
            print("  python -m translation_rag '<translation_query>' --no-rag")
            print("  python -m translation_rag --stats")
            print("  python -m translation_rag --seed")
            print("  python -m translation_rag --help")
            print("\nExamples:")
            print("  python -m translation_rag 'How do you say goodbye in Spanish?'")
            print("  python -m translation_rag 'I love you' --to fr")
            print("  python -m translation_rag 'What is the formal way to say hello in German?'")
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
        print("-" * 60)

        if target_lang:
            response = rag.translate(query, target_lang, use_rag=use_rag)
        else:
            response = rag.query(query, use_rag=use_rag)
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
