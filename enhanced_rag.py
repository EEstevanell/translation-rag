"""Enhanced RAG system with better ChromaDB integration."""
import os
import sys
from typing import List, Optional
from config import Config
from utils import (
    load_translation_data, 
    format_translation_examples, 
    setup_sample_data_file,
    get_supported_languages
)
from langchain_fireworks import ChatFireworks
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate


class EnhancedTranslationRAG:
    """Enhanced Translation RAG system with better ChromaDB integration."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Enhanced Translation RAG system."""
        Config.validate()
        self.config = Config
        self.setup_llm()
        self.setup_vectorstore()
        self.setup_retrieval_chain()
        self.load_translation_data()
    
    def setup_llm(self):
        """Setup the Fireworks LLM."""
        self.llm = ChatFireworks(
            model=self.config.MODEL_NAME,
            max_tokens=self.config.MAX_TOKENS,
            temperature=0.7,
            fireworks_api_key=self.config.FIREWORKS_API_KEY,
            fireworks_api_base=self.config.FIREWORKS_BASE_URL
        )
        print("✓ Fireworks LLM configured")
    
    def setup_vectorstore(self):
        """Setup ChromaDB vector store with proper embeddings."""
        self.vectorstore = None
        self.persist_directory = self.config.CHROMA_PERSIST_DIR
        print("✓ ChromaDB configuration ready")
    
    def setup_retrieval_chain(self):
        """Setup the retrieval chain with enhanced prompts."""
        # Enhanced translation-specific prompt template
        self.prompt_template = PromptTemplate(
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
            input_variables=["context", "question"]
        )
        print("✓ Enhanced retrieval chain configured")
    
    def load_translation_data(self):
        """Load translation data from file or create sample data."""
        # Ensure sample data file exists
        setup_sample_data_file()
        
        # Load translation data
        data = load_translation_data("translation_data.json")
        if data:
            formatted_examples = format_translation_examples(data)
            metadatas = [
                {
                    "type": item.get("type", "general"),
                    "languages": list(item.get("translations", {}).keys()),
                    "context": item.get("context", "General"),
                    "formality": item.get("formality", "neutral")
                }
                for item in data
            ]
            self.add_documents(formatted_examples, metadatas)
            print(f"✓ Loaded {len(data)} translation examples from file")
        else:
            # Fallback to basic examples
            self.add_basic_examples()
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to the vector store with improved embedding handling."""
        if not texts:
            print("No texts provided to add to vector store")
            return
        
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        documents = []
        for i, text in enumerate(texts):
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        # Try different embedding approaches
        embedding_success = False
        
        # Method 1: Try HuggingFace embeddings
        if not embedding_success:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                self.vectorstore.persist()
                print(f"✓ Added {len(documents)} document chunks to ChromaDB with HuggingFace embeddings")
                embedding_success = True
            except Exception as e:
                print(f"HuggingFace embeddings failed: {e}")
        
        # Method 2: Try Sentence Transformers
        if not embedding_success:
            try:
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                embeddings = SentenceTransformerEmbeddings(model_name=self.config.EMBEDDING_MODEL)
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                self.vectorstore.persist()
                print(f"✓ Added {len(documents)} document chunks to ChromaDB with SentenceTransformer embeddings")
                embedding_success = True
            except Exception as e:
                print(f"SentenceTransformer embeddings failed: {e}")
        
        # Method 3: Fallback to simple embeddings
        if not embedding_success:
            try:
                from langchain_community.embeddings import FakeEmbeddings
                embeddings = FakeEmbeddings(size=384)
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                self.vectorstore.persist()
                print(f"✓ Added {len(documents)} document chunks to ChromaDB with fallback embeddings")
                embedding_success = True
            except Exception as e:
                print(f"Fallback embeddings failed: {e}")
        
        # Final fallback: Store documents in memory
        if not embedding_success:
            print("Warning: Could not setup any embeddings, using in-memory storage")
            self.documents = documents
    
    def add_basic_examples(self):
        """Add basic translation examples as fallback."""
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
        
        self.add_documents(basic_examples, metadatas)
        print("✓ Added basic translation examples")
    
    def query(self, question: str, use_rag: bool = True) -> str:
        """Query the system with enhanced response handling."""
        if use_rag and self.vectorstore:
            # Use RAG with vector store
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Get more examples for better context
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            response = result["result"]
            
            # Optionally show source documents
            if self.config.LOG_LEVEL == "DEBUG":
                print("\nSource Documents:")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"  {i+1}. {doc.page_content[:100]}...")
            
        elif use_rag and hasattr(self, 'documents'):
            # Simple fallback: use document context
            context = "\n\n".join([doc.page_content for doc in self.documents[:4]])
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            response = response.content if hasattr(response, 'content') else str(response)
        else:
            # Direct LLM query without RAG
            enhanced_prompt = f"""You are an expert multilingual translator. 
            
            Supported Languages: {', '.join(get_supported_languages().values())}
            
            Translation Request: {question}
            
            Please provide accurate translations and explain any cultural nuances when relevant."""
            
            response = self.llm.invoke(enhanced_prompt)
            response = response.content if hasattr(response, 'content') else str(response)
        
        return response
    
    def display_stats(self):
        """Display system statistics."""
        print("\n" + "="*50)
        print("Translation RAG System Stats")
        print("="*50)
        
        if self.vectorstore:
            try:
                # Get collection info
                collection = self.vectorstore._collection
                count = collection.count()
                print(f"Documents in ChromaDB: {count}")
            except:
                print("ChromaDB: Active (stats unavailable)")
        elif hasattr(self, 'documents'):
            print(f"Documents in memory: {len(self.documents)}")
        else:
            print("No documents loaded")
        
        print(f"Supported Languages: {len(get_supported_languages())}")
        print(f"ChromaDB Directory: {self.persist_directory}")
        
        self.config.display()
        print("="*50)


def main():
    """Main function with enhanced CLI interface."""
    try:
        # Initialize the enhanced RAG system
        rag = EnhancedTranslationRAG()
        
        # Handle different command line options
        if len(sys.argv) < 2:
            print("\nTranslation RAG System")
            print("=====================")
            print("\nUsage:")
            print("  python enhanced_rag.py '<translation_query>'")
            print("  python enhanced_rag.py '<translation_query>' --no-rag")
            print("  python enhanced_rag.py --stats")
            print("  python enhanced_rag.py --help")
            print("\nExamples:")
            print("  python enhanced_rag.py 'How do you say goodbye in Spanish?'")
            print("  python enhanced_rag.py 'Translate I love you to French'")
            print("  python enhanced_rag.py 'What is the formal way to say hello in German?'")
            sys.exit(1)
        
        # Handle special commands
        if sys.argv[1] == "--stats":
            rag.display_stats()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("\nTranslation RAG System Help")
            print("===========================")
            print("This system uses RAG (Retrieval Augmented Generation) to provide")
            print("accurate translations with cultural context.")
            print("\nFeatures:")
            print("- Multi-language support")
            print("- Cultural context awareness")
            print("- Formality level consideration")
            print("- ChromaDB vector storage")
            print("- Fireworks AI integration")
            sys.exit(0)
        
        query = sys.argv[1]
        use_rag = "--no-rag" not in sys.argv
        
        print(f"\nTranslation Query: {query}")
        print(f"Using RAG: {use_rag}")
        print("-" * 60)
        
        # Query the system
        response = rag.query(query, use_rag=use_rag)
        print(f"\nResponse:\n{response}")
        
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
