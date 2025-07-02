"""Translation RAG system using Fireworks via LangChain and ChromaDB."""
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class TranslationRAG:
    """Translation RAG system with ChromaDB vector store."""
    
    def __init__(self):
        """Initialize the Translation RAG system."""
        self.setup_environment()
        self.setup_llm()
        self.setup_vectorstore()
        self.setup_retrieval_chain()
    
    def setup_environment(self):
        """Setup environment variables and validate configuration."""
        # Validate required environment variables
        required_vars = ['FIREWORKS_API_KEY', 'FIREWORKS_BASE_URL', 'MODEL_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Set up logging
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        print(f"Log level set to: {log_level}")
        
        # Print configuration
        print(f"Model: {os.getenv('MODEL_NAME')}")
        print(f"Max tokens: {os.getenv('MAX_TOKENS', '4096')}")
        print(f"Reflection enabled: {os.getenv('ENABLE_REFLECTION', 'false')}")
    
    def setup_llm(self):
        """Setup the Fireworks LLM."""
        self.llm = ChatFireworks(
            model=os.getenv('MODEL_NAME'),
            max_tokens=int(os.getenv('MAX_TOKENS', 4096)),
            temperature=0.7,
            fireworks_api_key=os.getenv('FIREWORKS_API_KEY'),
            fireworks_api_base=os.getenv('FIREWORKS_BASE_URL')
        )
        print("✓ Fireworks LLM configured")
    
    def setup_vectorstore(self):
        """Setup ChromaDB vector store."""
        self.vectorstore = None
        self.persist_directory = "./chroma_db"
        print("✓ ChromaDB configuration ready")
    
    def setup_retrieval_chain(self):
        """Setup the retrieval chain."""
        # Translation-specific prompt template
        self.prompt_template = PromptTemplate(
            template="""You are an expert translator. Use the following context to help with translation tasks.
            
            Context: {context}
            
            Question: {question}
            
            Provide accurate translations and explain any cultural nuances or alternative translations when relevant.
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        print("✓ Retrieval chain configured")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to the vector store."""
        if not texts:
            print("No texts provided to add to vector store")
            return
        
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        documents = []
        for i, text in enumerate(texts):
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        # Try to use ChromaDB with embeddings
        try:
            # Try HuggingFace embeddings first
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
            print(f"✓ Added {len(documents)} document chunks to ChromaDB with HuggingFace embeddings")
        except ImportError:
            print("HuggingFace embeddings not available, trying basic setup...")
            try:
                # Fallback to basic ChromaDB setup
                from langchain_community.embeddings import FakeEmbeddings
                embeddings = FakeEmbeddings(size=384)
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                self.vectorstore.persist()
                print(f"✓ Added {len(documents)} document chunks to ChromaDB with fake embeddings")
            except Exception as e:
                print(f"Warning: Could not setup ChromaDB: {e}")
                print("Falling back to simple text storage")
                self.documents = documents
    
    def query(self, question: str, use_rag: bool = True) -> str:
        """Query the system with or without RAG."""
        if use_rag and self.vectorstore:
            # Use RAG with vector store
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            response = qa_chain.run(question)
        elif use_rag and hasattr(self, 'documents'):
            # Simple fallback: use document context
            context = "\n\n".join([doc.page_content for doc in self.documents[:3]])
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            response = response.content if hasattr(response, 'content') else str(response)
        else:
            # Direct LLM query without RAG
            response = self.llm.invoke(question)
            response = response.content if hasattr(response, 'content') else str(response)
        
        return response
    
    def add_translation_examples(self):
        """Add some example translation documents."""
        example_texts = [
            """
            English: Hello, how are you?
            Spanish: Hola, ¿cómo estás?
            French: Bonjour, comment allez-vous?
            German: Hallo, wie geht es dir?
            Italian: Ciao, come stai?
            """,
            """
            English: Thank you very much.
            Spanish: Muchas gracias.
            French: Merci beaucoup.
            German: Vielen Dank.
            Italian: Grazie mille.
            """,
            """
            English: Where is the bathroom?
            Spanish: ¿Dónde está el baño?
            French: Où sont les toilettes?
            German: Wo ist die Toilette?
            Italian: Dove si trova il bagno?
            """,
            """
            English: I would like to order food.
            Spanish: Me gustaría pedir comida.
            French: J'aimerais commander de la nourriture.
            German: Ich möchte Essen bestellen.
            Italian: Vorrei ordinare del cibo.
            """
        ]
        
        metadatas = [
            {"type": "greeting", "languages": ["en", "es", "fr", "de", "it"]},
            {"type": "courtesy", "languages": ["en", "es", "fr", "de", "it"]},
            {"type": "question", "languages": ["en", "es", "fr", "de", "it"]},
            {"type": "request", "languages": ["en", "es", "fr", "de", "it"]}
        ]
        
        self.add_documents(example_texts, metadatas)
        print("✓ Added example translation documents")


def main():
    """Main function to run the Translation RAG system."""
    try:
        # Initialize the RAG system
        rag = TranslationRAG()
        
        # Add example translation data
        rag.add_translation_examples()
        
        if len(sys.argv) < 2:
            print("\nUsage:")
            print("  python rag.py '<translation_query>'")
            print("  python rag.py '<translation_query>' --no-rag")
            print("\nExamples:")
            print("  python rag.py 'How do you say goodbye in Spanish?'")
            print("  python rag.py 'Translate I love you to French'")
            sys.exit(1)
        
        query = sys.argv[1]
        use_rag = "--no-rag" not in sys.argv
        
        print(f"\nQuery: {query}")
        print(f"Using RAG: {use_rag}")
        print("-" * 50)
        
        # Query the system
        response = rag.query(query, use_rag=use_rag)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
