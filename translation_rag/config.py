"""Configuration settings for the Translation RAG system."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

class Config:
    """Configuration class for Translation RAG system."""
    
    # Fireworks API Configuration
    FIREWORKS_API_KEY = os.getenv('FIREWORKS_API_KEY')
    FIREWORKS_BASE_URL = os.getenv('FIREWORKS_BASE_URL', 'https://api.fireworks.ai/inference/v1')
    MODEL_NAME = os.getenv('MODEL_NAME', 'accounts/fireworks/models/llama4-scout-instruct-basic')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4096))
    
    # RAG Configuration
    CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    
    # System Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_REFLECTION = os.getenv('ENABLE_REFLECTION', 'false').lower() == 'true'
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    @classmethod
    def validate(cls):
        """Validate required configuration values."""
        required_vars = ['FIREWORKS_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required configuration: {', '.join(missing_vars)}")
        
        return True
    
    @classmethod
    def display(cls):
        """Display current configuration."""
        print("Current Configuration:")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Max Tokens: {cls.MAX_TOKENS}")
        print(f"  Chroma DB: {cls.CHROMA_PERSIST_DIR}")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Log Level: {cls.LOG_LEVEL}")
        print(f"  Reflection: {cls.ENABLE_REFLECTION}")
