# Translation RAG

This project implements a comprehensive Retrieval Augmented Generation (RAG) system for multilingual translation tasks using Fireworks AI and ChromaDB. The system provides contextually aware translations with cultural nuances and supports multiple formality levels.

## Features

- **Multi-language Support**: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Russian, Arabic, Hindi
- **Cultural Context Awareness**: Provides cultural nuances and alternative translations
- **ChromaDB Integration**: Vector storage for efficient similarity search
- **Fireworks AI**: Powered by advanced language models
- **Flexible Embedding Options**: HuggingFace, SentenceTransformers, or fallback embeddings
- **Reusable Pipeline**: `pipeline.py` provides a consistent RAG setup
- **Environment Configuration**: Configurable via `.env` file
- **Multiple Usage Modes**: With or without RAG, CLI interface

## Project Structure

```
translation-rag/
├── .env                    # Environment variables
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── pipeline.py            # Reusable RAG pipeline
├── rag.py                 # Translation RAG entry point
├── translation_data.json  # Sample translation data (auto-generated)
├── environment.yml        # Conda environment specification
├── chroma_db/            # ChromaDB persistent storage (auto-created)
└── tests/                # Test files
```

## Setup

### 1. Environment Setup

Create and activate the conda environment:

```bash
# Create the environment
conda env create -f environment.yml

# Activate it
conda activate translation-rag
```

### 2. Environment Variables

The system uses the provided `.env` file with the following configuration:

```properties
FIREWORKS_API_KEY=fw_3ZLfuJfWee5j68B4dGftTC3U
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
MODEL_NAME=accounts/fireworks/models/llama4-scout-instruct-basic
MAX_TOKENS=4096
LOG_LEVEL=INFO
ENABLE_REFLECTION=false
```

Additional optional configuration variables:
```properties
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Dependencies

The environment includes all necessary dependencies:
- `langchain` and `langchain_community` for RAG framework
- `langchain_fireworks` for Fireworks AI integration
- `chromadb` for vector storage
- `sentence-transformers` and `transformers` for embeddings
- `python-dotenv` for environment management

## Usage

### Basic RAG System

```bash
# Basic translation query
python rag.py "How do you say hello in Spanish?"

# Without RAG (direct LLM)
python rag.py "Translate 'goodbye' to French" --no-rag
```

### Context-Aware RAG System

```bash
# Translation with cultural context
python rag.py "What is the formal way to greet someone in German?"

# Show system statistics
python rag.py --stats

# Seed example data
python rag.py --seed

# Help information
python rag.py --help
```

### Example Queries

```bash
# Basic translations
python rag.py "How do you say 'thank you' in Italian?"

# Cultural context
python rag.py "What's the difference between formal and informal greetings in French?"

# Business translations
python rag.py "How do I politely decline a meeting in Spanish?"

# Multiple languages
python rag.py "Translate 'I love you' to German, Italian, and Portuguese"
```

## Features in Detail

### RAG System Components

1. **Document Processing**: Text splitting and chunking for optimal retrieval
2. **Vector Storage**: ChromaDB with persistent storage
3. **Embedding Models**: Multiple fallback options for embeddings
4. **Retrieval Chain**: Context-aware prompt templates
5. **LLM Integration**: Fireworks AI for generation

### Translation Context

The system provides:
- **Formality Levels**: Formal, informal, neutral
- **Cultural Nuances**: Regional variations and cultural context
- **Alternative Translations**: Multiple valid options
- **Context Categories**: Greetings, business, travel, courtesy, etc.

### Data Management

- **Sample Data**: Auto-generated translation examples
- **Custom Data**: Load your own translation datasets
- **Persistent Storage**: ChromaDB maintains vector embeddings
- **Configuration**: Flexible settings via environment variables

## Architecture

```
User Query
    ↓
Translation RAG System
    ↓
1. Document Retrieval (ChromaDB)
    ↓
2. Context Assembly
    ↓
3. Prompt Engineering
    ↓
4. Fireworks AI Generation
    ↓
Contextual Translation Response
```

## Development

### Adding New Languages

1. Update `utils.py` `get_supported_languages()` function
2. Add translation examples to `translation_data.json`
3. Update prompt templates in `rag.py`

### Custom Translation Data

Create a `translation_data.json` file with the following structure:

```json
[
  {
    "id": "example_1",
    "type": "greeting",
    "translations": {
      "en": "Hello",
      "es": "Hola",
      "fr": "Bonjour"
    },
    "context": "Common greeting",
    "formality": "informal"
  }
]
```

### Configuration

Modify `config.py` to add new configuration options or update existing ones.

## Troubleshooting

### ChromaDB Issues
```bash
# Clean ChromaDB storage
python -c "from utils import clean_chroma_db; clean_chroma_db()"
```

### Environment Issues
```bash
# Recreate environment
conda env remove -n translation-rag
conda env create -f environment.yml
conda activate translation-rag
```

### Embedding Issues
The system automatically falls back through multiple embedding options:
1. HuggingFace Embeddings
2. SentenceTransformer Embeddings  
3. Fake Embeddings (fallback)

## Testing

Run basic tests:
```bash
cd tests
python -m pytest
```

## Performance Tips

1. **Embedding Models**: Use smaller models for faster performance
2. **Chunk Size**: Adjust based on your translation data
3. **Vector Storage**: Clean ChromaDB periodically for better performance
4. **Batch Processing**: Process multiple translations in one session

## Contributing

1. Add new language support
2. Improve translation context
3. Enhance embedding strategies  
4. Add specialized domain translations
5. Improve cultural context awareness

## License

This project is for educational and research purposes. Please ensure compliance with Fireworks AI terms of service.
