# Translation Memory with RAG - Experimental Repository

This is an experimental repository where we're exploring the integration of Retrieval Augmented Generation (RAG) with translation memory concepts. We're testing how RAG can enhance translation systems by providing contextually relevant examples from a curated translation memory.

**Note: This is a research/testing project and not a production-ready translation system.**

## What We're Exploring

- **Translation Memory + RAG**: Combining traditional translation memory approaches with modern RAG techniques
- **Context-Aware Retrieval**: How well can vector search find relevant translation examples?
- **Cultural Context Integration**: Testing if RAG can surface cultural nuances and formality levels
- **Multi-language Vector Embeddings**: Experimenting with different embedding models for multilingual content
- **Scalable Architecture**: Building a flexible pipeline that could be extended for various translation workflows

## Current Implementation Features

- Multi-language support
- ChromaDB vector storage for translation examples
- Fireworks AI integration for generation
- Flexible embedding model fallbacks
- CLI interface for testing different scenarios

## Repository Structure

```
translation-rag/
├── translation_rag/
│   ├── __init__.py        # Package initializer
│   ├── __main__.py        # Enables `python -m translation_rag`
│   ├── cli.py             # Command line interface
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # Reusable RAG pipeline
│   ├── translation_memory.py  # Translation memory helpers
│   └── utils.py           # Utility functions
├── seed_memory/           # Sample translation memories
│   ├── en_es.json         # English→Spanish pairs
│   ├── es_en.json         # Spanish→English pairs
│   └── sample.json        # Minimal example data
├── environment.yml        # Conda environment specification
├── chroma_db/             # ChromaDB storage (created automatically)
└── tests/                 # Test suite
```

**About seed_memory**: The translation example files in `seed_memory/` were automatically generated using Claude Sonnet 4 to provide diverse, culturally-aware translation pairs for testing our RAG approach.

## Getting Started

### Prerequisites
- Anaconda or Miniconda
- Fireworks AI API access

### Setup

1. **Clone and setup environment**:
```bash
conda env create -f environment.yml
conda activate translation-rag
```

2. **Configure your API key**:
Create a `.env` file with your Fireworks API credentials:
```properties
FIREWORKS_API_KEY=your_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
MODEL_NAME=accounts/fireworks/models/llama4-scout-instruct-basic
```

3. **Test the setup**:
```bash
python -m translation_rag "How do you say hello in Spanish?"
```

4. **Load the sample translation memory** (run once):
```bash
python -m translation_rag --seed
```
This command populates the ChromaDB vector store with a few example translation
pairs stored under `seed_memory/`. Each example records the source and target
languages along with the sentence pair, allowing the RAG system to retrieve
relevant examples for a given language combination.

## Testing Different Approaches

### Basic RAG vs Direct LLM
```bash
# Using RAG (retrieves similar examples from translation memory)
python -m translation_rag "How do you say 'thank you' in Italian?"

# Direct LLM (no retrieval)
python -m translation_rag "How do you say 'thank you' in Italian?" --no-rag
```

You can control how many examples RAG retrieves by passing the `--k` parameter:

```bash
python -m translation_rag "How do you say 'thank you' in Italian?" --k 5
```

### Experimenting with Different Queries
```bash
# Test cultural context awareness
python -m translation_rag "What's the formal way to greet someone in German?"

# Test formality levels
python -m translation_rag "How do I politely decline a meeting in Spanish?"

# Test technical translations
python -m translation_rag "Translate 'machine learning algorithm' to French"
```

### Filtering by language pair
The vector store stores each translation example with its source and target
language codes. Retrieval is performed on the source sentence only and filtered
by the requested language pair so that only relevant examples are considered.
```bash
# Restrict retrieval to a specific source/target pair
python -m translation_rag "How do you say 'thank you' in Italian?" --from en --to es
```

### System Analysis
```bash
# Show retrieval statistics
python -m translation_rag --stats

# Initialize or update the translation memory
python -m translation_rag --seed
```

## What We've Learned So Far

### RAG Integration Benefits
- **Contextual Examples**: RAG successfully retrieves relevant translation pairs that provide context
- **Cultural Nuances**: The system can surface culturally-appropriate alternatives
- **Consistency**: Similar phrases get consistent translations when examples exist

### Current Limitations
- **Cold Start**: Performance depends heavily on the quality of seed translation memory
- **Cross-lingual Embeddings**: Some language pairs work better than others
- **Context Boundaries**: System sometimes retrieves overly similar examples

### Interesting Findings
- ChromaDB performs well with multilingual embeddings for European languages
- Cultural context examples significantly improve translation quality
- The fallback embedding strategy helps with robustness

## Architecture Overview

```
User Query
    ↓
Translation Memory RAG Pipeline
    ↓
1. Vector Similarity Search (ChromaDB)
    ↓
2. Context Assembly (Retrieved Examples)
    ↓
3. Prompt Construction (Query + Examples)
    ↓
4. LLM Generation (Fireworks AI)
    ↓
Enhanced Translation Response
```

## Extending the System

### Adding New Translation Pairs
1. Create new JSON files in `seed_memory/` following the existing format
2. Run `python -m translation_rag --seed` to update the vector database

### Testing New Embedding Models
Update the embedding configuration in `config.py` or environment variables:
```properties
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

### Experimenting with Different Retrievers
The `translation_rag.pipeline` module provides a flexible base for trying different retrieval strategies.

## Development Notes

### Key Dependencies
- **LangChain**: RAG framework and document processing
- **ChromaDB**: Vector storage with persistence
- **Fireworks AI**: LLM provider
- **SentenceTransformers**: Multilingual embeddings

### Testing
Run the test suite to validate changes:
```bash
python -m pytest tests/
```

### Common Issues
- **ChromaDB Persistence**: If you encounter database locks, delete the `chroma_db/` directory
- **Embedding Failures**: The system falls back through multiple embedding providers automatically
- **API Limits**: Fireworks AI has rate limits; consider adding delays for batch processing
