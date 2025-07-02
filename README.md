# Translation RAG

This project explores Retrieval Augmented Generation (RAG) workflows. A
Conda environment is provided to keep dependencies consistent.

## Setting up the environment

```bash
# Create the environment
conda env create -f environment.yml

# Activate it
conda activate translation-rag
```

This environment installs `langchain_fireworks` so you can connect to the
[Fireworks](https://fireworks.ai) API. Export your API key before running any
scripts:

```bash
export FIREWORKS_API_KEY="<your key>"
```

## Example usage

A simple example is available in `rag.py` and can be run with:

```bash
python rag.py "Translate this text"
```
