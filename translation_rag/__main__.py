"""CLI entry point for Translation RAG."""
import warnings
from requests.exceptions import RequestsDependencyWarning

# Suppress warnings from the requests package before other imports
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

from .cli import main

if __name__ == "__main__":
    main()
