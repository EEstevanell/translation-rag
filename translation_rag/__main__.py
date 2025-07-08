"""CLI entry point for Translation RAG."""

import warnings
from requests.exceptions import RequestsDependencyWarning

from .cli import main

# Suppress warnings from the requests package before the program runs
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

if __name__ == "__main__":
    main()
