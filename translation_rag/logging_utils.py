from __future__ import annotations

"""Logging setup for the Translation RAG system."""

import sys
from pathlib import Path
from loguru import logger

from .config import Config

# Determine log level and log file path
_LOG_LEVEL = Config.LOG_LEVEL.upper()
_LOG_FILE = Path("rag.log")

# Reconfigure logger
logger.remove()
logger.add(sys.stderr, level=_LOG_LEVEL)
logger.add(_LOG_FILE, rotation="1 MB", level=_LOG_LEVEL)

# Convenience function for external modules
get_logger = lambda: logger
