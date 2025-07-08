"""Logging setup for the Translation RAG system."""

from __future__ import annotations

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

def get_logger() -> "logger":
    """Return the configured logger instance."""
    return logger
