"""Oboyu - A Japanese-enhanced semantic search system for local documents.

This package provides utilities for creating a semantic search index of text documents,
with special handling for Japanese text.
"""

try:
    from importlib.metadata import version

    __version__ = version("oboyu")
except ImportError:
    # Fallback for development
    __version__ = "0.1.0-dev"
