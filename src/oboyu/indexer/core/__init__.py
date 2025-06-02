"""Core indexer components."""

from oboyu.indexer.core.document_chunker import DocumentChunker
from oboyu.indexer.core.document_processor import Chunk, DocumentProcessor
from oboyu.indexer.core.embedding_prefix_handler import EmbeddingPrefixHandler
from oboyu.indexer.core.language_processor import LanguageProcessor
from oboyu.indexer.core.text_normalizer import TextNormalizer
__all__ = [
    "Chunk",
    "DocumentChunker",
    "DocumentProcessor",
    "EmbeddingPrefixHandler",
    "LanguageProcessor",
    "TextNormalizer",
]
