"""Knowledge Graph extraction service interface.

This module defines the port interface for LLM-based knowledge graph extraction
from text chunks, following the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from oboyu.domain.models.knowledge_graph import KnowledgeGraphExtraction


class KGExtractionService(ABC):
    """Abstract interface for knowledge graph extraction from text."""

    @abstractmethod
    async def extract_knowledge_graph(
        self,
        text: str,
        chunk_id: str,
        language: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> KnowledgeGraphExtraction:
        """Extract entities and relations from text.

        Args:
            text: The text content to process
            chunk_id: Unique identifier for the source chunk
            language: Language code (e.g., 'ja', 'en') for language-specific processing
            entity_types: List of allowed entity types to extract
            relation_types: List of allowed relation types to extract

        Returns:
            KnowledgeGraphExtraction containing extracted entities and relations

        Raises:
            ExtractionError: If extraction fails

        """

    @abstractmethod
    async def batch_extract_knowledge_graph(
        self,
        texts_and_ids: List[tuple[str, str]],
        language: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> List[KnowledgeGraphExtraction]:
        """Extract knowledge graphs from multiple texts in batch.

        Args:
            texts_and_ids: List of (text, chunk_id) tuples
            language: Language code for processing
            entity_types: List of allowed entity types
            relation_types: List of allowed relation types

        Returns:
            List of KnowledgeGraphExtraction results

        Raises:
            ExtractionError: If batch extraction fails

        """

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if the LLM model is loaded and ready.

        Returns:
            True if model is ready for inference

        """


class ExtractionError(Exception):
    """Exception raised when knowledge graph extraction fails."""

    def __init__(self, message: str, chunk_id: Optional[str] = None) -> None:
        """Initialize extraction error.

        Args:
            message: Error description
            chunk_id: Optional chunk ID where error occurred

        """
        super().__init__(message)
        self.chunk_id = chunk_id
