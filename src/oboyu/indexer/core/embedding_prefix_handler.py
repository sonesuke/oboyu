"""Embedding prefix handling component.

This module manages prefix addition for different embedding models,
particularly for the Ruri v3 embedding model which requires specific
document prefixes for optimal performance.
"""


class EmbeddingPrefixHandler:
    """Manages prefix addition for different embedding models."""

    def __init__(self, document_prefix: str = "検索文書: ") -> None:
        """Initialize the prefix handler.

        Args:
            document_prefix: Default prefix to add to document chunks for embedding

        """
        self.document_prefix = document_prefix

    def add_document_prefix(self, text: str, model_type: str = "ruri") -> str:
        """Add appropriate prefix to text based on the embedding model type.

        Args:
            text: Original text content
            model_type: Type of embedding model being used

        Returns:
            Text with appropriate prefix applied

        """
        # For Ruri v3 model and similar models, add the document prefix
        if model_type in ["ruri", "ruri-v3"]:
            return f"{self.document_prefix}{text}"
            
        # For other models, return text as-is (can be extended in the future)
        return text

