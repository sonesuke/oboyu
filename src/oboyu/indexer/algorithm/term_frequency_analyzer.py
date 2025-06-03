"""Term frequency analyzer for document processing.

This module provides functionality for analyzing and computing term frequencies
for documents using tokenization services.
"""

import logging
from typing import Dict, Set

from oboyu.common.services import TokenizerService

logger = logging.getLogger(__name__)


class TermFrequencyAnalyzer:
    """Analyzes and computes term frequencies for documents."""

    def __init__(self, tokenizer: TokenizerService) -> None:
        """Initialize the term frequency analyzer.

        Args:
            tokenizer: Tokenizer service for text processing

        """
        self.tokenizer = tokenizer

    def analyze_document(self, content: str) -> Dict[str, int]:
        """Analyze document content and return term frequencies.

        Args:
            content: Document content to analyze

        Returns:
            Dictionary mapping terms to their frequencies

        """
        return self.tokenizer.get_term_frequencies(content)

    def get_document_length(self, content: str) -> int:
        """Get document length in terms.

        Args:
            content: Document content

        Returns:
            Number of terms in the document

        """
        term_frequencies = self.analyze_document(content)
        return sum(term_frequencies.values())

    def extract_unique_terms(self, content: str) -> Set[str]:
        """Extract unique terms from document content.

        Args:
            content: Document content

        Returns:
            Set of unique terms in the document

        """
        term_frequencies = self.analyze_document(content)
        return set(term_frequencies.keys())

    def get_term_positions(self, content: str, term: str) -> list[int]:
        """Get positions of a term in the document.

        Args:
            content: Document content
            term: Term to find positions for

        Returns:
            List of positions where the term appears

        """
        positions = []
        tokens = self.tokenizer.tokenize(content)
        
        for i, token in enumerate(tokens):
            if token == term:
                positions.append(i)
        
        return positions

    def analyze_batch(self, contents: list[str]) -> Dict[int, Dict[str, int]]:
        """Analyze multiple documents and return term frequencies.

        Args:
            contents: List of document contents to analyze

        Returns:
            Dictionary mapping content indices to term frequencies

        """
        results = {}
        
        for i, content in enumerate(contents):
            results[i] = self.analyze_document(content)
        
        return results

    def get_vocabulary_from_documents(self, contents: list[str]) -> Set[str]:
        """Extract vocabulary (unique terms) from multiple documents.

        Args:
            contents: List of document contents

        Returns:
            Set of all unique terms across all documents

        """
        vocabulary = set()
        
        for content in contents:
            unique_terms = self.extract_unique_terms(content)
            vocabulary.update(unique_terms)
        
        return vocabulary

