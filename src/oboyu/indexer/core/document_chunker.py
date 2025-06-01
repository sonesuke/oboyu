"""Document chunking component for splitting text into manageable chunks.

This module handles the splitting of documents into overlapping chunks
with intelligent boundary detection for sentences and paragraphs.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Responsible for splitting documents into chunks with overlap."""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 256) -> None:
        """Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters

        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks

        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return [""]
            
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        iteration_count = 0
        max_iterations = 10000  # Safety limit to prevent infinite loops

        while start < len(text) and iteration_count < max_iterations:
            iteration_count += 1

            # Get chunk of specified size
            end = start + self.chunk_size

            # Adjust chunk boundary to end at a sentence or paragraph if possible
            if end < len(text):
                end = self._find_best_break_point(text, start, end)

            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position for next chunk, considering overlap
            old_start = start
            start = end - self.chunk_overlap if end < len(text) else len(text)

            # Safety check to prevent infinite loops
            if start <= old_start and iteration_count > 1:
                logger.error(
                    f"chunk_text: Potential infinite loop detected! start={start}, old_start={old_start}"
                )
                start = old_start + max(1, self.chunk_size // 2)  # Force progress

        if iteration_count >= max_iterations:
            logger.error(f"chunk_text: Hit maximum iteration limit {max_iterations}")
            
        return chunks

    def _find_best_break_point(self, text: str, start: int, end: int) -> int:
        """Find the best break point for a chunk boundary.

        Args:
            text: The full text
            start: Start position of the chunk
            end: Proposed end position of the chunk

        Returns:
            Adjusted end position that aligns with sentence or paragraph boundary

        """
        # Try to find paragraph break first
        paragraph_break = text.rfind("\n\n", start, end)
        if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
            return paragraph_break

        # Try to find sentence break
        sentence_breaks = [
            text.rfind(". ", start, end),
            text.rfind("。", start, end),
            text.rfind("! ", start, end),
            text.rfind("？", start, end),
            text.rfind("? ", start, end),
            text.rfind("！", start, end),
            text.rfind("\n", start, end),
        ]

        # Find the latest valid break point
        valid_breaks = [
            b for b in sentence_breaks
            if b != -1 and b > start + self.chunk_size // 2
        ]
        
        if valid_breaks:
            return max(valid_breaks) + 1  # Include the punctuation

        return end

