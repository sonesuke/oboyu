"""Document processor domain service - pure business logic for document processing."""

import logging
from typing import List

from ..entities.chunk import Chunk
from ..entities.document import Document
from ..value_objects.chunk_id import ChunkId

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Pure domain service for document processing logic."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Initialize with chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_chunks(self, document: Document) -> List[Chunk]:
        """Create chunks from a document using pure business logic."""
        if not document.should_be_processed():
            return []
        
        content = document.content
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(content):
            end_pos = min(start_pos + self.chunk_size, len(content))
            
            chunk_content = content[start_pos:end_pos]
            
            if chunk_content.strip():
                chunk_id = ChunkId.create(document.path, chunk_index)
                
                chunk = Chunk(
                    id=chunk_id,
                    document_path=document.path,
                    title=document.extract_title_from_content(),
                    content=chunk_content,
                    chunk_index=chunk_index,
                    language=document.language,
                    created_at=document.created_at,
                    modified_at=document.modified_at,
                    metadata=document.metadata.copy(),
                    start_char=start_pos,
                    end_char=end_pos
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            if end_pos >= len(content):
                break
            
            start_pos = end_pos - self.chunk_overlap
        
        return chunks
    
    def should_merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> bool:
        """Determine if two chunks should be merged."""
        if chunk1.document_path != chunk2.document_path:
            return False
        
        if chunk1.language != chunk2.language:
            return False
        
        combined_size = len(chunk1.content) + len(chunk2.content)
        if combined_size > self.chunk_size * 1.5:
            return False
        
        return abs(chunk1.chunk_index - chunk2.chunk_index) == 1
    
    def merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks into one."""
        if not self.should_merge_chunks(chunk1, chunk2):
            raise ValueError("Chunks cannot be merged")
        
        earlier_chunk = chunk1 if chunk1.chunk_index < chunk2.chunk_index else chunk2
        later_chunk = chunk2 if chunk1.chunk_index < chunk2.chunk_index else chunk1
        
        merged_content = f"{earlier_chunk.content} {later_chunk.content}"
        merged_id = ChunkId.create(earlier_chunk.document_path, earlier_chunk.chunk_index)
        
        return Chunk(
            id=merged_id,
            document_path=earlier_chunk.document_path,
            title=earlier_chunk.title,
            content=merged_content,
            chunk_index=earlier_chunk.chunk_index,
            language=earlier_chunk.language,
            created_at=earlier_chunk.created_at,
            modified_at=later_chunk.modified_at,
            metadata=earlier_chunk.metadata,
            start_char=earlier_chunk.start_char,
            end_char=later_chunk.end_char
        )
