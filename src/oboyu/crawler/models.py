"""Document processing models with comprehensive validation."""

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class LanguageCode(str, Enum):
    """Supported language codes."""
    
    JAPANESE = "ja"
    ENGLISH = "en"
    UNKNOWN = "unknown"


class EncodingType(str, Enum):
    """Supported text encodings."""
    
    UTF8 = "utf-8"
    SHIFT_JIS = "shift-jis"
    EUC_JP = "euc-jp"
    ISO_2022_JP = "iso-2022-jp"


class DocumentMetadata(BaseModel):
    """Model for document metadata with validation."""
    
    file_size: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(..., description="File creation timestamp")
    modified_at: datetime = Field(..., description="File modification timestamp")
    encoding: EncodingType = Field(..., description="Text encoding detected")
    language: LanguageCode = Field(..., description="Detected language")
    content_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash of content")
    mime_type: Optional[str] = Field(default=None, description="MIME type of file")
    title_extracted: bool = Field(default=False, description="Whether title was extracted from content")
    
    @field_validator('content_hash')
    @classmethod
    def validate_content_hash(cls, v: str) -> str:
        """Validate SHA-256 hash format."""
        if not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError('Content hash must be a valid hexadecimal SHA-256 hash')
        return v.lower()
    
    @field_validator('mime_type')
    @classmethod
    def validate_mime_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate MIME type format."""
        if v is not None:
            if '/' not in v:
                raise ValueError('MIME type must contain forward slash')
            parts = v.split('/')
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise ValueError('Invalid MIME type format')
        return v
    
    @model_validator(mode='after')
    def validate_timestamps(self) -> 'DocumentMetadata':
        """Validate timestamp relationships."""
        if self.created_at > self.modified_at:
            raise ValueError('created_at cannot be later than modified_at')
        return self


class CrawlerResult(BaseModel):
    """Enhanced model for crawler results with validation."""
    
    path: Path = Field(..., description="Path to the source document")
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=1, description="Document content")
    language: LanguageCode = Field(..., description="Detected language")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    processing_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Processing time")
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that path exists and is a file."""
        if not v.exists():
            raise ValueError(f'File does not exist: {v}')
        if not v.is_file():
            raise ValueError(f'Path is not a file: {v}')
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not just whitespace."""
        if len(v.strip()) == 0:
            raise ValueError('Content cannot be empty or whitespace only')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and clean title."""
        # Remove excessive whitespace
        cleaned = ' '.join(v.split())
        if not cleaned:
            raise ValueError('Title cannot be empty')
        return cleaned
    
    def to_chunk_data(self) -> Dict[str, Any]:
        """Convert to format suitable for indexer with validation."""
        return {
            'path': str(self.path),
            'title': self.title,
            'content': self.content,
            'language': self.language.value,
            'metadata': self.metadata.model_dump(),
            'file_size': self.metadata.file_size,
            'modified_at': self.metadata.modified_at,
            'content_hash': self.metadata.content_hash
        }
    
    def generate_content_hash(self) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()
    
    @model_validator(mode='after')
    def validate_content_hash_matches(self) -> 'CrawlerResult':
        """Validate that content hash matches actual content."""
        expected_hash = self.generate_content_hash()
        if self.metadata.content_hash != expected_hash:
            raise ValueError('Content hash does not match actual content')
        return self


class ChunkData(BaseModel):
    """Model for document chunk data with validation."""
    
    id: str = Field(..., pattern=r'^[a-zA-Z0-9\-_/:.]+$', description="Unique chunk identifier")
    path: Path = Field(..., description="Source document path")
    title: str = Field(..., min_length=1, max_length=500, description="Chunk or document title")
    content: str = Field(..., min_length=1, description="Chunk content")
    chunk_index: int = Field(..., ge=0, description="Position in original document")
    language: LanguageCode = Field(..., description="Content language")
    created_at: datetime = Field(default_factory=datetime.now, description="Chunk creation timestamp")
    modified_at: datetime = Field(default_factory=datetime.now, description="Last modification timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    start_char: int = Field(..., ge=0, description="Start character position in original document")
    end_char: int = Field(..., ge=0, description="End character position in original document")
    
    @field_validator('id')
    @classmethod
    def validate_chunk_id(cls, v: str) -> str:
        """Validate chunk ID format."""
        if ':' not in v:
            raise ValueError('Chunk ID must contain path and index separated by ":"')
        return v
    
    @field_validator('content')
    @classmethod
    def validate_chunk_content(cls, v: str) -> str:
        """Validate chunk content."""
        if len(v.strip()) == 0:
            raise ValueError('Chunk content cannot be empty')
        return v
    
    @model_validator(mode='after')
    def validate_character_positions(self) -> 'ChunkData':
        """Validate character position relationships."""
        if self.start_char >= self.end_char:
            raise ValueError('start_char must be less than end_char')
        return self


class FileProcessingResult(BaseModel):
    """Model for file processing results."""
    
    file_path: Path = Field(..., description="Processed file path")
    status: ProcessingStatus = Field(..., description="Processing status")
    chunks_created: int = Field(ge=0, description="Number of chunks created")
    processing_time_ms: float = Field(ge=0.0, description="Processing time in milliseconds")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    language_detected: Optional[LanguageCode] = Field(default=None, description="Detected language")
    encoding_detected: Optional[EncodingType] = Field(default=None, description="Detected encoding")
    content_size: int = Field(ge=0, description="Content size in characters")
    
    @model_validator(mode='after')
    def validate_error_consistency(self) -> 'FileProcessingResult':
        """Validate error message consistency with status."""
        if self.status == ProcessingStatus.ERROR and self.error_message is None:
            raise ValueError('error_message is required when status is ERROR')
        if self.status != ProcessingStatus.ERROR and self.error_message is not None:
            raise ValueError('error_message should only be set when status is ERROR')
        return self


class BatchProcessingResult(BaseModel):
    """Model for batch processing results."""
    
    total_files: int = Field(ge=0, description="Total number of files to process")
    processed_files: int = Field(ge=0, description="Number of successfully processed files")
    failed_files: int = Field(ge=0, description="Number of files that failed processing")
    skipped_files: int = Field(ge=0, description="Number of files that were skipped")
    total_chunks: int = Field(ge=0, description="Total number of chunks created")
    processing_time_ms: float = Field(ge=0.0, description="Total processing time")
    file_results: List[FileProcessingResult] = Field(description="Individual file processing results")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    @model_validator(mode='after')
    def validate_file_counts(self) -> 'BatchProcessingResult':
        """Validate file count consistency."""
        calculated_total = self.processed_files + self.failed_files + self.skipped_files
        if calculated_total != self.total_files:
            raise ValueError('Sum of processed, failed, and skipped files must equal total_files')
        
        if len(self.file_results) != self.total_files:
            raise ValueError('Number of file_results must equal total_files')
        
        return self


class DiscoveryResult(BaseModel):
    """Model for file discovery results."""
    
    discovered_files: List[Path] = Field(description="List of discovered file paths")
    total_size_bytes: int = Field(ge=0, description="Total size of discovered files")
    file_type_counts: Dict[str, int] = Field(description="Count of files by extension")
    discovery_time_ms: float = Field(ge=0.0, description="Discovery time in milliseconds")
    filters_applied: Dict[str, List[str]] = Field(description="Filters applied during discovery")
    
    @field_validator('discovered_files')
    @classmethod
    def validate_discovered_files(cls, v: List[Path]) -> List[Path]:
        """Validate that all discovered files exist."""
        for file_path in v:
            if not file_path.exists():
                raise ValueError(f'Discovered file does not exist: {file_path}')
            if not file_path.is_file():
                raise ValueError(f'Discovered path is not a file: {file_path}')
        return v
    
    @field_validator('file_type_counts')
    @classmethod
    def validate_file_type_counts(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate file type counts."""
        for ext, count in v.items():
            if count < 0:
                raise ValueError(f'File count cannot be negative for extension {ext}')
        return v
    
    @model_validator(mode='after')
    def validate_file_count_consistency(self) -> 'DiscoveryResult':
        """Validate file count consistency."""
        total_counted = sum(self.file_type_counts.values())
        if total_counted != len(self.discovered_files):
            raise ValueError('Sum of file type counts must equal number of discovered files')
        return self


class ExtractionConfig(BaseModel):
    """Model for content extraction configuration."""
    
    max_file_size_mb: float = Field(default=10.0, gt=0.0, le=100.0, description="Maximum file size in MB")
    supported_extensions: List[str] = Field(
        default_factory=lambda: ['.txt', '.md', '.py', '.rst', '.html'],
        description="Supported file extensions"
    )
    encoding_detection: bool = Field(default=True, description="Enable automatic encoding detection")
    language_detection: bool = Field(default=True, description="Enable automatic language detection")
    japanese_processing: bool = Field(default=True, description="Enable Japanese-specific processing")
    extract_metadata: bool = Field(default=True, description="Extract file metadata")
    
    @field_validator('supported_extensions')
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Validate and normalize file extensions."""
        if not v:
            raise ValueError('At least one supported extension must be specified')
        
        normalized = []
        for ext in v:
            if not ext.startswith('.'):
                ext = '.' + ext
            normalized.append(ext.lower())
        return normalized
    
    @field_validator('max_file_size_mb')
    @classmethod
    def validate_max_file_size(cls, v: float) -> float:
        """Validate maximum file size."""
        if v <= 0:
            raise ValueError('Maximum file size must be positive')
        return v
