"""CLI request and response models with comprehensive validation."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchMode(str, Enum):
    """Search mode enumeration."""
    
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class OutputFormat(str, Enum):
    """Output format enumeration."""
    
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class SearchRequest(BaseModel):
    """Model for search request parameters."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Search mode")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    rrf_k: int = Field(default=60, ge=1, le=1000, description="RRF parameter for hybrid search")
    language: Optional[str] = Field(default=None, pattern=r'^[a-z]{2}$', description="Language filter")
    db_path: Optional[Path] = Field(default=None, description="Database path")
    rerank: bool = Field(default=True, description="Enable reranking")
    explain: bool = Field(default=False, description="Show explanation")
    format: OutputFormat = Field(default=OutputFormat.TEXT, description="Output format")
    interactive: bool = Field(default=False, description="Interactive mode")
    
    @field_validator('query')
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate and normalize query text."""
        # Remove excessive whitespace
        normalized = ' '.join(v.split())
        if len(normalized) == 0:
            raise ValueError('Query text cannot be empty')
        return normalized
    
    @model_validator(mode='after')
    def validate_rrf_k(self) -> 'SearchRequest':
        """Validate RRF parameter for hybrid mode."""
        if self.mode == SearchMode.HYBRID:
            if self.rrf_k <= 0:
                raise ValueError('rrf_k must be positive for hybrid mode')
        return self
    
    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate database path exists."""
        if v is not None and not v.exists():
            raise ValueError(f'Database path does not exist: {v}')
        return v


class IndexRequest(BaseModel):
    """Model for indexing request parameters."""
    
    directories: List[Path] = Field(..., min_length=1, description="Directories to index")
    include_patterns: List[str] = Field(
        default_factory=lambda: ["*.txt", "*.md", "*.py", "*.rst"],
        description="File patterns to include"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["*/node_modules/*", "*/venv/*", "*/__pycache__/*"],
        description="Patterns to exclude"
    )
    force: bool = Field(default=False, description="Force reindexing")
    incremental: bool = Field(default=True, description="Incremental indexing")
    chunk_size: int = Field(default=1024, ge=100, le=8192, description="Chunk size")
    chunk_overlap: int = Field(default=256, ge=0, description="Chunk overlap")
    db_path: Optional[Path] = Field(default=None, description="Database path")
    
    @field_validator('directories')
    @classmethod
    def validate_directories(cls, v: List[Path]) -> List[Path]:
        """Validate that all directories exist."""
        for directory in v:
            if not directory.exists():
                raise ValueError(f'Directory does not exist: {directory}')
            if not directory.is_dir():
                raise ValueError(f'Path is not a directory: {directory}')
        return v
    
    @field_validator('include_patterns')
    @classmethod
    def validate_include_patterns(cls, v: List[str]) -> List[str]:
        """Validate and normalize include patterns."""
        normalized = []
        for pattern in v:
            if not pattern.startswith('*'):
                pattern = '*' + pattern
            normalized.append(pattern)
        return normalized
    
    @model_validator(mode='after')
    def validate_chunk_config(self) -> 'IndexRequest':
        """Validate chunk configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return self


class SearchResponse(BaseModel):
    """Model for search response."""
    
    results: List[Dict[str, Union[str, int, float, bool, None]]] = Field(description="Search results")
    total_found: int = Field(ge=0, description="Total number of results found")
    query_time_ms: float = Field(ge=0.0, description="Query execution time in milliseconds")
    mode: SearchMode = Field(description="Search mode used")
    query: str = Field(description="Original query")
    
    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class IndexResponse(BaseModel):
    """Model for indexing response."""
    
    success: bool = Field(description="Whether indexing was successful")
    files_processed: int = Field(ge=0, description="Number of files processed")
    chunks_created: int = Field(ge=0, description="Number of chunks created")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    processing_time_ms: float = Field(ge=0.0, description="Processing time in milliseconds")
    skipped_files: List[str] = Field(default_factory=list, description="Files that were skipped")
    db_path: str = Field(description="Database path used")


class ConfigRequest(BaseModel):
    """Model for configuration requests."""
    
    section: Optional[str] = Field(default=None, pattern=r'^(crawler|indexer|query)$', description="Config section")
    key: Optional[str] = Field(default=None, description="Configuration key")
    value: Optional[Union[str, int, float, bool]] = Field(default=None, description="Configuration value")
    show_all: bool = Field(default=False, description="Show all configuration")
    
    @model_validator(mode='after')
    def validate_config_operation(self) -> 'ConfigRequest':
        """Validate configuration operation parameters."""
        if not self.show_all and self.section is None:
            raise ValueError('Either show_all must be True or section must be specified')
        if self.value is not None and self.key is None:
            raise ValueError('key must be specified when setting a value')
        return self


class ConfigResponse(BaseModel):
    """Model for configuration response."""
    
    config: Dict[str, Any] = Field(description="Configuration data")
    section: Optional[str] = Field(default=None, description="Configuration section")
    success: bool = Field(description="Whether operation was successful")
    message: Optional[str] = Field(default=None, description="Response message")


class ClearRequest(BaseModel):
    """Model for clear/reset requests."""
    
    db_path: Optional[Path] = Field(default=None, description="Database path to clear")
    confirm: bool = Field(default=False, description="Confirmation flag")
    backup: bool = Field(default=True, description="Create backup before clearing")
    
    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate database path exists."""
        if v is not None and not v.exists():
            raise ValueError(f'Database path does not exist: {v}')
        return v


class ClearResponse(BaseModel):
    """Model for clear/reset response."""
    
    success: bool = Field(description="Whether operation was successful")
    backup_path: Optional[str] = Field(default=None, description="Path to backup file")
    message: str = Field(description="Operation result message")
    db_path: str = Field(description="Database path that was cleared")
