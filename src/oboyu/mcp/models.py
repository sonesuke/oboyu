"""MCP protocol models for better type safety and validation."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchMode(str, Enum):
    """Search mode enumeration for MCP requests."""
    
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class LanguageCode(str, Enum):
    """Supported language codes."""
    
    JAPANESE = "ja"
    ENGLISH = "en"


class MCPSearchRequest(BaseModel):
    """Model for MCP search requests."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Search algorithm mode")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    language: Optional[LanguageCode] = Field(default=None, description="Language filter")
    db_path: Optional[str] = Field(default=None, description="Database file path")
    snippet_config: Optional[Dict[str, Any]] = Field(default=None, description="Snippet generation config")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    
    @field_validator('query')
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Validate and normalize query text."""
        normalized = ' '.join(v.split())
        if len(normalized) == 0:
            raise ValueError('Query text cannot be empty')
        return normalized
    
    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate database path format."""
        if v is not None:
            path = Path(v)
            if not path.suffix == '.db':
                raise ValueError('Database path must have .db extension')
        return v
    
    class Config:
        """Pydantic configuration with example."""

        schema_extra = {
            "examples": [
                {
                    "query": "機械学習アルゴリズム",
                    "mode": "hybrid",
                    "top_k": 10,
                    "language": "ja",
                    "snippet_config": {
                        "length": 200,
                        "highlight_matches": True
                    }
                },
                {
                    "query": "Python async programming",
                    "mode": "vector",
                    "top_k": 5,
                    "language": "en"
                }
            ]
        }


class MCPSearchResult(BaseModel):
    """Model for individual search result."""
    
    title: str = Field(description="Document or chunk title")
    content: str = Field(description="Document content or snippet")
    uri: str = Field(description="File URI (file:// format)")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    language: str = Field(description="Detected language code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('uri')
    @classmethod
    def validate_uri(cls, v: str) -> str:
        """Validate URI format."""
        if not v.startswith('file://'):
            raise ValueError('URI must start with file://')
        return v
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return round(v, 6)  # Round to 6 decimal places


class MCPSearchStats(BaseModel):
    """Model for search statistics."""
    
    count: int = Field(ge=0, description="Number of results returned")
    query: str = Field(description="Original search query")
    language_filter: str = Field(default="none", description="Applied language filter")
    query_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Query execution time")
    mode: SearchMode = Field(description="Search mode used")


class MCPSearchResponse(BaseModel):
    """Model for MCP search response."""
    
    results: List[MCPSearchResult] = Field(description="Search results")
    stats: MCPSearchStats = Field(description="Search statistics")
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    @model_validator(mode='after')
    def validate_response_consistency(self) -> 'MCPSearchResponse':
        """Validate response consistency."""
        if self.error is None and len(self.results) != self.stats.count:
            raise ValueError('Results count must match stats.count when no error')
        return self
    
    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class MCPIndexRequest(BaseModel):
    """Model for MCP indexing requests."""
    
    directory_path: str = Field(..., description="Directory path to index")
    incremental: bool = Field(default=True, description="Use incremental indexing")
    db_path: Optional[str] = Field(default=None, description="Database file path")
    include_patterns: List[str] = Field(
        default_factory=lambda: ["*.txt", "*.md", "*.py", "*.rst"],
        description="File patterns to include"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["*/node_modules/*", "*/venv/*"],
        description="Patterns to exclude"
    )
    
    @field_validator('directory_path')
    @classmethod
    def validate_directory_path(cls, v: str) -> str:
        """Validate directory exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Directory does not exist: {v}')
        if not path.is_dir():
            raise ValueError(f'Path is not a directory: {v}')
        return str(path.absolute())
    
    @field_validator('include_patterns')
    @classmethod
    def validate_include_patterns(cls, v: List[str]) -> List[str]:
        """Validate include patterns format."""
        if not v:
            raise ValueError('At least one include pattern must be specified')
        return v


class MCPIndexResponse(BaseModel):
    """Model for MCP indexing response."""
    
    success: bool = Field(description="Whether indexing was successful")
    directory: str = Field(description="Indexed directory path")
    documents_indexed: int = Field(ge=0, description="Number of documents indexed")
    chunks_indexed: int = Field(ge=0, description="Number of chunks created")
    db_path: str = Field(description="Database path used")
    processing_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Processing time")
    error: Optional[str] = Field(default=None, description="Error message if any")


class MCPIndexInfo(BaseModel):
    """Model for index information."""
    
    document_count: int = Field(ge=0, description="Number of indexed documents")
    chunk_count: int = Field(ge=0, description="Number of chunks in index")
    languages: List[str] = Field(description="Detected languages in index")
    embedding_model: str = Field(description="Embedding model used")
    db_path: str = Field(description="Database file path")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")
    db_size_mb: Optional[float] = Field(default=None, ge=0.0, description="Database size in MB")
    
    @field_validator('languages')
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        """Validate language codes."""
        valid_codes = {'ja', 'en', 'unknown'}
        for lang in v:
            if lang not in valid_codes:
                raise ValueError(f'Invalid language code: {lang}')
        return v


class MCPIndexInfoResponse(BaseModel):
    """Model for index info response."""
    
    info: Optional[MCPIndexInfo] = Field(default=None, description="Index information")
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    @model_validator(mode='after')
    def validate_info_or_error(self) -> 'MCPIndexInfoResponse':
        """Validate that either info or error is provided."""
        if self.info is None and self.error is None:
            raise ValueError('Either info or error must be provided')
        return self


class MCPErrorResponse(BaseModel):
    """Model for MCP error responses."""
    
    error: str = Field(..., min_length=1, description="Error message")
    error_code: str = Field(default="UNKNOWN_ERROR", description="Error code")
    details: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    @field_validator('error_code')
    @classmethod
    def validate_error_code(cls, v: str) -> str:
        """Validate error code format."""
        if not v.isupper() or not v.replace('_', '').isalnum():
            raise ValueError('Error code must be uppercase alphanumeric with underscores')
        return v


class SnippetConfig(BaseModel):
    """Model for snippet generation configuration."""
    
    max_length: int = Field(default=200, ge=50, le=1000, description="Maximum snippet length")
    context_window: int = Field(default=50, ge=0, le=200, description="Context window size")
    highlight_terms: bool = Field(default=True, description="Highlight matching terms")
    prefer_sentences: bool = Field(default=True, description="Prefer complete sentences")
    japanese_aware: bool = Field(default=True, description="Japanese language awareness")
    
    @model_validator(mode='after')
    def validate_snippet_config(self) -> 'SnippetConfig':
        """Validate snippet configuration."""
        if self.context_window >= self.max_length:
            raise ValueError('context_window must be less than max_length')
        return self


class SearchFilters(BaseModel):
    """Model for search filters."""
    
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Date range filter")
    path_patterns: Optional[List[str]] = Field(default=None, description="Path pattern filters")
    file_types: Optional[List[str]] = Field(default=None, description="File type filters")
    language_filter: Optional[LanguageCode] = Field(default=None, description="Language filter")
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum score threshold")
    
    @field_validator('file_types')
    @classmethod
    def validate_file_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate file type extensions."""
        if v is not None:
            normalized = []
            for ext in v:
                if not ext.startswith('.'):
                    ext = '.' + ext
                normalized.append(ext.lower())
            return normalized
        return v
    
    @field_validator('path_patterns')
    @classmethod
    def validate_path_patterns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate path patterns."""
        if v is not None and not v:
            raise ValueError('path_patterns cannot be empty if specified')
        return v
