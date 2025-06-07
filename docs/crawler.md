# Oboyu Crawler Architecture

## Overview

The Crawler component is responsible for discovering, extracting, and normalizing documents from the local file system. It serves as the foundation of the Oboyu pipeline, preparing documents for the indexing process with specialized handling for Japanese content.

## Design Goals

- Efficiently traverse file systems to discover documents
- Properly handle various file formats and encodings
- Detect and process Japanese text with high accuracy
- Extract relevant metadata from documents
- Support incremental processing for changed documents

## Component Structure

### Document Discovery

The document discovery subsystem implements:

- Recursive directory traversal with configurable depth
- File pattern matching using glob patterns
- Exclusion rules for unwanted files and directories
- Support for .gitignore files to respect project-specific exclusions
- Metadata extraction including creation and modification times
- Change detection for incremental updates

```python
def discover_documents(
    root_directory: Path,
    patterns: List[str],
    exclude_patterns: List[str],
    max_depth: int = 10,
    respect_gitignore: bool = True
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Discover documents matching patterns with metadata."""
    # Returns list of (path, metadata) tuples
    # Note: max_file_size is hard-coded to 10MB, follow_symlinks is hard-coded to False
```

### Content Extraction

The content extraction subsystem handles:

- File format detection based on extension and content analysis
- Text file content extraction (currently supports only text-based formats)
- Text normalization for consistent processing
- Language detection with emphasis on Japanese identification

**Note**: Oboyu currently supports only text-based file formats. Binary formats like PDF, Word documents, or Excel files are not supported.

```python
def extract_content(
    file_path: Path,
    encoding: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """Extract and normalize content from a file.
    
    Returns:
        Tuple of (content, detected_language, metadata)
    """
```

### Crawler Result Structure

The main `Crawler` class processes documents and returns structured results:

```python
@dataclass
class CrawlerResult:
    """Result of a document crawl operation."""
    
    path: Path                    # Path to the document
    title: str                    # Document title (from filename or content)
    content: str                  # Normalized document content
    language: str                 # Detected language code
    metadata: Dict[str, object]   # Additional metadata
    
    def to_chunk_data(self) -> Dict[str, Any]:
        """Convert to format suitable for indexer."""
```

### Japanese Text Processing

The Japanese text subsystem provides:

- Automatic encoding detection (UTF-8, Shift-JIS, EUC-JP)
- Encoding conversion to UTF-8 for consistent processing
- Initial text normalization for Japanese content
- Character set validation and repair

```python
def process_japanese_text(content, encoding):
    # Implementation details
    # Returns normalized UTF-8 text
```

## Data Flow

1. User specifies directories to index
2. Crawler discovers all matching documents
3. For each document:
   - Extract content based on file type
   - Detect language and encoding
   - Apply appropriate text normalization
   - Extract metadata
4. Pass normalized documents to the Indexer

## Configuration Options

The Crawler is configured through the following settings in `config.yaml`:

```yaml
crawler:
  depth: 10                      # Maximum directory traversal depth
  include_patterns:              # File patterns to include
    - "*.txt"
    - "*.md"
    - "*.html"
    - "*.py"
    - "*.java"
  exclude_patterns:              # Patterns to exclude
    - "*/node_modules/*"
    - "*/venv/*"
  respect_gitignore: true        # Whether to respect .gitignore files
  # Japanese encoding detection is automatic
  max_workers: 4                 # Number of parallel workers for processing
  
  # Hard-coded values (no longer configurable):
  # max_file_size: 10MB          # Maximum file size
  # follow_symlinks: false       # Never follow symbolic links
```

## Performance Considerations

- Uses parallel processing for I/O-bound operations
- Implements streaming for large file handling
- Only re-processes documents that have changed
- Efficiently manages memory usage for large directory structures

## Integration with Other Components

The Crawler provides documents to the Indexer in a standardized format:

```python
document = {
    "path": "/path/to/file.txt",
    "title": "Document Title",
    "content": "Normalized document content...",
    "language": "ja",
    "metadata": {
        "created_at": "2025-05-01T12:34:56",
        "modified_at": "2025-05-15T09:12:34",
        "file_size": 1234,
        # Other metadata
    }
}
```

This standardized format ensures clean separation between the Crawler and Indexer components.