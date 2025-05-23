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
def discover_documents(directory, patterns, exclude_patterns, max_depth, respect_gitignore=True):
    # Implementation details
    # Returns list of document paths with metadata
```

### Content Extraction

The content extraction subsystem handles:

- File format detection based on extension and content analysis
- Format-specific content extraction for various file types
- Text normalization for consistent processing
- Language detection with emphasis on Japanese identification

```python
def extract_content(file_path):
    # Implementation details
    # Returns normalized document content and detected language
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
  max_file_size: 10485760        # 10MB maximum file size
  follow_symlinks: false         # Whether to follow symbolic links
  respect_gitignore: true        # Whether to respect .gitignore files
  japanese_encodings:            # Japanese encodings to detect
    - "utf-8"
    - "shift-jis"
    - "euc-jp"
  max_workers: 4                 # Number of parallel workers for processing
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