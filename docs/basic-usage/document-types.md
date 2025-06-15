---
id: document-types
title: Handling Different Document Formats
sidebar_position: 2
---

# Handling Different Document Formats

Oboyu supports a wide range of document formats. This guide shows you how to effectively search and manage different types of files in your collection.

## PDF Documents

Oboyu fully supports PDF document indexing and search. Text content is automatically extracted from PDF files, including multi-page documents, and metadata is preserved for enhanced search capabilities.

### Supported PDF Features
- **Text extraction**: Extracts plain text content from all pages
- **Metadata extraction**: Preserves title, author, creation date, and modification date
- **Multi-page support**: Handles documents with multiple pages seamlessly
- **Japanese text support**: Full support for Japanese text in PDFs

### Basic PDF Search
```bash
# Search all indexed PDF files
oboyu search "annual report"

# Find PDFs with specific content
oboyu search "financial statement 2024" --mode hybrid

# Search for content across multiple pages
oboyu search "conclusion recommendations"
```

### PDF Indexing
```bash
# Index a directory containing PDFs
oboyu index ~/Documents/PDFs

# Index with mixed file types (PDFs included by default)
oboyu index ~/Documents --include "*.pdf,*.txt,*.md"
```

### PDF-Specific Examples
```bash
# Find research papers
oboyu search "machine learning neural networks"

# Find forms and applications
oboyu search "application form"

# Find presentations converted to PDF
oboyu search "slide deck presentation"

# Search PDF metadata
oboyu search "author:Smith" --mode vector
```

### Japanese PDF Support
```bash
# Search Japanese PDFs
oboyu search "機械学習"

# Mixed language search
oboyu search "machine learning 機械学習" --mode hybrid
```

### Performance Tips for PDFs
- Large PDFs are processed efficiently with automatic text chunking
- Metadata extraction is fast and preserves document properties
- Text-based PDFs perform better than image-only PDFs
- Consider using semantic search mode for concept-based queries

## Markdown Files

Markdown is perfect for notes, documentation, and technical writing.

### Basic Markdown Search
```bash
# Search only markdown files
oboyu search "TODO"

# Find markdown with specific headers
oboyu search "## Installation"
```

### Markdown Structure Search
```bash
# Find files with code blocks
oboyu search "```python"

# Find files with links
oboyu search "[link]("

# Find files with images
oboyu search "![image]"
```

### Common Markdown Workflows
```bash
# Find all README files
oboyu search "*" --db-path ~/indexes/example.db

# Find documentation files
oboyu search "documentation"

# Find blog posts
oboyu search "date:"
```

## Office Documents

### Microsoft Word (.docx)
```bash
# Search Word documents
oboyu search "contract"

# Find templates
oboyu search "template"

# Find track changes comments
oboyu search "comment:" --mode vector
```

### Excel Files (.xlsx)
```bash
# Search spreadsheets
oboyu search "budget"

# Find files with specific data
oboyu search "Q4 revenue"

# Find formulas (if extracted)
oboyu search "SUM(A1:"
```

### PowerPoint (.pptx)
```bash
# Search presentations
oboyu search "roadmap"

# Find slide titles
oboyu search "Agenda"

# Find speaker notes
oboyu search "note:" --mode vector
```

## Plain Text Files

Simple but powerful for logs, notes, and data files.

### Basic Text Search
```bash
# Search text files
oboyu search "error"

# Search log files
oboyu search "ERROR"

# Search configuration files
oboyu search "port"
```

### Structured Text Files
```bash
# Search CSV files
oboyu search "customer_id"

# Search JSON files
oboyu search '"api_key"'

# Search XML files
oboyu search "<configuration>"
```

## Code Files

Oboyu can search through source code effectively.

### Language-Specific Search
```bash
# Python files
oboyu search "def process_data"

# JavaScript files
oboyu search "async function"

# Java files
oboyu search "public class"
```

### Code Pattern Search
```bash
# Find imports
oboyu search "import pandas"

# Find function definitions
oboyu search "function.*export"

# Find TODO comments
oboyu search "TODO:|FIXME:"
```

## Email Files

If you export emails to files:

### Email Search Patterns
```bash
# Search email files
oboyu search "meeting invitation"

# Find emails from specific sender
oboyu search "From: boss@company.com"

# Find emails with attachments
oboyu search "attachment" --mode vector
```

## Web Documents

### HTML Files
```bash
# Search HTML content
oboyu search "contact form"

# Find specific tags
oboyu search "<form"

# Find meta descriptions
oboyu search 'meta name="description"'
```

## Mixed Format Workflows

### Project Documentation
When projects have multiple file types:
```bash
# Search across all documentation
oboyu search "API endpoint"

# Find all files about a feature
oboyu search "user authentication" --mode vector
```

### Research Collection
For mixed academic materials:
```bash
# Search papers and notes
oboyu search "hypothesis testing"

# Find citations
oboyu search "et al. 2024"
```

## Format-Specific Tips

### Large Files
```bash
# For large PDFs or documents
oboyu index ~/large-docs --chunk-size 1000

# Search with context
oboyu search "conclusion"
```

### Compressed Archives
```bash
# Index contents of archives
oboyu index ~/Documents --extract-archives

# Search within extracted content
oboyu search "readme"
```

### Binary Files with Metadata
```bash
# Search image metadata
oboyu search "Canon EOS"

# Search audio file tags
oboyu search "Beatles"
```

## Best Practices by Format

### For PDFs
1. Keep OCR quality high for scanned documents
2. Use semantic search for concept-based queries
3. Index regularly as PDFs are often updated

### For Markdown
1. Use consistent formatting for better search
2. Include front matter for metadata
3. Use headers for structure-based search

### For Office Documents
1. Use document properties and metadata
2. Keep formatting consistent
3. Extract tables and charts when possible

### For Code Files
1. Include comments for context
2. Use consistent naming conventions
3. Index documentation alongside code

## Format Conversion Tips

### When to Convert
- Convert proprietary formats to open formats
- Convert old formats to supported formats
- Extract text from complex formats

### Conversion Examples
```bash
# Convert before indexing
pandoc input.docx -o output.md
oboyu index ~/converted-docs

# Batch conversion
find . -name "*.doc" -exec pandoc {} -o {}.md \;
```

## Troubleshooting Format Issues

### Unsupported Format
```bash
# Check if format is supported
# (Note: formats list command not available)

# Use text extraction for unsupported formats
textract unsupported.xyz > supported.txt
```

### Corrupted Files
```bash
# Skip corrupted files
oboyu index ~/Documents

# Find problematic files
oboyu index ~/Documents | grep "ERROR"
```

### Encoding Issues
```bash
# Handle different encodings
oboyu index ~/Documents

# Force specific encoding
# (Note: encoding options not available in current implementation)
```

## Next Steps

- Explore [Search Patterns](search-patterns.md) for format-specific search techniques
- Learn about [Performance Tuning](../reference/configuration.md) for large collections
- See [Real-world Scenarios](../use-cases/technical-docs.md) for practical examples