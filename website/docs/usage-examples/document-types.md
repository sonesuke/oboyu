---
id: document-types
title: Handling Different Document Formats
sidebar_position: 2
---

# Handling Different Document Formats

Oboyu supports a wide range of document formats. This guide shows you how to effectively search and manage different types of files in your collection.

## PDF Documents

PDFs are common for reports, papers, and formal documents. Oboyu extracts and indexes text content from PDFs automatically.

### Basic PDF Search
```bash
# Search only in PDF files
oboyu query --query "annual report"

# Find PDFs with specific content
oboyu query --query "financial statement 2024" --mode hybrid
```

### Handling Scanned PDFs
For scanned documents with OCR text:
```bash
# Index with OCR enhancement (if available)
oboyu index ~/Documents/scanned --ocr-enhance

# Search with fuzzy matching for OCR errors
oboyu query --query "bugdet reprot"
```

### PDF-Specific Examples
```bash
# Find research papers
oboyu query --query "machine learning" --db-path ~/indexes/research.db

# Find forms and applications
oboyu query --query "application form"

# Find presentations converted to PDF
oboyu query --query "slide deck presentation"
```

## Markdown Files

Markdown is perfect for notes, documentation, and technical writing.

### Basic Markdown Search
```bash
# Search only markdown files
oboyu query --query "TODO"

# Find markdown with specific headers
oboyu query --query "## Installation"
```

### Markdown Structure Search
```bash
# Find files with code blocks
oboyu query --query "```python"

# Find files with links
oboyu query --query "[link]("

# Find files with images
oboyu query --query "![image]"
```

### Common Markdown Workflows
```bash
# Find all README files
oboyu query --query "*" --db-path ~/indexes/example.db

# Find documentation files
oboyu query --query "documentation"

# Find blog posts
oboyu query --query "date:"
```

## Office Documents

### Microsoft Word (.docx)
```bash
# Search Word documents
oboyu query --query "contract"

# Find templates
oboyu query --query "template"

# Find track changes comments
oboyu query --query "comment:" --mode vector
```

### Excel Files (.xlsx)
```bash
# Search spreadsheets
oboyu query --query "budget"

# Find files with specific data
oboyu query --query "Q4 revenue"

# Find formulas (if extracted)
oboyu query --query "SUM(A1:"
```

### PowerPoint (.pptx)
```bash
# Search presentations
oboyu query --query "roadmap"

# Find slide titles
oboyu query --query "Agenda"

# Find speaker notes
oboyu query --query "note:" --mode vector
```

## Plain Text Files

Simple but powerful for logs, notes, and data files.

### Basic Text Search
```bash
# Search text files
oboyu query --query "error"

# Search log files
oboyu query --query "ERROR"

# Search configuration files
oboyu query --query "port"
```

### Structured Text Files
```bash
# Search CSV files
oboyu query --query "customer_id"

# Search JSON files
oboyu query --query '"api_key"'

# Search XML files
oboyu query --query "<configuration>"
```

## Code Files

Oboyu can search through source code effectively.

### Language-Specific Search
```bash
# Python files
oboyu query --query "def process_data"

# JavaScript files
oboyu query --query "async function"

# Java files
oboyu query --query "public class"
```

### Code Pattern Search
```bash
# Find imports
oboyu query --query "import pandas"

# Find function definitions
oboyu query --query "function.*export"

# Find TODO comments
oboyu query --query "TODO:|FIXME:"
```

## Email Files

If you export emails to files:

### Email Search Patterns
```bash
# Search email files
oboyu query --query "meeting invitation"

# Find emails from specific sender
oboyu query --query "From: boss@company.com"

# Find emails with attachments
oboyu query --query "attachment" --mode vector
```

## Web Documents

### HTML Files
```bash
# Search HTML content
oboyu query --query "contact form"

# Find specific tags
oboyu query --query "<form"

# Find meta descriptions
oboyu query --query 'meta name="description"'
```

## Mixed Format Workflows

### Project Documentation
When projects have multiple file types:
```bash
# Search across all documentation
oboyu query --query "API endpoint"

# Find all files about a feature
oboyu query --query "user authentication" --mode vector
```

### Research Collection
For mixed academic materials:
```bash
# Search papers and notes
oboyu query --query "hypothesis testing"

# Find citations
oboyu query --query "et al. 2024"
```

## Format-Specific Tips

### Large Files
```bash
# For large PDFs or documents
oboyu index ~/large-docs --chunk-size 1000

# Search with context
oboyu query --query "conclusion"
```

### Compressed Archives
```bash
# Index contents of archives
oboyu index ~/Documents --extract-archives

# Search within extracted content
oboyu query --query "readme"
```

### Binary Files with Metadata
```bash
# Search image metadata
oboyu query --query "Canon EOS"

# Search audio file tags
oboyu query --query "Beatles"
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
- Learn about [Performance Tuning](../configuration-optimization/performance-tuning.md) for large collections
- See [Real-world Scenarios](../real-world-scenarios/technical-docs.md) for practical examples