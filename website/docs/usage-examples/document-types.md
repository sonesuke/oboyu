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
oboyu query "annual report" --file-type pdf

# Find PDFs with specific content
oboyu query "financial statement 2024" --file-type pdf --mode hybrid
```

### Handling Scanned PDFs
For scanned documents with OCR text:
```bash
# Index with OCR enhancement (if available)
oboyu index ~/Documents/scanned --ocr-enhance

# Search with fuzzy matching for OCR errors
oboyu query "bugdet reprot" --fuzzy --file-type pdf
```

### PDF-Specific Examples
```bash
# Find research papers
oboyu query "machine learning" --file-type pdf --index research

# Find forms and applications
oboyu query "application form" --file-type pdf

# Find presentations converted to PDF
oboyu query "slide deck presentation" --file-type pdf
```

## Markdown Files

Markdown is perfect for notes, documentation, and technical writing.

### Basic Markdown Search
```bash
# Search only markdown files
oboyu query "TODO" --file-type md

# Find markdown with specific headers
oboyu query "## Installation" --file-type md
```

### Markdown Structure Search
```bash
# Find files with code blocks
oboyu query "```python" --file-type md

# Find files with links
oboyu query "[link](" --file-type md

# Find files with images
oboyu query "![image]" --file-type md
```

### Common Markdown Workflows
```bash
# Find all README files
oboyu query "*" --name "README.md"

# Find documentation files
oboyu query "documentation" --file-type md --path "**/docs/**"

# Find blog posts
oboyu query "date:" --file-type md --path "**/posts/**"
```

## Office Documents

### Microsoft Word (.docx)
```bash
# Search Word documents
oboyu query "contract" --file-type docx

# Find templates
oboyu query "template" --file-type docx,dotx

# Find track changes comments
oboyu query "comment:" --file-type docx --mode semantic
```

### Excel Files (.xlsx)
```bash
# Search spreadsheets
oboyu query "budget" --file-type xlsx

# Find files with specific data
oboyu query "Q4 revenue" --file-type xlsx,xls

# Find formulas (if extracted)
oboyu query "SUM(A1:" --file-type xlsx
```

### PowerPoint (.pptx)
```bash
# Search presentations
oboyu query "roadmap" --file-type pptx

# Find slide titles
oboyu query "Agenda" --file-type pptx

# Find speaker notes
oboyu query "note:" --file-type pptx --mode semantic
```

## Plain Text Files

Simple but powerful for logs, notes, and data files.

### Basic Text Search
```bash
# Search text files
oboyu query "error" --file-type txt

# Search log files
oboyu query "ERROR" --file-type log

# Search configuration files
oboyu query "port" --file-type conf,cfg
```

### Structured Text Files
```bash
# Search CSV files
oboyu query "customer_id" --file-type csv

# Search JSON files
oboyu query '"api_key"' --file-type json

# Search XML files
oboyu query "<configuration>" --file-type xml
```

## Code Files

Oboyu can search through source code effectively.

### Language-Specific Search
```bash
# Python files
oboyu query "def process_data" --file-type py

# JavaScript files
oboyu query "async function" --file-type js,jsx

# Java files
oboyu query "public class" --file-type java
```

### Code Pattern Search
```bash
# Find imports
oboyu query "import pandas" --file-type py

# Find function definitions
oboyu query "function.*export" --file-type js --regex

# Find TODO comments
oboyu query "TODO:|FIXME:" --file-type py,js,java
```

## Email Files

If you export emails to files:

### Email Search Patterns
```bash
# Search email files
oboyu query "meeting invitation" --file-type eml,msg

# Find emails from specific sender
oboyu query "From: boss@company.com" --file-type eml

# Find emails with attachments
oboyu query "attachment" --file-type eml --mode semantic
```

## Web Documents

### HTML Files
```bash
# Search HTML content
oboyu query "contact form" --file-type html

# Find specific tags
oboyu query "<form" --file-type html,htm

# Find meta descriptions
oboyu query 'meta name="description"' --file-type html
```

## Mixed Format Workflows

### Project Documentation
When projects have multiple file types:
```bash
# Search across all documentation
oboyu query "API endpoint" --file-type md,pdf,docx

# Find all files about a feature
oboyu query "user authentication" --mode semantic
```

### Research Collection
For mixed academic materials:
```bash
# Search papers and notes
oboyu query "hypothesis testing" --file-type pdf,md,txt

# Find citations
oboyu query "et al. 2024" --file-type pdf,docx
```

## Format-Specific Tips

### Large Files
```bash
# For large PDFs or documents
oboyu index ~/large-docs --chunk-size 1000

# Search with context
oboyu query "conclusion" --context 500 --file-type pdf
```

### Compressed Archives
```bash
# Index contents of archives
oboyu index ~/Documents --extract-archives

# Search within extracted content
oboyu query "readme" --path "**/*.zip/**"
```

### Binary Files with Metadata
```bash
# Search image metadata
oboyu query "Canon EOS" --file-type jpg,jpeg --metadata

# Search audio file tags
oboyu query "Beatles" --file-type mp3,flac --metadata
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
oboyu formats list

# Use text extraction for unsupported formats
textract unsupported.xyz > supported.txt
```

### Corrupted Files
```bash
# Skip corrupted files
oboyu index ~/Documents --skip-errors

# Find problematic files
oboyu index ~/Documents --verbose | grep "ERROR"
```

### Encoding Issues
```bash
# Handle different encodings
oboyu index ~/Documents --encoding auto

# Force specific encoding
oboyu index ~/japanese-docs --encoding shift-jis
```

## Next Steps

- Explore [Search Patterns](search-patterns.md) for format-specific search techniques
- Learn about [Performance Tuning](../configuration-optimization/performance-tuning.md) for large collections
- See [Real-world Scenarios](../real-world-scenarios/technical-docs.md) for practical examples