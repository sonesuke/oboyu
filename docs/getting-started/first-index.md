---
id: first-index
title: Creating Your First Index
sidebar_position: 2
---

# Creating Your First Index

This guide will walk you through creating your first document index with Oboyu. In just a few minutes, you'll have a searchable collection of your documents.

## What is an Index?

An index is Oboyu's searchable database of your documents. When you create an index, Oboyu:
- Discovers documents in specified directories
- Extracts and processes text content
- Creates semantic embeddings for intelligent search
- Builds a fast search structure

## Quick Start: Index a Directory

The simplest way to create an index is to point Oboyu at a directory:

```bash
oboyu index ~/Documents/my-notes
```

This command will:
1. Scan the `my-notes` directory for supported files
2. Process all documents found
3. Create a searchable index

## Supported File Types

Oboyu currently supports plain text files and automatically recognizes these file types:

- **Text Documents**: `.txt`, `.md`, `.markdown`
- **Code Files**: `.py`, `.js`, `.java`, `.cpp`, etc.
- **Web Documents**: `.html`, `.htm`
- **Configuration**: `.json`, `.yaml`, `.xml`

**Note**: Binary files like `.pdf` and `.docx` are not currently supported.

## Monitoring Progress

During indexing, you'll see a progress display showing:
- Directory scanning progress
- Documents being processed
- Final summary with total files and chunks indexed

To reduce screen output for large collections, use:
```bash
oboyu index ~/Documents --quiet-progress
```

## Basic Indexing Examples

### Index Multiple Directories

```bash
oboyu index ~/Documents/projects ~/Documents/notes ~/Documents/research
```

### Index with a Custom Database Path

Specify a custom database location:

```bash
oboyu index ~/Documents/work-docs --db-path ~/my-indexes/work.db
```

Later, search using this specific database:
```bash
oboyu query --query "meeting notes" --db-path ~/my-indexes/work.db
```

### Index Specific File Types

Focus on particular file types using include patterns:

```bash
oboyu index ~/Documents --include-patterns "*.md" --include-patterns "*.txt"
```

### Exclude Directories

Skip certain folders using exclude patterns:

```bash
oboyu index ~/Documents --exclude-patterns "*/archive/*" --exclude-patterns "*/temp/*"
```

## Understanding Index Output

After indexing completes, you'll see a summary like:

```
Indexed 156 files (234 chunks) in 45.2s
```

This tells you:
- **Files**: Number of documents processed
- **Chunks**: Number of text segments created for search
- **Time**: Total processing time

You can now search your documents:
```bash
oboyu query --query "your search terms"
```

## Best Practices for Indexing

### 1. Start Small
Begin with a focused directory to understand how Oboyu works:
```bash
oboyu index ~/Documents/current-project
```

### 2. Organize Your Documents
Structure your files logically before indexing:
```
~/Documents/
├── projects/
│   ├── project-a/
│   └── project-b/
├── meeting-notes/
└── research/
```

### 3. Use Separate Database Files
Create separate indices for different purposes using custom database paths:
```bash
oboyu index ~/work-docs --db-path ~/indexes/work.db
oboyu index ~/personal-notes --db-path ~/indexes/personal.db
oboyu index ~/research-papers --db-path ~/indexes/research.db
```

### 4. Regular Updates
Keep your index current by re-indexing periodically (Oboyu performs incremental updates by default):
```bash
oboyu index ~/Documents
```

## Checking Index Status

Check the status of what would be indexed:

```bash
# Check what files would be processed
oboyu manage status

# Check differences (what would be updated)
oboyu manage diff
```

The index database is stored at `~/.oboyu/oboyu.db` by default, or at the path specified with `--db-path`.

## Incremental Indexing

Oboyu supports incremental updates to save time and performs them by default:

```bash
# Incremental indexing (default behavior)
oboyu index ~/Documents

# Force full reindex
oboyu index ~/Documents --force
```

## Handling Large Document Collections

For large collections (10,000+ files):

### 1. Index in Batches
```bash
oboyu index ~/Documents/2023 --db-path ~/indexes/docs-2023.db
oboyu index ~/Documents/2024 --db-path ~/indexes/docs-2024.db
```

### 2. Adjust Chunk Settings for Performance
```bash
# Adjust chunk size for better performance
oboyu index ~/Documents --chunk-size 1024

# Set chunk overlap for better search results
oboyu index ~/Documents --chunk-overlap 100
```

### 3. Use Minimal Progress Output
```bash
# Reduce screen output for faster processing
oboyu index ~/large-collection --quiet-progress
```

## Common Indexing Scenarios

### Text Documents and Notes
```bash
oboyu index ~/Papers --include-patterns "*.txt" --include-patterns "*.md" --db-path ~/indexes/research.db
```

### Software Documentation
```bash
oboyu index ~/dev/docs --include-patterns "*.md" --include-patterns "*.rst" --db-path ~/indexes/dev-docs.db
```

### Meeting Notes
```bash
oboyu index ~/OneDrive/MeetingNotes --db-path ~/indexes/meetings.db
```

### Mixed Language Documents
```bash
# Oboyu automatically detects Japanese content
oboyu index ~/Documents/日本語資料 --db-path ~/indexes/japanese-docs.db
```

## Troubleshooting

### Index Creation Fails

If indexing fails, check:
1. **Permissions**: Ensure you have read access to all files
2. **Disk Space**: Indices typically need 10-20% of source document size
3. **File Corruption**: Corrupted files are skipped automatically

### Slow Indexing

To speed up indexing:
- Close other applications to free up resources
- Use SSD storage for better performance
- Index smaller directories separately

### Missing Documents

If some documents aren't indexed:
```bash
# Check which files were skipped
oboyu index ~/Documents --verbose
```

## Next Steps

Now that you've created your first index, you're ready to:
- [Execute Your First Search](first-search.md) - Learn how to search your indexed documents
- [Basic Workflows](../usage-examples/basic-workflow.md) - Discover daily usage patterns