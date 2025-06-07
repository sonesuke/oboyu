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

Oboyu automatically recognizes and indexes these file types:

- **Text Documents**: `.txt`, `.md`, `.markdown`
- **Office Documents**: `.docx`, `.pdf`
- **Code Files**: `.py`, `.js`, `.java`, `.cpp`, etc.
- **Web Documents**: `.html`, `.htm`
- **Configuration**: `.json`, `.yaml`, `.xml`

## Monitoring Progress

During indexing, you'll see a progress display:

```
Indexing documents...
[████████████████████████████████████████] 100% | 156/156 files
Processing: technical-report.pdf
Time elapsed: 00:02:34
Documents indexed: 156
Total size: 45.2 MB
```

## Basic Indexing Examples

### Index Multiple Directories

```bash
oboyu index ~/Documents/projects ~/Documents/notes ~/Documents/research
```

### Index with a Custom Name

Give your index a memorable name:

```bash
oboyu index ~/Documents/work-docs --name "work"
```

Later, search this specific index:
```bash
oboyu query "meeting notes" --index work
```

### Index Specific File Types

Focus on particular file types:

```bash
oboyu index ~/Documents --include "*.md" --include "*.txt"
```

### Exclude Directories

Skip certain folders:

```bash
oboyu index ~/Documents --exclude "archive" --exclude "temp"
```

## Understanding Index Output

After indexing completes, you'll see a summary:

```
Index created successfully!

Summary:
- Total documents: 234
- Successfully indexed: 230
- Skipped: 4 (unsupported format)
- Index size: 128 MB
- Processing time: 3m 45s

Index location: ~/.oboyu/indices/default.db

You can now search your documents:
  oboyu query "your search terms"
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

### 3. Use Meaningful Index Names
Create separate indices for different purposes:
```bash
oboyu index ~/work-docs --name work
oboyu index ~/personal-notes --name personal
oboyu index ~/research-papers --name research
```

### 4. Regular Updates
Keep your index current by re-indexing periodically:
```bash
oboyu index ~/Documents --update
```

## Checking Index Status

View information about your indices:

```bash
# List all indices
oboyu index list

# Show details about a specific index
oboyu index info --name work
```

Example output:
```
Index: work
Created: 2024-01-15 10:30:00
Last updated: 2024-01-20 14:22:00
Documents: 1,234
Total size: 256 MB
Directories:
  - /Users/you/work-docs
  - /Users/you/shared/team-docs
```

## Incremental Indexing

Oboyu supports incremental updates to save time:

```bash
# Only index new or modified files
oboyu index ~/Documents --update

# Force full reindex
oboyu index ~/Documents --force
```

## Handling Large Document Collections

For large collections (10,000+ files):

### 1. Index in Batches
```bash
oboyu index ~/Documents/2023 --name docs-2023
oboyu index ~/Documents/2024 --name docs-2024
```

### 2. Use Background Indexing
```bash
oboyu index ~/large-collection --background
```

### 3. Adjust Performance Settings
```bash
# Use more threads for faster processing
oboyu index ~/Documents --threads 8

# Limit memory usage
oboyu index ~/Documents --memory-limit 2GB
```

## Common Indexing Scenarios

### Academic Papers
```bash
oboyu index ~/Papers --include "*.pdf" --name research
```

### Software Documentation
```bash
oboyu index ~/dev/docs --include "*.md" --include "*.rst" --name dev-docs
```

### Meeting Notes
```bash
oboyu index ~/OneDrive/MeetingNotes --name meetings
```

### Mixed Language Documents
```bash
# Oboyu automatically detects Japanese content
oboyu index ~/Documents/日本語資料 --name japanese-docs
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