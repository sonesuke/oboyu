# Quick Start Guide

Welcome to Oboyu! This guide will help you get up and running with semantic document search in under 5 minutes.

## What is Oboyu?

Oboyu is a powerful local semantic search engine that lets you search through your documents using natural language queries. It excels at:
- Finding relevant content even when you don't know the exact keywords
- Supporting Japanese text with automatic encoding detection
- Providing fast, accurate search results through hybrid search (combining keyword and semantic search)
- Working entirely offline with your private documents

## Prerequisites

- Python 3.10 or higher
- 2GB of free disk space (for models)
- macOS or Linux (Windows support via WSL)

## Installation

### 1. Install Oboyu

Using UV (recommended):
```bash
uv tool install oboyu
```

Using pip:
```bash
pip install oboyu
```

### 2. Download Models (One-time Setup)

Oboyu will automatically download required models on first use. To pre-download them:

```bash
oboyu --download-models
```

This downloads:
- Embedding model for semantic search (~90MB)
- Tokenizer for text processing (~5MB)

## Your First Search - 3 Simple Steps

### Step 1: Index Your Documents

Index a directory of documents (supports text-based formats: .txt, .md, .html, .py, .java, and other plain text files):

```bash
oboyu index /path/to/your/documents
```

Example - index your notes directory:
```bash
oboyu index ~/Documents/notes
```

You'll see progress like:
```
Indexing documents...
Processed 150 files in 45 seconds
Index created successfully!
```

### Step 2: Search Your Documents

Search using natural language:

```bash
oboyu query "how to configure Python logging"
```

### Step 3: Interactive Search (Recommended)

For the best experience, use interactive mode:

```bash
oboyu query --interactive
```

This opens a search interface where you can:
- Type queries and see instant results
- Navigate results with arrow keys
- Press Enter to view full content
- Use `/mode` to switch search modes
- Type `/help` for all commands

## Common Use Cases

### 1. Search Technical Documentation
```bash
# Index your project documentation
oboyu index ~/projects/myapp/docs

# Find specific implementation details
oboyu query "authentication flow"
```

### 2. Research Notes and Documentation
```bash
# Index research notes
oboyu index ~/research/notes --include "*.md"

# Find related concepts
oboyu query "machine learning optimization techniques"
```

### 3. Code Documentation Search
```bash
# Index only markdown files
oboyu index ~/projects --include "*.md"

# Search for API usage
oboyu query "API authentication examples"
```

### 4. Japanese Document Search
```bash
# Index Japanese documents (encoding auto-detected)
oboyu index ~/documents/japanese

# Search in Japanese
oboyu query "機械学習の最適化"
```

## Search Modes

Oboyu offers three search modes for different needs:

1. **Hybrid** (default): Combines keyword and semantic search
   ```bash
   oboyu query "Python logging" --mode hybrid
   ```

2. **Semantic**: Finds conceptually related content
   ```bash
   oboyu query "how to debug applications" --mode semantic
   ```

3. **Keyword**: Traditional keyword matching
   ```bash
   oboyu query "ERROR FileNotFoundError" --mode keyword
   ```

## Pro Tips

### 1. Use Filters for Targeted Search
```bash
# Search only in specific files
oboyu query "configuration" --filter "*.yaml"

# Search in a specific directory
oboyu query "testing" --filter "tests/"
```

### 2. Rerank for Better Accuracy
Enable reranking for more accurate results (slower but better):
```bash
oboyu query "complex technical concept" --rerank
```

### 3. Index Incrementally
Update your index with only new or modified files:
```bash
oboyu index /path/to/documents --incremental
```

### 4. Clear Index When Needed
Start fresh with a new index:
```bash
oboyu index --clear
```

## What's Next?

Now that you're up and running:

1. **Explore Interactive Mode**: Try `oboyu query -i` for the full experience
2. **Read Configuration Guide**: Learn about [customization options](configuration.md)
3. **Check CLI Reference**: See all available [commands and options](cli.md)
4. **Learn About Architecture**: Understand [how Oboyu works](architecture.md)

## Need Help?

- Run `oboyu --help` for command help
- Check [Troubleshooting Guide](troubleshooting.md) for common issues
- Visit our [GitHub repository](https://github.com/sonesuke/oboyu) for support

## Quick Command Reference

| Task | Command |
|------|---------|
| Install Oboyu | `uv tool install oboyu` |
| Index documents | `oboyu index /path/to/docs` |
| Search | `oboyu query "your search"` |
| Interactive mode | `oboyu query --interactive` |
| Update index | `oboyu index /path --incremental` |
| Clear index | `oboyu index --clear` |
| Show version | `oboyu --version` |

Happy searching! 🔍