# Developer Quick Start Guide

Welcome to the Oboyu development quick start! This guide assumes you already have Oboyu installed and covers development-specific workflows.

## Prerequisites

Before following this guide, make sure you have:
- Oboyu installed on your system ([see installation guide](../getting-started/installation))
- Basic familiarity with command-line tools
- Understanding of semantic search concepts

## Development-Specific Setup

For development work with Oboyu, you may want to install the development version:

```bash
# Install development version with all dependencies
pip install oboyu[dev]
```

Or for contributing to Oboyu:

```bash
# Clone and install from source
git clone https://github.com/sonesuke/oboyu.git
cd oboyu
uv sync
uv run oboyu --version
```

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
2. **Read Configuration Guide**: Learn about [customization options](../system-administrators/reference/configuration.md)
3. **Check CLI Reference**: See all available [commands and options](cli.md)
4. **Learn About Architecture**: Understand [how Oboyu works](architecture.md)

## Need Help?

- Run `oboyu --help` for command help
- Check [Troubleshooting Guide](../system-administrators/troubleshooting/troubleshooting.md) for common issues
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