---
id: troubleshooting
title: Common Issues and Solutions
sidebar_position: 1
---

# Common Issues and Solutions

Quick solutions to common problems you might encounter while using Oboyu. This guide is organized by the type of issue for easy navigation.

## Installation Issues

### Python Version Compatibility

**Issue**: `oboyu` installation fails with version error
```
ERROR: Package 'oboyu' requires a different Python: 3.8.0 not in '>=3.9'
```

**Solution**: Upgrade to Python 3.9 or higher
```bash
# Check current version
python --version

# Install Python 3.9+ using pyenv
pyenv install 3.11.0
pyenv global 3.11.0

# Or using conda
conda install python=3.11
```

### Build Dependencies Missing

**Issue**: Installation fails with compilation errors

**macOS Solution**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install build dependencies
brew install cmake pkg-config
```

**Linux Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libssl-dev

# Fedora/RHEL
sudo dnf install gcc-c++ cmake openssl-devel
```

**Windows Solution**:
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Permission Errors

**Issue**: Permission denied during installation

**Solution**: Use user installation
```bash
# Install for current user only
pip install --user oboyu

# Or use virtual environment
python -m venv oboyu-env
source oboyu-env/bin/activate  # Linux/macOS
# oboyu-env\Scripts\activate     # Windows
pip install oboyu
```

## Indexing Problems

### Files Not Being Indexed

**Issue**: Expected files don't appear in search results

**Diagnosis**:
```bash
# Check if files were indexed
oboyu index status ~/Documents --verbose

# List indexed files
oboyu index list-files --db-path ~/indexes/personal.db

# Check file patterns
oboyu index info --db-path ~/indexes/personal.db --show-patterns
```

**Common Causes & Solutions**:

1. **File type not included**:
```bash
# Check current patterns
oboyu config show | grep include_patterns

# Add missing file types
oboyu config set crawler.include_patterns "*.md,*.txt,*.pdf"
```

2. **Files excluded by pattern**:
```bash
# Check exclude patterns
oboyu config show | grep exclude_patterns

# Remove problematic exclusions
oboyu config unset crawler.exclude_patterns
```

3. **File size limits**:
```bash
# Check for large files
find ~/Documents -size +10M -type f

# Increase size limit
oboyu config set crawler.max_file_size "50MB"
```

### Indexing Stuck or Very Slow

**Issue**: Indexing process hangs or takes too long

**Quick Fixes**:
```bash
# Kill existing process
pkill -f "oboyu index"

# Start with limited resources
oboyu index ~/Documents --threads 2 --batch-size 8

# Index smaller batches
find ~/Documents -name "*.md" | head -100 | xargs oboyu index
```

**Performance Solutions**:
```bash
# Check available memory
free -h

# Use memory limit
oboyu index ~/Documents --memory-limit 2GB

# Skip problematic files
oboyu index ~/Documents --skip-errors --timeout 60
```

### Corrupted Index

**Issue**: Search returns errors or unexpected results

**Solution**: Rebuild the index
```bash
# Backup current index
cp ~/.oboyu/index.db ~/.oboyu/index.db.backup

# Rebuild from scratch
oboyu index ~/Documents --force --clean

# Verify integrity
oboyu index verify --db-path ~/indexes/personal.db
```

## Search Issues

### No Results Found

**Issue**: Search returns no results for known content

**Troubleshooting Steps**:

1. **Verify content is indexed**:
```bash
# Check if file is in index
oboyu index contains "path/to/file.md"

# List all indexed files
oboyu index list-files | grep "search-term"
```

2. **Try different search modes**:
```bash
# Try BM25 for exact matches
oboyu query "exact phrase" --mode bm25

# Try vector search for concepts
oboyu query "concept description" --mode vector

# Try hybrid for best of both
oboyu query "search terms" --mode hybrid
```

3. **Check query syntax**:
```bash
# Use quotes for exact phrases
oboyu query '"exact phrase here"'

# Use simpler terms
oboyu query "main keyword"

# Check for typos
oboyu query "correct spelling"
```

### Poor Search Quality

**Issue**: Irrelevant results or missing relevant content

**Solutions**:

1. **Enable reranking**:
```bash
oboyu config set indexer.use_reranker true
```

2. **Adjust search mode**:
```bash
# For technical content
oboyu query "technical term" --mode hybrid

# For conceptual search
oboyu query "what is the process for" --mode vector

# For exact matches
oboyu query "ERROR_CODE_123" --mode bm25
```

3. **Optimize chunk size**:
```bash
# For code/technical docs (smaller chunks)
oboyu config set indexer.chunk_size 512

# For long-form content (larger chunks)
oboyu config set indexer.chunk_size 2048
```

### Slow Search Performance

**Issue**: Searches take too long to complete

**Quick Fixes**:
```bash
# Use faster mode
oboyu query "search term" --mode bm25

# Limit results
oboyu query "search term" --limit 5

# Add filters to narrow search
oboyu query "search term" --file-type md --days 30
```

**Optimization**:
```bash
# Optimize index
oboyu index optimize --db-path ~/indexes/personal.db

# Check index size
oboyu index info --db-path ~/indexes/personal.db

# Move to SSD storage
mv ~/.oboyu/index.db /ssd/oboyu/
ln -s /ssd/oboyu/index.db ~/.oboyu/index.db
```

## Memory and Performance Issues

### Out of Memory Errors

**Issue**: Process killed due to memory usage

**Solutions**:
```bash
# Reduce batch size
oboyu config set indexer.batch_size 8

# Limit memory usage
export OBOYU_MEMORY_LIMIT=2GB

# Process files individually
find ~/Documents -name "*.md" | xargs -I {} oboyu index {}
```

### High CPU Usage

**Issue**: Oboyu consuming too much CPU

**Solutions**:
```bash
# Limit thread count
oboyu config set indexer.threads 2

# Use nice to lower priority
nice -n 10 oboyu index ~/Documents

# Process in background
nohup oboyu index ~/Documents &
```

## Japanese Language Issues

### Poor Japanese Search Results

**Issue**: Japanese text search not working well

**Solutions**:
```bash
# Use Japanese-optimized model
oboyu config set indexer.embedding_model "cl-nagoya/ruri-v3-30m"

# Smaller chunk size for Japanese
oboyu config set indexer.chunk_size 512

# Enable Japanese preprocessing
oboyu config set indexer.japanese_preprocessing true
```

### Encoding Problems

**Issue**: Japanese characters appear as garbage

**Solutions**:
```bash
# Set UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Force UTF-8 for specific files
oboyu index ~/japanese-docs --encoding utf-8

# Auto-detect encoding
oboyu index ~/mixed-docs --encoding auto
```

## Configuration Issues

### Configuration Not Loading

**Issue**: Changes to config file don't take effect

**Troubleshooting**:
```bash
# Check config file location
oboyu config show --source

# Validate config syntax
oboyu config validate

# Show effective configuration
oboyu config show --effective
```

### Invalid Configuration Values

**Issue**: Config validation errors

**Solutions**:
```bash
# Reset to defaults
oboyu config reset

# Fix specific issues
oboyu config set indexer.chunk_size 1024  # Must be power of 2
oboyu config set query.top_k 10            # Must be positive integer

# Check valid values
oboyu config help indexer.chunk_size
```

## Model and Download Issues

### Model Download Failures

**Issue**: Embedding models fail to download

**Solutions**:
```bash
# Check internet connection
ping huggingface.co

# Use mirror or proxy
export HF_ENDPOINT=https://hf-mirror.com

# Download manually
oboyu models download cl-nagoya/ruri-v3-30m

# Use local model cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Model Loading Errors

**Issue**: Models fail to load at runtime

**Solutions**:
```bash
# Check available models
oboyu models list

# Clear model cache
rm -rf ~/.cache/huggingface/transformers

# Use smaller model
oboyu config set indexer.embedding_model "sentence-transformers/paraphrase-MiniLM-L6-v2"
```

## Database Issues

### Database Locked

**Issue**: "Database is locked" error

**Solutions**:
```bash
# Kill existing processes
pkill -f oboyu

# Remove lock file
rm -f ~/.oboyu/*.lock

# Repair database
oboyu index repair --db-path ~/indexes/personal.db
```

### Database Corruption

**Issue**: Database file appears corrupted

**Solutions**:
```bash
# Check database integrity
oboyu index verify --db-path ~/indexes/personal.db

# Attempt repair
oboyu index repair --db-path ~/indexes/personal.db --force

# Last resort: rebuild
oboyu index rebuild --db-path ~/indexes/personal.db
```

## MCP Server Issues

### MCP Server Won't Start

**Issue**: Claude Desktop can't connect to Oboyu

**Troubleshooting**:
```bash
# Test MCP server manually
oboyu mcp serve --debug

# Check Claude Desktop logs
# macOS: ~/Library/Logs/Claude/
# Windows: %APPDATA%/Claude/logs/
# Linux: ~/.config/claude/logs/

# Verify configuration
cat ~/.config/claude/config.json
```

### MCP Connection Timeout

**Issue**: MCP requests timeout

**Solutions**:
```bash
# Increase timeout
oboyu mcp serve --timeout 60

# Reduce result count
oboyu mcp serve --max-results 5

# Optimize index
oboyu index optimize --db-path ~/indexes/personal.db
```

## General Debugging

### Enable Debug Logging

```bash
# Global debug mode
export OBOYU_LOG_LEVEL=DEBUG

# Debug specific operation
oboyu index ~/Documents --verbose --debug

# Save debug log
oboyu query "search" --debug > debug.log 2>&1
```

### Collect Diagnostic Information

```bash
# System information
oboyu diagnose

# Index statistics
oboyu index stats --all

# Performance metrics
oboyu benchmark
```

## Getting Help

### Before Asking for Help

1. **Check the FAQ**: Review this troubleshooting guide
2. **Search existing issues**: Check [GitHub issues](https://github.com/sonesuke/oboyu/issues)
3. **Collect information**:
```bash
# System info
oboyu version
python --version
uname -a

# Error logs
oboyu command --debug > error.log 2>&1
```

### How to Report Issues

Include this information in bug reports:

1. **Oboyu version**: `oboyu --version`
2. **Python version**: `python --version`
3. **Operating system**: `uname -a` (Linux/macOS) or system info (Windows)
4. **Full error message**: Copy complete error output
5. **Steps to reproduce**: Exact commands that cause the issue
6. **Configuration**: `oboyu config show` (remove sensitive paths)

### Community Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/sonesuke/oboyu/issues)
- **Discussions**: [Community Q&A](https://github.com/sonesuke/oboyu/discussions)
- **Documentation**: [Latest docs](https://sonesuke.github.io/oboyu/)

### Emergency Recovery

If Oboyu is completely broken:

```bash
# Nuclear option: complete reset
rm -rf ~/.oboyu
rm -rf ~/.cache/oboyu
pip uninstall oboyu
pip install oboyu

# Restore from backup
oboyu index restore ~/backup/oboyu-index.db --db-path ~/indexes/personal.db
```

Remember: Always backup important indices before making major changes!