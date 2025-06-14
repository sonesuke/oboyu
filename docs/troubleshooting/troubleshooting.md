# Troubleshooting Guide

This guide covers common issues and their solutions when using Oboyu. Quick solutions to common problems you might encounter, organized by type of issue for easy navigation.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Indexing Problems](#indexing-problems)
- [Search Issues](#search-issues)
- [Performance Problems](#performance-problems)
- [Japanese Language Issues](#japanese-language-issues)
- [MCP Server Issues](#mcp-server-issues)
- [Model Download Issues](#model-download-issues)
- [Memory Issues](#memory-issues)
- [Configuration Issues](#configuration-issues)
- [Database Issues](#database-issues)
- [General Debugging](#general-debugging)
- [Getting Help](#getting-help)

## Installation Issues

### Python Version Compatibility

**Issue**: `oboyu` installation fails with version error
```
ERROR: Package 'oboyu' requires a different Python: 3.8.0 not in '>=3.10'
```

**Solution**: Upgrade to Python 3.10 or higher
```bash
# Check current version
python --version

# Install Python 3.10+ using pyenv
pyenv install 3.10.11
pyenv global 3.10.11

# Or using conda
conda install python=3.10
```

### Build Dependencies Missing

**Issue**: Installation fails with compilation errors like "error: Microsoft Visual C++ 14.0 is required" (Windows) or "error: cannot compile sentencepiece"

**macOS Solution**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install build dependencies
brew install cmake pkg-config
```

**Linux (Ubuntu/Debian) Solution**:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libssl-dev
```

**Linux (Fedora/RHEL) Solution**:
```bash
sudo dnf install gcc-c++ cmake openssl-devel
```

**Windows Solution**:
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### "command not found: oboyu" after installation

**Solution**: Ensure the installation directory is in your PATH

```bash
# Check where oboyu is installed
which oboyu

# If using UV, add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
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
oboyu config set crawler.include_patterns "*.md,*.txt,*.py"
```

2. **Files excluded by pattern**:
```bash
# Check exclude patterns
oboyu config show | grep exclude_patterns

# Remove problematic exclusions
oboyu config unset crawler.exclude_patterns
```

### "Permission denied" when indexing files

**Solution**: Check file permissions
```bash
# Check permissions
ls -la /path/to/documents

# Fix permissions (be careful with this)
chmod -R u+r /path/to/documents
```

### Indexing is very slow or stuck

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

### "Out of memory" during indexing

**Solutions**:

1. **Reduce batch size** in configuration:
   ```yaml
   # oboyu_config.yaml
   indexer:
     processing:
       batch_size: 16  # Reduce from default 32
   ```

2. **Index in smaller chunks**:
   ```bash
   # Index subdirectories separately
   oboyu index /docs/part1
   oboyu index /docs/part2 --incremental
   ```

### PDF or other binary files not being indexed

**Important**: Oboyu currently only supports text-based files. PDF, Word documents, and other binary formats are not supported and will not be indexed properly. They will be processed as raw text, which may result in garbled content.

**Supported formats**: Plain text (.txt), Markdown (.md), code files (.py, .js, etc.), and other text-based formats.

**Workaround**: Convert PDFs to text files using external tools before indexing.

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

# Check if documents were indexed
oboyu index --stats
```

2. **Try different search modes**:
```bash
# Try BM25 for exact matches
oboyu query --query "exact phrase" --mode bm25

# Try vector search for concepts
oboyu query --query "concept description" --mode vector

# Try hybrid for best of both
oboyu query --query "search terms" --mode hybrid
```

3. **Use broader queries**:
```bash
# Instead of: "Python asyncio event loop error"
# Try: "Python asyncio error"

# Use quotes for exact phrases
oboyu query '"exact phrase here"'

# Use simpler terms
oboyu query --query "main keyword"
```

### Poor Search Quality

**Issue**: Search results are not relevant

**Solutions**:

1. **Enable reranking** for better accuracy:
   ```bash
   oboyu query --query "your search" --rerank
   oboyu config set indexer.use_reranker true
   ```

2. **Adjust search mode**:
   - Use `hybrid` for balanced results
   - Use `semantic` for concept-based search
   - Use `keyword` for exact matches

3. **Re-index with better chunking**:
   ```yaml
   # oboyu_config.yaml
   indexer:
     processing:
       chunk_size: 512  # Smaller chunks for more precise results
       chunk_overlap: 128
   ```

4. **Optimize chunk size**:
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
oboyu query --query "search term" --mode bm25

# Limit results
oboyu query --query "search term" --limit 5

# Add filters to narrow search
oboyu query --query "search term" --file-type md --days 30

# Disable reranking for faster searches
oboyu query --query "search" --no-rerank
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

### Interactive mode not working properly

**Solutions**:

1. **Check terminal compatibility**:
   ```bash
   # Ensure terminal supports ANSI colors
   echo $TERM
   # Should show something like: xterm-256color
   ```

2. **Try a different terminal**:
   - macOS: Use Terminal.app or iTerm2
   - Linux: Use gnome-terminal or konsole
   - Windows: Use Windows Terminal

## Performance Problems

### Memory Issues

**Issue**: "Killed" or "MemoryError" during operations, or high memory usage during search

**Solutions**:

1. **Monitor memory usage**:
   ```bash
   # While running oboyu
   top -p $(pgrep -f oboyu)
   ```

2. **Reduce memory usage**:
   ```yaml
   # oboyu_config.yaml
   indexer:
     processing:
       batch_size: 8  # Reduce batch size
       max_workers: 2  # Reduce parallel workers
   
   query_engine:
     embedding:
       batch_size: 16  # Reduce embedding batch
     search:
       max_candidates: 100  # Reduce from default
       rerank_batch_size: 8  # Reduce for less memory
   ```

3. **Process files in smaller batches**:
   ```bash
   # Instead of indexing everything at once
   find /docs -name "*.md" | head -100 | xargs oboyu index
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

### Japanese text appears garbled

**Solutions**:

1. **Check terminal encoding**:
   ```bash
   # Set UTF-8 encoding
   export LANG=en_US.UTF-8
   export LC_ALL=en_US.UTF-8
   ```

2. **Verify file encoding**:
   ```bash
   file -i japanese_document.txt
   # Should show: charset=utf-8 or shift_jis
   ```

### Japanese tokenization not working

**Solution**: Install MeCab (optional but recommended)
```bash
# macOS
brew install mecab mecab-ipadic

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# Install Python binding
pip install mecab-python3
```

### Poor search results for Japanese queries

**Solutions**:

1. **Use Japanese-specific configuration**:
   ```yaml
   # oboyu_config.yaml
   indexer:
     language:
       japanese_tokenizer: "mecab"  # or "sentencepiece"
   ```

2. **Use Japanese-optimized model**:
```bash
oboyu config set indexer.embedding_model "cl-nagoya/ruri-v3-30m"

# Smaller chunk size for Japanese
oboyu config set indexer.chunk_size 512

# Enable Japanese preprocessing
oboyu config set indexer.japanese_preprocessing true
```

3. **Try different query formats**:
   ```bash
   # With spaces
   oboyu query --query "機械 学習 最適化"
   
   # Without spaces
   oboyu query --query "機械学習最適化"
   ```

## MCP Server Issues

### MCP server won't start

**Issue**: Claude Desktop can't connect to Oboyu

**Troubleshooting**:
```bash
# Test MCP server manually
oboyu mcp serve --debug

# Check if port is in use
lsof -i :3333  # Default MCP port

# Start with debug logging
oboyu mcp --log-level debug

# Check Claude Desktop logs
# macOS: ~/Library/Logs/Claude/
# Windows: %APPDATA%/Claude/logs/
# Linux: ~/.config/claude/logs/

# Verify configuration
cat ~/.config/claude/config.json
```

### Claude Desktop can't connect to MCP server

**Solution**: Verify MCP configuration in Claude Desktop
```json
{
  "mcpServers": {
    "oboyu": {
      "command": "oboyu",
      "args": ["mcp"],
      "env": {}
    }
  }
}
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

# Check server logs
oboyu mcp --log-level debug

# Check MCP server status
curl http://localhost:3333/health
```

## Model Download Issues

### Model download fails or is interrupted

**Solutions**:

1. **Retry download**:
   ```bash
   oboyu --download-models
   ```

2. **Clear cache and retry**:
   ```bash
   # Remove corrupted downloads
   rm -rf ~/.cache/oboyu/models
   oboyu --download-models
   ```

3. **Use alternative download method**:
   ```bash
   # Set custom timeout
   export OBOYU_DOWNLOAD_TIMEOUT=3600
   oboyu --download-models
   
   # Use mirror or proxy
   export HF_ENDPOINT=https://hf-mirror.com
   
   # Download manually
   oboyu models download cl-nagoya/ruri-v3-30m
   
   # Use local model cache
   export TRANSFORMERS_CACHE=/path/to/cache
   ```

### "No space left on device" during model download

**Solutions**:

1. **Check available space**:
   ```bash
   df -h ~/.cache/oboyu
   ```

2. **Change cache directory**:
   ```bash
   export OBOYU_CACHE_DIR=/path/with/more/space
   oboyu --download-models
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

## General Debugging

### Enable Debug Logging

```bash
# Global debug mode
export OBOYU_LOG_LEVEL=DEBUG

# Debug specific operation
oboyu index ~/Documents --verbose --debug

# Save debug log
oboyu query --query "search" --debug > debug.log 2>&1
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

If you're still experiencing issues:

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

# Check the logs
export OBOYU_LOG_LEVEL=DEBUG
oboyu [command]

# Verify your configuration
oboyu config --validate

# Get system information
oboyu --version
python --version
uname -a
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

## Quick Fixes Checklist

When encountering issues, try these steps in order:

1. ✓ Update to the latest version: `pip install --upgrade oboyu`
2. ✓ Clear and rebuild index: `oboyu index --clear && oboyu index /path`
3. ✓ Check disk space: `df -h`
4. ✓ Verify Python version: `python --version` (need 3.10+)
5. ✓ Run with debug logging: `OBOYU_LOG_LEVEL=DEBUG oboyu [command]`
6. ✓ Try with default configuration (remove custom config file)
7. ✓ Restart your terminal session
8. ✓ Check file permissions on documents and index directory

## Common Error Messages Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'oboyu'` | Installation issue | Reinstall with `pip install oboyu` |
| `FileNotFoundError: [Errno 2] No such file or directory` | Invalid path | Check file path exists |
| `PermissionError: [Errno 13] Permission denied` | No read access | Fix file permissions |
| `JSONDecodeError: Expecting value` | Corrupted index | Clear and rebuild index |
| `ConnectionError: Failed to download model` | Network issue | Check internet connection |
| `RuntimeError: CUDA out of memory` | GPU memory full | Disable GPU or reduce batch size |
| `UnicodeDecodeError` | Encoding issue | Oboyu auto-detects, but check file encoding |

Remember: Always backup important indices before making major changes!