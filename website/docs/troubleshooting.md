# Troubleshooting Guide

This guide covers common issues and their solutions when using Oboyu.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Indexing Problems](#indexing-problems)
- [Search Issues](#search-issues)
- [Performance Problems](#performance-problems)
- [Japanese Language Issues](#japanese-language-issues)
- [MCP Server Issues](#mcp-server-issues)
- [Model Download Issues](#model-download-issues)
- [Memory Issues](#memory-issues)
- [Getting Help](#getting-help)

## Installation Issues

### Issue: Installation fails with "error: Microsoft Visual C++ 14.0 is required" (Windows)

**Solution**: Install Microsoft C++ Build Tools
```bash
# Download and install from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Issue: Installation fails with "error: cannot compile sentencepiece"

**Solution**: Install system dependencies

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Using Homebrew
brew install cmake
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install gcc-c++ cmake
```

### Issue: "command not found: oboyu" after installation

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

### Issue: Python version compatibility error

**Solution**: Ensure Python 3.10 or higher is installed
```bash
# Check Python version
python --version

# Install Python 3.10+ using pyenv
pyenv install 3.10.11
pyenv global 3.10.11
```

## Indexing Problems

### Issue: "Permission denied" when indexing files

**Solution**: Check file permissions
```bash
# Check permissions
ls -la /path/to/documents

# Fix permissions (be careful with this)
chmod -R u+r /path/to/documents
```

### Issue: Indexing is very slow

**Solutions**:

1. **Use incremental indexing** for updates:
   ```bash
   oboyu index /path/to/documents --incremental
   ```

2. **Exclude unnecessary files**:
   ```bash
   oboyu index /path/to/documents --exclude "*.log,*.tmp,node_modules/"
   ```

3. **Index specific file types only**:
   ```bash
   oboyu index /path/to/documents --include "*.md,*.txt,*.pdf"
   ```

### Issue: "Out of memory" during indexing

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

### Issue: PDF or other binary files not being indexed

**Important**: Oboyu currently only supports text-based files. PDF, Word documents, and other binary formats are not supported and will not be indexed properly. They will be processed as raw text, which may result in garbled content.

**Supported formats**: Plain text (.txt), Markdown (.md), code files (.py, .js, etc.), and other text-based formats.

**Workaround**: Convert PDFs to text files using external tools before indexing.

## Search Issues

### Issue: No results found for queries

**Solutions**:

1. **Check if documents were indexed**:
   ```bash
   oboyu index --stats
   ```

2. **Try different search modes**:
   ```bash
   # Try semantic search
   oboyu query "your search" --mode semantic
   
   # Try keyword search
   oboyu query "your search" --mode keyword
   ```

3. **Use broader queries**:
   ```bash
   # Instead of: "Python asyncio event loop error"
   # Try: "Python asyncio error"
   ```

### Issue: Search results are not relevant

**Solutions**:

1. **Enable reranking** for better accuracy:
   ```bash
   oboyu query "your search" --rerank
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

### Issue: Interactive mode not working properly

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

### Issue: Searches are too slow

**Solutions**:

1. **Disable reranking** for faster searches:
   ```bash
   oboyu query "search" --no-rerank
   ```

2. **Limit results**:
   ```bash
   oboyu query "search" --limit 10
   ```

3. **Use keyword mode** for simple searches:
   ```bash
   oboyu query "exact phrase" --mode keyword
   ```

### Issue: High memory usage during search

**Solution**: Configure memory limits
```yaml
# oboyu_config.yaml
query_engine:
  search:
    max_candidates: 100  # Reduce from default
    rerank_batch_size: 8  # Reduce for less memory
```

## Japanese Language Issues

### Issue: Japanese text appears garbled

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

### Issue: Japanese tokenization not working

**Solution**: Install MeCab (optional but recommended)
```bash
# macOS
brew install mecab mecab-ipadic

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# Install Python binding
pip install mecab-python3
```

### Issue: Poor search results for Japanese queries

**Solutions**:

1. **Use Japanese-specific configuration**:
   ```yaml
   # oboyu_config.yaml
   indexer:
     language:
       japanese_tokenizer: "mecab"  # or "sentencepiece"
   ```

2. **Try different query formats**:
   ```bash
   # With spaces
   oboyu query "機械 学習 最適化"
   
   # Without spaces
   oboyu query "機械学習最適化"
   ```

## MCP Server Issues

### Issue: MCP server won't start

**Solutions**:

1. **Check if port is in use**:
   ```bash
   lsof -i :3333  # Default MCP port
   ```

2. **Start with debug logging**:
   ```bash
   oboyu mcp --log-level debug
   ```

### Issue: Claude Desktop can't connect to MCP server

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

### Issue: MCP tools not working correctly

**Solution**: Check server logs
```bash
# Enable debug logging
oboyu mcp --log-level debug

# Check MCP server status
curl http://localhost:3333/health
```

## Model Download Issues

### Issue: Model download fails or is interrupted

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
   ```

### Issue: "No space left on device" during model download

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

## Memory Issues

### Issue: "Killed" or "MemoryError" during operations

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
   ```

3. **Process files in smaller batches**:
   ```bash
   # Instead of indexing everything at once
   find /docs -name "*.md" | head -100 | xargs oboyu index
   ```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**:
   ```bash
   # Enable debug logging
   export OBOYU_LOG_LEVEL=DEBUG
   oboyu [command]
   ```

2. **Verify your configuration**:
   ```bash
   oboyu config --validate
   ```

3. **Get system information**:
   ```bash
   oboyu --version
   python --version
   uname -a
   ```

4. **Report issues**:
   - GitHub Issues: [github.com/sonesuke/oboyu/issues](https://github.com/sonesuke/oboyu/issues)
   - Include:
     - Error message
     - Steps to reproduce
     - System information
     - Configuration file (if customized)

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