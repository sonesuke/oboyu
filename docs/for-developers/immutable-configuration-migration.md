# Immutable Configuration Migration Guide

This guide explains the new immutable configuration system introduced to fix parameter override issues in Oboyu.

## Overview

The new immutable configuration system addresses the most reported user experience issue: user-specified parameters (especially `use_reranker=True`) being silently overridden by default values.

### Key Improvements

- **Explicit Source Tracking**: Every configuration value tracks its source (CLI, file, environment, default)
- **Guaranteed Precedence**: CLI arguments always take precedence, followed by config files, environment variables, and defaults
- **Immutable Contexts**: Configuration objects are immutable, preventing accidental overwrites
- **Clear Logging**: Comprehensive logs show exactly where each configuration value comes from

## What's Changed

### 1. Configuration Precedence (New Behavior)

**Before**: Configuration values could be silently overridden by defaults
```yaml
# config.yaml
query:
  rerank: true

# User runs: oboyu query "test" --rerank=false
# Result: reranker might still be enabled due to config file
```

**After**: CLI arguments always win
```yaml
# config.yaml  
query:
  rerank: true

# User runs: oboyu query "test" --rerank=false
# Result: reranker is GUARANTEED to be disabled
# Logs show: "🎯 Reranker: false (EXPLICITLY set from CLI)"
```

### 2. Enhanced Logging

The new system provides clear visibility into configuration resolution:

```
🔧 Resolving search configuration...
📋 All configuration values:
  ⌨️ From CLI:
    search.use_reranker = true
    search.top_k = 5
  📄 From FILE:
    search.reranker_model = cl-nagoya/ruri-reranker-small
  ⚙️ From DEFAULT:
    search.reranker_top_k = 3

✅ Search configuration resolved:
  📝 Query: my search query
  🔍 Mode: HYBRID
  🔢 Top-k: 5 (from CLI)
  🎯 Reranker: true (EXPLICITLY set from CLI)
  🔥 Reranker EXPLICITLY ENABLED via CLI - will be used regardless of config defaults
```

### 3. Configuration Key Changes

Some configuration keys have been normalized for consistency:

| Old Key | New Key | Notes |
|---------|---------|-------|
| `query.rerank` | `search.use_reranker` | Unified naming |
| `query.rerank_model` | `search.reranker_model` | Consistent prefix |
| `indexer.use_reranker` | `indexer.use_reranker` | No change |

**Backward Compatibility**: Old keys still work and are automatically mapped to new keys.

## Migration Steps

### 1. Update Configuration Files (Optional)

If you want to use the new unified naming:

```yaml
# Before
query:
  rerank: true
  rerank_model: "cl-nagoya/ruri-reranker-small"
  top_k: 10

# After (recommended)
search:
  use_reranker: true
  reranker_model: "cl-nagoya/ruri-reranker-small"
  top_k: 10
```

**Note**: The old format still works due to automatic key mapping.

### 2. CLI Usage (No Changes Required)

CLI commands work exactly the same:

```bash
# These commands work as before, but now with guaranteed precedence
oboyu query "test" --rerank=true
oboyu query "test" --top-k=5 --rerank=false
```

### 3. Programmatic Usage

If you're using Oboyu programmatically, you can now use the new configuration system:

```python
from oboyu.config import ConfigurationResolver, ConfigSource, SearchContext

# Create resolver
resolver = ConfigurationResolver()

# Load from file
resolver.load_from_dict(config_dict, ConfigSource.FILE)

# Set CLI overrides (highest precedence)
resolver.set_from_cli_args(use_reranker=True, top_k=10)

# Resolve final configuration
config = resolver.resolve_search_config("query", SearchMode.HYBRID)

# Config is immutable and tracks sources
print(f"Reranker enabled: {config.use_reranker}")
print(f"Source: {config.sources['search.use_reranker']}")
```

## Troubleshooting

### Issue: Configuration Not Working as Expected

**Solution**: Enable detailed logging to see configuration resolution:

```bash
# Enable debug logging
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.INFO)
# Run your query here
"
```

The logs will show exactly where each configuration value comes from.

### Issue: Conflicting Configuration Values

The system now detects and warns about conflicts:

```
⚠️ Configuration conflicts detected:
  ⚠️ Reranker setting conflict: search.use_reranker=true (CLI) vs indexer.use_reranker=false (FILE)
```

**Solution**: Align your configuration to avoid conflicts, or rely on the precedence system.

### Issue: Legacy Configuration Not Loading

**Problem**: Using very old configuration format

**Solution**: Update to use supported keys:

```yaml
# Supported legacy format
query:
  rerank: true          # Maps to search.use_reranker
  rerank_model: "..."   # Maps to search.reranker_model
  top_k: 10            # Maps to search.top_k

indexer:
  use_reranker: false   # Stays as indexer.use_reranker
```

## Configuration Reference

### Search/Query Configuration

| Key | Type | Default | Source Priority | Description |
|-----|------|---------|----------------|-------------|
| `search.use_reranker` | bool | `true` | CLI > File > Default | Enable reranking for search |
| `search.reranker_model` | string | `"cl-nagoya/ruri-reranker-small"` | CLI > File > Default | Reranker model to use |
| `search.top_k` | int | `10` | CLI > File > Default | Number of results to return |
| `search.reranker_top_k` | int | `3` | CLI > File > Default | Number of results after reranking |

### Indexer Configuration

| Key | Type | Default | Source Priority | Description |
|-----|------|---------|----------------|-------------|
| `indexer.use_reranker` | bool | `false` | CLI > File > Default | Enable reranker during indexing |
| `indexer.reranker_model` | string | `"cl-nagoya/ruri-reranker-small"` | CLI > File > Default | Reranker model for indexing |
| `indexer.reranker_device` | string | `"cpu"` | CLI > File > Default | Device for reranker |
| `indexer.reranker_use_onnx` | bool | `false` | CLI > File > Default | Use ONNX for reranker |

## Implementation Details

### Source Precedence Order

1. **CLI** (Highest): Command-line arguments (`--rerank=true`)
2. **FILE**: Configuration files (YAML/JSON)
3. **ENV**: Environment variables
4. **DEFAULT** (Lowest): System defaults

### Immutable Configuration Objects

All resolved configuration objects are immutable (frozen dataclasses), preventing accidental modification:

```python
config = resolver.resolve_search_config("query", SearchMode.HYBRID)
# config.use_reranker = False  # This would raise an error
```

### Explicit Value Tracking

The system distinguishes between explicit user settings and defaults:

```python
# Check if user explicitly set a value
if resolver.builder.has_explicit_value("search.use_reranker"):
    print("User explicitly set reranker preference")
else:
    print("Using default reranker setting")
```

## Benefits

1. **Predictable Behavior**: User settings are never silently overridden
2. **Better Debugging**: Clear logs show configuration resolution process
3. **Conflict Detection**: Automatic detection of configuration conflicts
4. **Source Transparency**: Always know where a configuration value came from
5. **Backward Compatibility**: Existing configurations continue to work

## Need Help?

If you encounter issues with the new configuration system:

1. Enable debug logging to see configuration resolution
2. Check for conflict warnings in the logs
3. Verify your configuration keys match the reference above
4. Report issues with detailed logs at: https://github.com/your-repo/issues

The new system is designed to be more reliable and transparent while maintaining full backward compatibility.