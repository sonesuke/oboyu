---
id: indexing-strategies
title: Indexing Best Practices
sidebar_position: 2
---

# Indexing Best Practices

Master the art of efficient document indexing with these proven strategies. Learn how to optimize indexing for different scenarios, manage large collections, and maintain high-quality search indices.

## Understanding Indexing

### What Happens During Indexing

When you run `oboyu index`, the system:

1. **Discovers** files matching your patterns
2. **Extracts** text content from each file
3. **Chunks** documents into searchable segments
4. **Embeds** text into vector representations
5. **Stores** everything in an optimized database

### Key Indexing Concepts

- **Chunks**: Document segments for processing
- **Embeddings**: Vector representations of text
- **BM25**: Traditional keyword index
- **HNSW**: Vector similarity index

## Indexing Strategies by Scale

### Small Collections (&lt;1,000 files)

For personal notes or small projects:

```bash
# Simple one-time index
oboyu index ~/Documents --db-path ~/indexes/personal.db

# With custom settings
oboyu index ~/Notes \
  --chunk-size 1536 \
  --overlap 384 \
  --db-path ~/indexes/notes.db
```

**Best Practices:**
- Use larger chunk sizes (1536-2048)
- Enable reranking for quality
- Index everything at once

### Medium Collections (1,000-10,000 files)

For team documentation or research:

```bash
# Index by category
oboyu index ~/docs/api --db-path ~/indexes/api-docs.db
oboyu index ~/docs/guides --db-path ~/indexes/guides.db
oboyu index ~/docs/reference --db-path ~/indexes/reference.db

# Merge indices for unified search
oboyu index merge api-docs guides reference --db-path ~/indexes/all-docs.db
```

**Best Practices:**
- Organize by logical categories
- Use moderate chunk sizes (1024)
- Consider separate indices

### Large Collections (10,000+ files)

For enterprise repositories:

```bash
# Batch indexing by directory
for dir in ~/repository/*/; do
    oboyu index "$dir" \
      --db-path ~/indexes/example.db \
      --chunk-size 512 \
      --overlap 128
done

# Parallel indexing
find ~/docs -type d -maxdepth 1 | \
  parallel -j 4 'oboyu index {} --db-path ~/indexes/.db{/}'
```

**Best Practices:**
- Use smaller chunks (512) for memory efficiency
- Minimal overlap to save space
- Consider distributed indexing

## Incremental Indexing

### Update Strategies

#### 1. Time-Based Updates
```bash
# Index only recent changes
oboyu index ~/Documents \
  --update \
  --modified-after "7 days ago"

# Scheduled updates via cron
# 0 2 * * * oboyu index ~/Documents --update
```

#### 2. Event-Based Updates
```bash
# Watch for changes
oboyu index ~/Documents --watch

# Using inotify (Linux)
inotifywait -m -r ~/Documents -e modify,create |
  while read path event file; do
    oboyu index "$path/$file" --update
  done
```

#### 3. Smart Updates
```bash
# Check what needs updating
oboyu index status ~/Documents

# Update only changed files
oboyu index ~/Documents --update --dry-run
oboyu index ~/Documents --update
```

### Handling Deletions

```bash
# Remove deleted files from index
oboyu index ~/Documents --update --prune

# Clean up orphaned entries
oboyu index clean --db-path ~/indexes/personal.db
```

## Optimizing Chunk Strategies

### Choosing Chunk Size

#### Small Chunks (256-512)
```yaml
indexer:
  chunk_size: 512
  chunk_overlap: 128
```

**Good for:**
- Code files
- Technical documentation
- Precise retrieval needs

#### Medium Chunks (1024)
```yaml
indexer:
  chunk_size: 1024
  chunk_overlap: 256
```

**Good for:**
- General documents
- Mixed content types
- Balanced performance

#### Large Chunks (2048-4096)
```yaml
indexer:
  chunk_size: 2048
  chunk_overlap: 512
```

**Good for:**
- Long-form content
- Academic papers
- Context-heavy documents

### Overlap Strategies

```python
# Overlap formula
optimal_overlap = chunk_size * 0.25  # 25% overlap

# Examples:
# chunk_size: 512  → overlap: 128
# chunk_size: 1024 → overlap: 256
# chunk_size: 2048 → overlap: 512
```

## Language-Specific Indexing

### Japanese Documents

```yaml
# Optimized for Japanese
indexer:
  chunk_size: 512          # Smaller for character density
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
  encoding: "auto"         # Handles Shift-JIS, UTF-8
```

### Mixed Language Collections

```bash
# Index with language detection
oboyu index ~/multilingual \
  --detect-language \
  --chunk-by-language

# Separate indices by language
oboyu index ~/docs/en --db-path ~/indexes/docs-en.db
oboyu index ~/docs/ja --db-path ~/indexes/docs-ja.db
oboyu index ~/docs/zh --db-path ~/indexes/docs-zh.db
```

## Performance Optimization

### Memory Management

```bash
# Limit memory usage
oboyu index ~/large-collection \
  --memory-limit 4GB \
  --batch-size 16

# Monitor memory during indexing
oboyu index ~/Documents --verbose --metrics
```

### CPU Optimization

```bash
# Use multiple threads
oboyu index ~/Documents --threads 8

# Limit CPU usage
nice -n 10 oboyu index ~/Documents

# Background indexing
nohup oboyu index ~/Documents &> indexing.log &
```

### Storage Optimization

```bash
# Use fast storage
oboyu index ~/Documents \
  --db-path /ssd/oboyu/index.db

# Compress index
oboyu index optimize --db-path ~/indexes/personal.db

# Check index size
oboyu index info --db-path ~/indexes/personal.db --detailed
```

## Advanced Indexing Patterns

### Filtered Indexing

```bash
# Index only specific content
oboyu index ~/projects \
  --include "*.py" \
  --exclude "*test*" \
  --min-size 1KB \
  --max-size 1MB

# Content-based filtering
oboyu index ~/Documents \
  --content-filter "project|report|analysis"
```

### Metadata-Enhanced Indexing

```bash
# Extract and index metadata
oboyu index ~/Documents \
  --extract-metadata \
  --metadata-fields "author,date,tags"

# Custom metadata extraction
oboyu index ~/notes \
  --metadata-regex "^---\n(.*?)\n---" \
  --metadata-format yaml
```

### Hierarchical Indexing

```bash
# Preserve directory structure
oboyu index ~/knowledge-base \
  --preserve-hierarchy \
  --path-as-metadata

# Query with path context
oboyu query "API" --filter "path:/docs/api/*"
```

## Quality Control

### Index Validation

```bash
# Verify index integrity
oboyu index verify --db-path ~/indexes/personal.db

# Check for issues
oboyu index diagnose --db-path ~/indexes/personal.db

# Repair if needed
oboyu index repair --db-path ~/indexes/personal.db
```

### Content Quality

```bash
# Find duplicate content
oboyu index duplicates --db-path ~/indexes/personal.db

# Find empty or tiny files
oboyu index ~/Documents \
  --min-words 10 \
  --report-skipped

# Analyze index quality
oboyu index analyze --db-path ~/indexes/personal.db
```

## Indexing Workflows

### Development Documentation

```bash
#!/bin/bash
# index-dev-docs.sh

# Index source code
oboyu index ./src \
  --include "*.py,*.js" \
  --exclude "*test*" \
  --db-path ~/indexes/code.db

# Index documentation  
oboyu index ./docs \
  --include "*.md,*.rst" \
  --db-path ~/indexes/docs.db

# Index comments from code
oboyu index ./src \
  --extract-comments \
  --db-path ~/indexes/comments.db

# Combine for unified search
oboyu index merge code docs comments --db-path ~/indexes/project.db
```

### Research Papers

```bash
#!/bin/bash
# index-research.sh

# Index by year
for year in {2020..2024}; do
  oboyu index ~/papers/$year \
    --db-path ~/indexes/example.db \
    --chunk-size 2048 \
    --extract-metadata
done

# Create master index
oboyu index merge papers-* --db-path ~/indexes/all-papers.db

# Extract citations
oboyu index ~/papers \
  --extract-citations \
  --db-path ~/indexes/citations.db
```

## Monitoring and Maintenance

### Index Health

```bash
# Regular health checks
oboyu index health --all

# Monitor growth
oboyu index stats --db-path ~/indexes/personal.db --timeline

# Set up alerts
oboyu index monitor \
  --alert-size 10GB \
  --alert-age 30d
```

### Maintenance Schedule

```bash
# Daily: Update recent changes
0 2 * * * oboyu index ~/Documents --update --modified-after 1d

# Weekly: Full update  
0 3 * * 0 oboyu index ~/Documents --update

# Monthly: Optimize
0 4 1 * * oboyu index optimize --all

# Quarterly: Full reindex
0 5 1 */3 * oboyu index ~/Documents --force
```

## Troubleshooting Indexing

### Common Issues

**"Indexing is slow"**
- Reduce chunk size
- Use SSD storage
- Disable reranker temporarily
- Index in batches

**"Out of memory"**
```bash
# Reduce batch size
oboyu index ~/Documents --batch-size 8

# Process files individually
find ~/Documents -type f -name "*.md" | \
  xargs -I {} oboyu index {} --update
```

**"Index corrupted"**
```bash
# Backup and rebuild
cp ~/.oboyu/index.db ~/.oboyu/index.db.bak
oboyu index ~/Documents --force --clean
```

## Best Practices Summary

1. **Start Small**: Test with subset first
2. **Organize First**: Structure improves results
3. **Regular Updates**: Keep index fresh
4. **Monitor Health**: Prevent issues early
5. **Backup Indices**: Before major changes
6. **Document Strategy**: Record what works

## Next Steps

- Configure [Search Optimization](search-optimization.md) for your indexed content
- Learn about [Performance Tuning](performance-tuning.md) for faster indexing
- Explore [CLI Workflows](../integration/cli-workflows.md) for automation