---
id: japanese-support
title: Japanese Language Support Details
sidebar_position: 4
---

# Japanese Language Support Details

Comprehensive guide to Oboyu's Japanese language capabilities, optimizations, and best practices for indexing and searching Japanese content.

## Overview

Oboyu is specifically optimized for Japanese text processing, offering:

- **Native Japanese tokenization**: Proper handling of hiragana, katakana, and kanji
- **Japanese-optimized embeddings**: Semantic understanding of Japanese concepts
- **Mixed-language support**: Seamless handling of Japanese-English mixed content
- **Character encoding detection**: Automatic handling of various Japanese encodings

## Japanese Language Features

### Character Set Support

Oboyu handles all Japanese character sets:

| Character Set | Description | Example | Support Level |
|---------------|-------------|---------|---------------|
| **Hiragana** | Native Japanese syllabary | あいうえお | Full |
| **Katakana** | Foreign word syllabary | アイウエオ | Full |
| **Kanji** | Chinese characters | 漢字文書 | Full |
| **Romaji** | Romanized Japanese | konnichiwa | Full |
| **Mixed** | Combined character sets | 今日はGood day | Full |

### Text Processing

#### Tokenization
```bash
# Japanese text is properly tokenized
oboyu index ~/日本語文書 --verbose

# Shows proper token boundaries:
# Input: "今日は良い天気ですね"
# Tokens: ["今日", "は", "良い", "天気", "です", "ね"]
```

#### Normalization
- Unicode normalization (NFC/NFD)
- Full-width/half-width character normalization
- Katakana-hiragana normalization (optional)

### Search Capabilities

#### Japanese Query Types

**1. Exact Character Matching**
```bash
# Search for exact kanji
oboyu query "機械学習"

# Search for hiragana
oboyu query "こんにちは"

# Search for katakana
oboyu query "コンピューター"
```

**2. Semantic Search in Japanese**
```bash
# Conceptual search
oboyu query "人工知能について教えて" --mode vector

# Natural language questions
oboyu query "明日の会議の議題は何ですか" --mode semantic
```

**3. Mixed Language Search**
```bash
# Japanese-English mixed
oboyu query "machine learning 機械学習"

# Technical terms
oboyu query "API ドキュメント"

# Code and comments
oboyu query "Python プログラミング"
```

## Configuration for Japanese

### Optimal Settings

```yaml
# Japanese-optimized configuration
indexer:
  chunk_size: 512                    # Smaller for Japanese character density
  chunk_overlap: 128                 # 25% overlap
  embedding_model: "cl-nagoya/ruri-v3-30m"  # Japanese-optimized model
  use_reranker: true                 # Improves Japanese result quality

crawler:
  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.org"
    - "*.tex"
  encoding: "auto"                   # Auto-detect Japanese encodings

query:
  default_mode: "hybrid"             # Best for mixed content
  top_k: 10
```

### Encoding Handling

Oboyu automatically detects common Japanese encodings:

```bash
# Auto-detect encoding
oboyu index ~/日本語ファイル --encoding auto

# Force specific encoding
oboyu index ~/shift-jis-files --encoding shift_jis
oboyu index ~/euc-jp-files --encoding euc-jp
oboyu index ~/utf8-files --encoding utf-8
```

**Supported Encodings**:
- UTF-8 (recommended)
- Shift-JIS (Windows Japanese)
- EUC-JP (Unix Japanese)
- ISO-2022-JP (Email Japanese)

## Japanese-Specific Use Cases

### Academic Papers (Japanese)

```yaml
# Configuration for Japanese academic papers
indexer:
  db_path: "~/research/日本語論文.db"
  chunk_size: 1024                   # Larger for academic context
  chunk_overlap: 256
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.tex"                        # LaTeX papers
    - "*.md"                         # Markdown notes
    - "*.txt"                        # Plain text
  exclude_patterns:
    - "*/backup/*"
    - "*.aux"
    - "*.log"
```

**Usage Examples**:
```bash
# Search for research topics
oboyu query "深層学習の応用" --mode vector

# Find methodology sections
oboyu query "手法 実験方法" --mode hybrid

# Search for specific authors
oboyu query "田中氏の研究" --mode semantic
```

### Business Documents

```yaml
# Configuration for Japanese business documents
indexer:
  db_path: "~/work/業務文書.db"
  chunk_size: 768
  chunk_overlap: 192
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.md"                         # Markdown documents
    - "*.txt"                        # Text files
    - "会議録/**/*"                  # Meeting minutes
    - "報告書/**/*"                  # Reports
```

**Usage Examples**:
```bash
# Find meeting minutes
oboyu query "会議議事録" --days 30

# Search for project status
oboyu query "プロジェクト進捗" --mode semantic

# Find budget information
oboyu query "予算 費用" --mode hybrid
```

### Personal Notes (Japanese)

```yaml
# Configuration for personal Japanese notes
indexer:
  db_path: "~/notes/個人メモ.db"
  chunk_size: 512
  chunk_overlap: 128
  embedding_model: "cl-nagoya/ruri-v3-30m"
  use_reranker: true

crawler:
  include_patterns:
    - "*.md"
    - "*.txt"
    - "*.org"
    - "日記/**/*"                    # Diary entries
    - "メモ/**/*"                    # Memo files
```

**Usage Examples**:
```bash
# Search diary entries
oboyu query "今日の出来事" --mode semantic

# Find learning notes
oboyu query "勉強したこと" --mode vector

# Search for ideas
oboyu query "アイデア 企画" --mode hybrid
```

## Japanese Text Analysis

### Character Type Detection

```bash
# Search by character type
oboyu query "[ひらがな]" --regex  # Hiragana only
oboyu query "[カタカナ]" --regex  # Katakana only
oboyu query "[漢字]" --regex      # Kanji only
```

### Length Considerations

Japanese text has different density characteristics:

| Content Type | Recommended Chunk Size | Reason |
|-------------|----------------------|---------|
| **Technical docs** | 512-768 | Dense information |
| **Academic papers** | 768-1024 | Context preservation |
| **Personal notes** | 512 | Quick retrieval |
| **Business docs** | 768 | Balanced approach |

## Advanced Japanese Features

### Semantic Understanding

The Japanese-optimized model understands:

**Synonyms and Related Terms**:
```bash
# These queries find related content:
oboyu query "AI" --mode vector          # Also finds: 人工知能, 機械学習
oboyu query "会議" --mode vector        # Also finds: ミーティング, 打ち合わせ
oboyu query "問題" --mode vector        # Also finds: 課題, イシュー
```

**Cultural Context**:
```bash
# Understands Japanese business concepts
oboyu query "根回し" --mode vector      # Finds related planning documents
oboyu query "改善" --mode vector        # Understands kaizen context
oboyu query "おもてなし" --mode vector  # Hospitality-related content
```

### Cross-Language Understanding

```bash
# Finds both Japanese and English content
oboyu query "machine learning 機械学習" --mode hybrid

# Technical terms work in both languages
oboyu query "API documentation" --mode vector  # Finds APIドキュメント
oboyu query "データベース設計" --mode vector  # Finds database design docs
```

## Performance Optimization for Japanese

### Memory Usage

Japanese text typically requires more memory due to:
- Complex character encoding
- Larger vocabulary size
- Rich semantic relationships

**Optimization strategies**:
```bash
# Reduce batch size for Japanese
oboyu config set indexer.batch_size 16

# Use memory limits
export OBOYU_MEMORY_LIMIT=4GB

# Process in smaller chunks
oboyu index ~/large-japanese-collection --chunk-size 512
```

### Processing Speed

**Faster processing for Japanese content**:
```yaml
indexer:
  chunk_size: 512                    # Smaller chunks process faster
  use_reranker: false               # Disable for initial speed
  
query:
  default_mode: "bm25"              # Faster for exact matches
```

**Quality-optimized processing**:
```yaml
indexer:
  chunk_size: 768                    # Balanced size
  chunk_overlap: 192                # Good context preservation
  use_reranker: true                # Better quality

query:
  default_mode: "hybrid"            # Best overall results
```

## Common Japanese Text Patterns

### Search Patterns

**Date Patterns**:
```bash
# Japanese date formats
oboyu query "2024年1月15日" --mode bm25
oboyu query "令和6年" --mode bm25
oboyu query "平成.*年" --regex
```

**Name Patterns**:
```bash
# Japanese names
oboyu query "田中.*さん" --regex
oboyu query "山田部長" --mode bm25
oboyu query "佐藤氏" --mode semantic
```

**Business Patterns**:
```bash
# Meeting-related
oboyu query "会議|ミーティング|打ち合わせ" --regex
oboyu query "議事録" --mode bm25
oboyu query "決定事項" --mode semantic

# Project-related
oboyu query "プロジェクト.*進捗" --regex
oboyu query "作業完了" --mode bm25
oboyu query "次のステップ" --mode semantic
```

## Troubleshooting Japanese Issues

### Character Encoding Problems

**Issue**: Japanese characters appear as garbage
```bash
# Check file encoding
file ~/japanese-file.txt

# Force UTF-8
oboyu index ~/japanese-files --encoding utf-8

# Auto-detect encoding
oboyu index ~/japanese-files --encoding auto
```

### Poor Search Results

**Issue**: Japanese search returns poor results

**Solutions**:
```bash
# Use Japanese-optimized model
oboyu config set indexer.embedding_model "cl-nagoya/ruri-v3-30m"

# Reduce chunk size
oboyu config set indexer.chunk_size 512

# Enable reranking
oboyu config set indexer.use_reranker true

# Try different search modes
oboyu query "日本語クエリ" --mode vector
oboyu query "日本語クエリ" --mode hybrid
```

### Performance Issues

**Issue**: Slow processing of Japanese content

**Solutions**:
```bash
# Reduce batch size
oboyu config set indexer.batch_size 8

# Use smaller chunks
oboyu config set indexer.chunk_size 512

# Limit memory usage
export OBOYU_MEMORY_LIMIT=2GB
```

## Best Practices for Japanese Content

### Document Organization

1. **Use consistent encoding**: Prefer UTF-8 for all new documents
2. **Organize by content type**: Separate technical, business, and personal content
3. **Include metadata**: Use frontmatter for document categorization

### Search Strategies

1. **Start with hybrid mode**: Best balance for Japanese content
2. **Use semantic search for concepts**: Better understanding of Japanese nuances
3. **Combine search modes**: Use different modes for different query types

### Index Management

1. **Regular updates**: Japanese content often includes date-specific information
2. **Monitor performance**: Japanese processing can be memory-intensive
3. **Backup indices**: Japanese indices can be time-consuming to rebuild

## Examples and Templates

### Japanese Documentation Search
```bash
#!/bin/bash
# japanese-doc-search.sh

# Search for Japanese documentation
search_japanese_docs() {
    local query="$1"
    
    echo "=== Japanese Document Search ==="
    echo "Query: $query"
    echo
    
    # Hybrid search for best results
    oboyu query "$query" \
        --mode hybrid \
        --db-path ~/indexes/japanese-docs.db \
        --limit 10 \
        --context 300
}

# Usage examples
search_japanese_docs "機械学習の基礎"
search_japanese_docs "プロジェクト管理"
search_japanese_docs "API仕様書"
```

### Multi-language Search
```bash
#!/bin/bash
# multi-language-search.sh

# Search across Japanese and English content
multi_search() {
    local japanese_term="$1"
    local english_term="$2"
    
    echo "=== Multi-language Search ==="
    echo "Japanese: $japanese_term"
    echo "English: $english_term"
    echo
    
    # Combined search
    oboyu query "$japanese_term OR $english_term" \
        --mode hybrid \
        --limit 15
}

# Usage
multi_search "機械学習" "machine learning"
multi_search "データベース" "database"
```

## Next Steps

- Review [Search Patterns](../usage-examples/search-patterns.md) for Japanese-specific patterns
- Explore [Configuration](../configuration-optimization/configuration.md) for optimization
- Check [Troubleshooting](troubleshooting.md) for specific issues