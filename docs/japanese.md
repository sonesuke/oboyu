# Japanese Language Support in Oboyu

Oboyu provides first-class support for Japanese text processing, making it one of the most effective semantic search tools for Japanese documents.

## Overview

Japanese text processing presents unique challenges that Oboyu addresses comprehensively:

- **No word boundaries**: Japanese text doesn't use spaces to separate words
- **Multiple writing systems**: Hiragana, Katakana, Kanji, and Romaji often mixed within documents
- **Encoding variations**: Historical use of multiple character encodings
- **Complex tokenization**: Proper morphological analysis required for meaningful search

## Tokenization

### MeCab Integration

Oboyu uses MeCab (via the fugashi library) for accurate Japanese tokenization:

```python
# Example tokenization
Input:  "機械学習ではPythonがよく使われています"
Output: ["機械学習", "Python", "使わ", "れる"]
```

**Features:**
- **Morphological analysis**: Breaks text into meaningful units (morphemes)
- **Part-of-speech filtering**: Extracts content words (nouns, verbs, adjectives)
- **Inflection handling**: Normalizes verb and adjective forms
- **Compound word recognition**: Handles complex Japanese compound terms

### Fallback Tokenization

When MeCab is unavailable, Oboyu provides a fallback tokenizer:

```python
# Simple character-based fallback
Input:  "機械学習"
Output: ["機", "械", "学", "習"]
```

**When fallback is used:**
- MeCab installation issues
- Performance-critical scenarios where simple tokenization suffices
- Cross-platform compatibility requirements

## Character Normalization

### Unicode Normalization

All Japanese text undergoes NFKC (Normalization Form KC) processing:

```python
# Example normalization
Input:  "ｈｅｌｌｏ　ｗｏｒｌｄ"  # Full-width
Output: "hello world"              # Half-width
```

### Character Variant Handling

Oboyu handles common Japanese character variants:

```python
# Hiragana/Katakana normalization
Input:  "コンピュータ" (Katakana)
Output: "こんぴゅーた" (Hiragana) # When configured
```

## Encoding Detection

### Supported Encodings

Oboyu automatically detects and handles multiple Japanese encodings:

- **UTF-8**: Modern standard encoding
- **Shift-JIS**: Legacy Windows encoding
- **EUC-JP**: Legacy Unix encoding
- **ISO-2022-JP**: Legacy email encoding

### Configuration

Enable encoding detection in your configuration:

```yaml
crawler:
  japanese_encodings:
    - "utf-8"
    - "shift-jis" 
    - "euc-jp"
    - "iso-2022-jp"
```

### CLI Usage

```bash
# Enable Japanese encoding detection
oboyu index ./docs --japanese-encodings "utf-8,shift-jis,euc-jp"

# Use with specific patterns for Japanese files
oboyu index ./docs --include-patterns "*.txt,*.md" --japanese-encodings "shift-jis,euc-jp"
```

## Optimized Models

### Ruri v3 Embedding Model

Oboyu uses the Ruri v3 model specifically optimized for Japanese text:

**Model Details:**
- **Model**: `cl-nagoya/ruri-v3-30m`
- **Size**: ~90MB download
- **Optimization**: Japanese and multilingual content
- **Performance**: Excellent semantic understanding for Japanese queries

**Example Performance:**
```python
# Semantic similarity (high scores indicate good understanding)
Query: "機械学習のアルゴリズム"
Results:
- "深層学習の手法について" (Score: 0.89)
- "AIアルゴリズムの比較" (Score: 0.85)
- "プログラミング言語" (Score: 0.32)
```

### Ruri Reranker Models

Japanese-optimized reranker models for improved accuracy:

- **Small model**: `cl-nagoya/ruri-reranker-small` (~400MB memory)
- **Large model**: `cl-nagoya/ruri-v3-reranker-310m` (~1.2GB memory)

## Search Modes for Japanese Content

### Vector Search

**Best for:** Conceptual Japanese queries, synonym matching, cross-language search

```bash
# Conceptual search in Japanese
oboyu query --mode vector "機械学習の基本概念"
```

**Strengths:**
- Understands semantic relationships
- Handles synonyms and related terms
- Works well with modern Japanese terminology

### BM25 Search

**Best for:** Exact Japanese term matching, technical documentation

```bash
# Exact term search
oboyu query --mode bm25 "データベース設計"
```

**Strengths:**
- Precise keyword matching
- Good for specific technical terms
- Fast execution for large document sets

### Hybrid Search (Recommended)

**Best for:** Most Japanese search scenarios

```bash
# Default hybrid search (recommended)
oboyu query "Pythonでの非同期処理の実装方法"

# Custom weights favoring semantic understanding
oboyu query --vector-weight 0.8 --bm25-weight 0.2 "システム設計の原則"
```

**Benefits:**
- Combines semantic and keyword matching
- Robust across different query types
- Optimal balance for Japanese content

## Interactive Mode for Japanese

The interactive mode is particularly useful for Japanese queries:

```bash
# Start interactive session
oboyu query --interactive --rerank

> /mode hybrid
> /weights 0.8 0.2
> 機械学習のアルゴリズムについて
> /mode vector  
> プログラミングのベストプラクティス
```

**Japanese-specific benefits:**
- Test different search strategies for Japanese queries
- Iterate on complex Japanese terminology
- Compare results across different modes

## Performance Considerations

### Memory Usage

Japanese processing requires additional memory:

- **MeCab dictionary**: ~50MB
- **Character normalization tables**: ~10MB
- **Ruri model**: ~300MB (embedding) + ~400MB (reranker)

### Indexing Performance

Japanese text indexing characteristics:

```bash
# Typical performance (Japanese documents)
- Text extraction: 100-200 files/second
- Tokenization: 50-100 files/second (with MeCab)
- Embedding generation: 10-20 files/second
```

### Query Performance

Japanese query performance:

```bash
# Response times (typical)
- Vector search: 50-100ms
- BM25 search: 10-30ms  
- Hybrid search: 60-120ms
- With reranking: +50-200ms
```

## Best Practices

### Indexing Japanese Documents

1. **Enable encoding detection** for legacy documents
2. **Use appropriate file patterns** to include Japanese text files
3. **Configure proper chunk sizes** for Japanese text density

```bash
# Recommended indexing command for Japanese content
oboyu index ./japanese_docs \
  --include-patterns "*.txt,*.md,*.html" \
  --japanese-encodings "utf-8,shift-jis,euc-jp" \
  --chunk-size 512 \
  --chunk-overlap 128
```

### Querying Japanese Content

1. **Use hybrid search** as the default mode
2. **Enable reranking** for better accuracy
3. **Adjust weights** based on content type

```bash
# Recommended query command for Japanese content
oboyu query "検索クエリ" --mode hybrid --rerank --vector-weight 0.7 --bm25-weight 0.3
```

### Configuration for Japanese

```yaml
# Optimal configuration for Japanese content
crawler:
  japanese_encodings: ["utf-8", "shift-jis", "euc-jp"]
  include_patterns: ["*.txt", "*.md", "*.html", "*.py"]
  
indexer:
  chunk_size: 512  # Smaller chunks for Japanese density
  chunk_overlap: 128
  use_reranker: true
  reranker_model: "cl-nagoya/ruri-reranker-small"

query:
  default_mode: "hybrid"
  vector_weight: 0.7  # Favor semantic understanding
  bm25_weight: 0.3
  use_reranker: true
```

## Troubleshooting

### Common Issues

1. **MeCab installation problems**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install mecab mecab-ipadic-utf8
   
   # On macOS
   brew install mecab mecab-ipadic
   ```

2. **Encoding detection errors**
   - Verify file encoding with `file -i filename`
   - Add specific encodings to configuration
   - Use UTF-8 for new documents

3. **Poor search results for Japanese**
   - Enable reranking: `--rerank`
   - Try vector-focused weights: `--vector-weight 0.8 --bm25-weight 0.2`
   - Use interactive mode to test different strategies

4. **Performance issues with Japanese**
   - Reduce chunk size for dense Japanese text
   - Consider using smaller reranker model
   - Disable reranking for speed-critical applications

### Debug Mode

Enable debug logging for Japanese processing:

```python
import logging
logging.getLogger("oboyu.indexer.tokenizer").setLevel(logging.DEBUG)
logging.getLogger("oboyu.crawler.japanese").setLevel(logging.DEBUG)
```

## Examples

### Mixed Language Documents

Oboyu handles mixed Japanese-English content effectively:

```bash
# Query mixing Japanese and English
oboyu query "REST APIの設計パターン"
oboyu query "データベースのNormalization理論"
```

### Technical Documentation

For technical Japanese documentation:

```bash
# Technical terms with exact matching
oboyu query --mode hybrid --bm25-weight 0.5 "非同期処理とPromise"

# Conceptual technical queries
oboyu query --mode vector "マイクロサービスアーキテクチャの利点"
```

### Business Documents

For business Japanese content:

```bash
# Business process queries
oboyu query "業務プロセスの改善方法"

# Policy and procedure searches
oboyu query --rerank "コンプライアンス規定について"
```

## Future Enhancements

Planned improvements for Japanese support:

- [ ] Custom MeCab dictionaries for domain-specific terms
- [ ] Furigana (reading) extraction and search
- [ ] Historical Japanese text support (Classical Japanese)
- [ ] Enhanced compound word recognition
- [ ] Japanese-specific relevance tuning