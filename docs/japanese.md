# Japanese Language Support in Oboyu

Oboyu provides specialized support for Japanese text processing, focusing on accurate encoding detection and text normalization to ensure reliable search results.

## Encoding Detection

### Overview

Japanese text files often use various character encodings due to historical reasons. Oboyu automatically detects and handles multiple Japanese encodings to ensure all documents are properly indexed.

### Supported Encodings

Oboyu supports automatic detection and conversion of common Japanese encodings:

- **UTF-8**: Modern standard encoding (default)
- **Shift-JIS**: Windows Japanese encoding (CP932)
- **EUC-JP**: Unix/Linux Japanese encoding
- **ISO-2022-JP**: Email and older systems encoding

### How It Works

When processing files, Oboyu:

1. **Attempts UTF-8 first** as the modern standard
2. **Detects encoding** using byte patterns specific to Japanese encodings
3. **Converts to UTF-8** internally for consistent processing
4. **Handles mixed encodings** in different files within the same directory

### Automatic Detection

Oboyu automatically detects and handles Japanese encodings without requiring configuration. The system intelligently identifies the encoding of each file and converts it to UTF-8 for consistent processing.

## Text Normalization

### Unicode Normalization

All Japanese text undergoes Unicode NFKC (Normalization Form KC) normalization to ensure consistent matching:

```
Input:  "ｈｅｌｌｏ　ワールド"  (mixed width)
Output: "hello ワールド"      (normalized)
```

This normalization handles:
- **Full-width to half-width conversion** for ASCII characters
- **Character variant unification** (異体字の統一)
- **Combining character normalization**

### Character Width Normalization

Oboyu normalizes character widths for consistent search:

```
Full-width: ＡＢＣ１２３
Half-width: ABC123

Full-width: ｶﾀｶﾅ
Full-width: カタカナ
```

### Benefits

- **Consistent search results** regardless of input method
- **Improved matching** between different document sources
- **Reduced index size** through normalization

## Japanese Text Processing Pipeline

### Processing Steps

1. **File Reading**
   - Detect encoding using configured encoding list
   - Read file with detected encoding

2. **Text Extraction**
   - Convert to UTF-8 if needed
   - Extract text content

3. **Normalization**
   - Apply Unicode NFKC normalization
   - Normalize character widths
   - Clean up whitespace

4. **Tokenization** (for BM25 search)
   - Use MeCab for morphological analysis
   - Extract meaningful tokens

### Example Pipeline

```python
# Original file (Shift-JIS encoded)
"データベース　設計　パターン"

# After encoding detection and conversion
"データベース　設計　パターン" (UTF-8)

# After normalization
"データベース 設計 パターン" (normalized spaces)

# After tokenization
["データベース", "設計", "パターン"]
```

## Best Practices

### For Modern Projects

Oboyu automatically handles all encoding scenarios:

```bash
# Modern UTF-8 documents
oboyu index ./modern_docs

# Legacy systems with Shift-JIS or EUC-JP
oboyu index ./legacy_system

# Mixed encoding environments
oboyu index ./mixed_docs
```

The system will automatically detect and handle UTF-8, Shift-JIS, EUC-JP, CP932, and ISO-2022-JP encodings.

## Troubleshooting

### Encoding Detection Issues

**Symptoms:**
- Garbled characters (文字化け)
- Missing Japanese text
- Indexing errors

**Solutions:**
1. Check file encoding: `file -i filename.txt`
2. Ensure the file is in a supported encoding (UTF-8, Shift-JIS, EUC-JP, CP932, or ISO-2022-JP)
3. Convert unsupported encodings to UTF-8 before indexing

### Normalization Issues

**Symptoms:**
- Search misses documents with different character widths
- Inconsistent search results

**Solutions:**
1. Ensure all documents are re-indexed after configuration changes
2. Use consistent input methods for queries
3. Check normalization in debug mode

### Debug Mode

Enable debug logging for Japanese processing:

```bash
# Index with debug output
oboyu index ./docs --debug
```

## Performance Considerations

### Encoding Detection Overhead

- **Minimal impact**: ~1-5ms per file
- **Cached results**: Encoding detected once per file
- **Parallel processing**: Multiple files processed concurrently

### Normalization Performance

- **Fast processing**: <1ms per document
- **In-memory operation**: No disk I/O required
- **Single-pass**: Normalization done during initial processing

## Configuration

No encoding configuration is required. Oboyu automatically detects and handles all common Japanese encodings:

```yaml
crawler:
  respect_gitignore: true
  max_file_size: 10485760  # 10MB
  # Encoding detection is automatic - no configuration needed
```

## Integration with Search

The encoding detection and normalization ensure that:

- **Vector search** works with properly encoded text
- **BM25 search** matches normalized tokens correctly
- **Hybrid search** combines both effectively

All search modes benefit from consistent text processing, ensuring reliable results regardless of the original file encoding or character variations.