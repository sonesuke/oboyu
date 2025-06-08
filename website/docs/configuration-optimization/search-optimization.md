---
id: search-optimization
title: Search Accuracy Improvement Tips
sidebar_position: 3
---

# Search Accuracy Improvement Tips

Maximize the relevance and quality of your search results with these optimization techniques. Learn how to fine-tune search behavior, improve result ranking, and handle edge cases effectively.

## Understanding Search Quality

### What Affects Search Accuracy

1. **Index Quality**: How well documents are chunked and embedded
2. **Query Formulation**: How queries are expressed
3. **Search Mode**: Vector vs BM25 vs Hybrid
4. **Reranking**: Post-processing for relevance
5. **Filtering**: Narrowing the search space

### Measuring Search Quality

```bash
# Test search quality
oboyu query test --queries sample-queries.txt

# Analyze search patterns
oboyu query analyze --days 30

# Get relevance feedback
oboyu query "search term" --feedback
```

## Optimizing Search Modes

### Keyword Search (BM25)

Best for exact matches and technical terms:

```bash
# Precise technical searches
oboyu query "ERROR_CODE_404" --mode bm25

# Case-sensitive search
oboyu query "className" --mode bm25 --case-sensitive

# Exact phrase matching
oboyu query '"exact error message"' --mode bm25
```

**Optimization Tips:**
- Use quotes for exact phrases
- Include technical identifiers
- Leverage boolean operators

### Semantic Search (Vector)

Best for conceptual and natural language queries:

```bash
# Conceptual queries
oboyu query "documents about improving team productivity" --mode vector

# Question-based search
oboyu query "how to configure database connections?" --mode vector

# Finding similar content
oboyu query "similar to our onboarding guide" --mode vector
```

**Optimization Tips:**
- Use natural language
- Include context and intent
- Ask questions directly

### Hybrid Search

Best balance of precision and recall:

```bash
# Default for most searches
oboyu query "python async programming" --mode hybrid

# Adjust weights for your needs
oboyu query "bug fix authentication" \
  --mode hybrid \
  --keyword-weight 0.3 \
  --vector-weight 0.7
```

**Optimization Tips:**
- Start with hybrid mode
- Adjust weights based on results
- Use for mixed technical/conceptual queries

## Query Optimization Techniques

### Query Expansion

Improve recall by expanding queries:

```bash
# Manual expansion
oboyu query "auth OR authentication OR login"

# Synonym expansion
oboyu query "bug error issue problem" --expand-synonyms

# Stem expansion
oboyu query "configure" --expand-stems  # Also finds: configuring, configuration
```

### Query Refinement

Improve precision by refining queries:

```bash
# Add context
oboyu query "index" --context "database performance"

# Exclude noise
oboyu query "python -snake -animal"

# Specify domain
oboyu query "transformer" --domain "machine learning"
```

### Query Templates

Create reusable query patterns:

```bash
# Save query template
oboyu query save \
  "error {{CODE}} in {{FILE}}" \
  --db-path ~/indexes/error-search.db

# Use template
oboyu query template error-search \
  --CODE "500" \
  --FILE "api.py"
```

## Relevance Tuning

### Boosting Strategies

#### Field Boosting
```bash
# Boost title matches
oboyu query "configuration" --boost "title:2.0"

# Boost recent documents
oboyu query "release notes" --boost "date:1.5"

# Multiple boosts
oboyu query "api documentation" \
  --boost "title:2.0" \
  --boost "path:**/api/**:1.5"
```

#### Term Boosting
```bash
# Boost important terms
oboyu query "python^2.0 tutorial"

# Negative boosting
oboyu query "java -deprecated^0.5"
```

### Reranking Configuration

```yaml
# Enhanced reranking settings
query:
  use_reranker: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  reranker_top_k: 50  # Rerank top 50 results
```

```bash
# Query with custom reranking
oboyu query "machine learning" \
  --rerank \
  --rerank-top-k 100 \
  --rerank-threshold 0.7
```

## Filter Optimization

### Metadata Filters

```bash
# Filter by file type
oboyu query "configuration" --file-type md,yaml

# Filter by date
oboyu query "meeting notes" \
  --from "2024-01-01" \
  --to "2024-01-31"

# Filter by path
oboyu query "readme" --path "**/docs/**"

# Combine filters
oboyu query "api" \
  --file-type md \
  --path "**/api/**" \
  --days 30
```

### Content Filters

```bash
# Filter by language
oboyu query "エラー" --language ja

# Filter by length
oboyu query "summary" --min-words 100 --max-words 500

# Filter by metadata
oboyu query "report" --has-metadata "author"
```

## Search Result Quality

### Result Ranking

```bash
# Sort by relevance (default)
oboyu query "python tutorial" --sort relevance

# Sort by date
oboyu query "changelog" --sort date-desc

# Sort by path
oboyu query "readme" --sort path

# Custom scoring
oboyu query "important" \
  --score-function "0.7*relevance + 0.3*recency"
```

### Result Grouping

```bash
# Group by directory
oboyu query "test" --group-by directory

# Group by file type
oboyu query "documentation" --group-by extension

# Group by date
oboyu query "meeting" --group-by date
```

## Advanced Optimization

### Query Analysis

```bash
# Analyze query performance
oboyu query "search term" --analyze

# Output:
# Query Analysis:
# - Parse time: 2ms
# - Search time: 45ms
# - Rerank time: 15ms
# - Total matches: 1,234
# - Returned: 10
```

### A/B Testing

```bash
# Test different strategies
oboyu query ab-test "search term" \
  --strategies "bm25,vector,hybrid" \
  --metrics "precision,recall"

# Compare reranker impact
oboyu query compare "complex query" \
  --with-reranker \
  --without-reranker
```

### Search Personalization

```bash
# Learn from feedback
oboyu query "python async" --learn

# Use learned preferences
oboyu query "programming tutorial" --personalized

# Reset learning
oboyu query reset-personalization
```

## Domain-Specific Optimization

### Code Search

```yaml
# Optimized for code
query:
  default_mode: "hybrid"
  code_search: true
  extract_symbols: true
  boost_definitions: 2.0
```

```bash
# Search for function definitions
oboyu query "def process_data" --code-search

# Find implementations
oboyu query "implements UserInterface" --code-search

# Search by signature
oboyu query "function(string, number): boolean" --signature-search
```

### Documentation Search

```yaml
# Optimized for docs
query:
  default_mode: "hybrid"
  boost_headers: 1.5
  boost_examples: 1.3
  snippet_context: 500
```

```bash
# Find API documentation
oboyu query "REST endpoint" --doc-search

# Search in specific sections
oboyu query "installation" --section "getting-started"
```

### Academic Search

```yaml
# Optimized for papers
query:
  default_mode: "vector"
  extract_citations: true
  boost_abstract: 1.5
  boost_conclusions: 1.3
```

```bash
# Search with citations
oboyu query "neural networks" --include-citations

# Find by methodology
oboyu query "quantitative analysis" --section "methodology"
```

## Performance vs Quality Trade-offs

### Fast but Good Enough

```yaml
# Speed-optimized
query:
  default_mode: "bm25"
  use_reranker: false
  top_k: 10
```

### Slow but Accurate

```yaml
# Quality-optimized
query:
  default_mode: "hybrid"
  use_reranker: true
  reranker_top_k: 100
  top_k: 10
```

### Balanced Approach

```yaml
# Balanced
query:
  default_mode: "hybrid"
  use_reranker: true
  reranker_top_k: 50
  top_k: 10
  cache_results: true
```

## Troubleshooting Poor Results

### Diagnostic Steps

1. **Check Index Quality**
```bash
oboyu index analyze --db-path ~/indexes/personal.db
```

2. **Test Query Parsing**
```bash
oboyu query "your search" --explain
```

3. **Compare Search Modes**
```bash
oboyu query "problematic search" --compare-modes
```

4. **Analyze Result Distribution**
```bash
oboyu query "search term" --show-distribution
```

### Common Issues and Fixes

**Too Many Irrelevant Results**
- Use more specific terms
- Add filters
- Enable reranking
- Switch to BM25 mode

**Missing Expected Results**
- Check if files are indexed
- Use semantic mode
- Broaden search terms
- Check filters aren't too restrictive

**Poor Ranking**
- Enable reranker
- Adjust boost factors
- Use hybrid mode
- Provide feedback for learning

## Search Optimization Checklist

- [ ] Choose appropriate search mode
- [ ] Use natural language for semantic search
- [ ] Apply relevant filters
- [ ] Enable reranking for quality
- [ ] Test with representative queries
- [ ] Monitor search analytics
- [ ] Gather user feedback
- [ ] Iterate based on results

## Next Steps

- Implement [Performance Tuning](performance-tuning.md) for faster searches
- Explore [CLI Workflows](../integration/cli-workflows.md) for search automation
- Review [Troubleshooting](../reference-troubleshooting/troubleshooting.md) for specific issues