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
# (Note: test functionality not available in current implementation)
oboyu query --query "test queries"

# Analyze search patterns
# (Note: analyze functionality not available in current implementation)

# Get relevance feedback
# (Note: feedback functionality not available in current implementation)
oboyu query --query "search term"
```

## Optimizing Search Modes

### Keyword Search (BM25)

Best for exact matches and technical terms:

```bash
# Precise technical searches
oboyu query --query "ERROR_CODE_404" --mode bm25

# Case-sensitive search
# (Note: case-sensitive option not available in current implementation)
oboyu query --query "className" --mode bm25

# Exact phrase matching
oboyu query --query '"exact error message"' --mode bm25
```

**Optimization Tips:**
- Use quotes for exact phrases
- Include technical identifiers
- Leverage boolean operators

### Semantic Search (Vector)

Best for conceptual and natural language queries:

```bash
# Conceptual queries
oboyu query --query "documents about improving team productivity" --mode vector

# Question-based search
oboyu query --query "how to configure database connections?" --mode vector

# Finding similar content
oboyu query --query "similar to our onboarding guide" --mode vector
```

**Optimization Tips:**
- Use natural language
- Include context and intent
- Ask questions directly

### Hybrid Search

Best balance of precision and recall:

```bash
# Default for most searches
oboyu query --query "python async programming" --mode hybrid

# Adjust weights for your needs
# (Note: weight adjustment not available in current implementation)
oboyu query --query "bug fix authentication" --mode hybrid
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
oboyu query --query "auth OR authentication OR login"

# Synonym expansion
# (Note: expand-synonyms option not available in current implementation)
oboyu query --query "bug error issue problem"

# Stem expansion
# (Note: expand-stems option not available in current implementation)
oboyu query --query "configure"  # Also finds: configuring, configuration
```

### Query Refinement

Improve precision by refining queries:

```bash
# Add context
# (Note: context option not available in current implementation)
oboyu query --query "index database performance"

# Exclude noise
oboyu query --query "python -snake -animal"

# Specify domain
# (Note: domain option not available in current implementation)
oboyu query --query "transformer machine learning"
```

### Query Templates

Create reusable query patterns:

```bash
# Save query template
# (Note: save/template functionality not available in current implementation)
# Use individual queries instead

# Use template
# (Note: template functionality not available in current implementation)
oboyu query --query "error 500 in api.py"
```

## Relevance Tuning

### Boosting Strategies

#### Field Boosting
```bash
# Boost title matches
# (Note: boost functionality not available in current implementation)
oboyu query --query "configuration"

# Boost recent documents
# (Note: boost functionality not available in current implementation)
oboyu query --query "release notes"

# Multiple boosts
# (Note: boost functionality not available in current implementation)
oboyu query --query "api documentation"
```

#### Term Boosting
```bash
# Boost important terms
# (Note: term boosting not available in current implementation)
oboyu query --query "python tutorial"

# Negative boosting
# (Note: negative boosting not available in current implementation)
oboyu query --query "java -deprecated"
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
# (Note: custom reranking options not available in current implementation)
oboyu query --query "machine learning"
```

## Filter Optimization

### Metadata Filters

```bash
# Filter by file type
# (Note: file-type filtering not available in current implementation)
oboyu query --query "configuration"

# Filter by date
# (Note: date filtering not available in current implementation)
oboyu query --query "meeting notes"

# Filter by path
# (Note: path filtering not available in current implementation)
oboyu query --query "readme"

# Combine filters
# (Note: filtering options not available in current implementation)
oboyu query --query "api"
```

### Content Filters

```bash
# Filter by language
# (Note: language filtering not available in current implementation)
oboyu query --query "エラー"

# Filter by length
# (Note: word count filtering not available in current implementation)
oboyu query --query "summary"

# Filter by metadata
# (Note: metadata filtering not available in current implementation)
oboyu query --query "report"
```

## Search Result Quality

### Result Ranking

```bash
# Sort by relevance (default)
oboyu query --query "python tutorial"

# Sort by date
# (Note: sorting options not available in current implementation)
oboyu query --query "changelog"

# Sort by path
# (Note: sorting options not available in current implementation)
oboyu query --query "readme"

# Custom scoring
# (Note: custom scoring not available in current implementation)
oboyu query --query "important"
```

### Result Grouping

```bash
# Group by directory
# (Note: grouping options not available in current implementation)
oboyu query --query "test"

# Group by file type
# (Note: grouping options not available in current implementation)
oboyu query --query "documentation"

# Group by date
# (Note: grouping options not available in current implementation)
oboyu query --query "meeting"
```

## Advanced Optimization

### Query Analysis

```bash
# Analyze query performance
# (Note: analyze option not available in current implementation)
oboyu query --query "search term"

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
# (Note: ab-test functionality not available in current implementation)
oboyu query --query "search term" --mode bm25
oboyu query --query "search term" --mode vector
oboyu query --query "search term" --mode hybrid

# Compare reranker impact
# (Note: compare functionality not available in current implementation)
oboyu query --query "complex query"
```

### Search Personalization

```bash
# Learn from feedback
# (Note: learning functionality not available in current implementation)
oboyu query --query "python async"

# Use learned preferences
# (Note: personalization not available in current implementation)
oboyu query --query "programming tutorial"

# Reset learning
# (Note: personalization reset not available in current implementation)
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
# (Note: code-search option not available in current implementation)
oboyu query --query "def process_data"

# Find implementations
# (Note: code-search option not available in current implementation)
oboyu query --query "implements UserInterface"

# Search by signature
# (Note: signature-search option not available in current implementation)
oboyu query --query "function(string, number): boolean"
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
# (Note: doc-search option not available in current implementation)
oboyu query --query "REST endpoint"

# Search in specific sections
# (Note: section option not available in current implementation)
oboyu query --query "installation getting-started"
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
# (Note: include-citations option not available in current implementation)
oboyu query --query "neural networks"

# Find by methodology
# (Note: section option not available in current implementation)
oboyu query --query "quantitative analysis methodology"
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
# (Note: explain option not available in current implementation)
oboyu query --query "your search"
```

3. **Compare Search Modes**
```bash
# (Note: compare-modes option not available in current implementation)
oboyu query --query "problematic search" --mode bm25
oboyu query --query "problematic search" --mode vector
oboyu query --query "problematic search" --mode hybrid
```

4. **Analyze Result Distribution**
```bash
# (Note: show-distribution option not available in current implementation)
oboyu query --query "search term"
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