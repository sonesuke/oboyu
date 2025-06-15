---
id: first-search
title: Executing Your First Search
sidebar_position: 3
---

# Executing Your First Search

Now that you've created an index, let's explore how to search your documents effectively with Oboyu.

## Basic Search

The simplest search is just a few keywords:

```bash
oboyu search "project deadline"
```

Oboyu will return the most relevant documents containing information about project deadlines.

## Understanding Search Results

A typical search result looks like this:

```
Searching for: "project deadline"

1. project-plan-2024.md (Score: 0.92)
   Path: ~/Documents/projects/project-plan-2024.md
   Modified: 2024-01-15
   
   ...The project deadline has been set for March 15th, with key milestones 
   scheduled for February 1st and February 20th. All team members should...

2. github-issue-123.md (Score: 0.87)
   Path: ~/Documents/github-issues/github-issue-123.md
   Modified: 2024-01-10
   
   ...discussed the upcoming project deadline and resource allocation. 
   The team agreed that the March deadline is achievable if we...

Found 15 results in 0.23 seconds
```

## Search Modes

Oboyu offers three search modes to match your needs:

### 1. Hybrid Search (Default)
Combines keyword and semantic search for best results:
```bash
oboyu search "quarterly revenue analysis"
```

### 2. Vector Search
Understands meaning and context using semantic embeddings:
```bash
oboyu search "documents about financial planning" --mode vector
```

### 3. BM25 Search
Fast, traditional keyword matching:
```bash
oboyu search "budget report 2024" --mode bm25
```

## Refining Your Search

### Limit Number of Results
```bash
oboyu search "meeting notes" --top-k 5
```

### Enable Reranking
Improve search quality with neural reranking:
```bash
oboyu search "architecture design" --rerank
```

### Get Detailed Explanations
See why documents matched your query:
```bash
oboyu search "configuration" --explain
```

### Output as JSON
Get structured results for automation:
```bash
oboyu search "status update" --format json
```

## Search Examples by Use Case

### Finding Meeting Notes
```bash
# Meeting about specific topic
oboyu search "budget discussion"

# All meetings with specific person using semantic search
oboyu search "meeting with Sarah" --mode vector
```

### Searching Technical Documentation
```bash
# Find API documentation
oboyu search "REST API authentication"

# Find configuration examples with reranking
oboyu search "database connection config" --rerank
```

### Research and Academic Papers
```bash
# Find papers on specific topic with limited results
oboyu search "machine learning optimization" --top-k 10

# Find recent research using hybrid search
oboyu search "neural networks 2024" --mode hybrid
```

## Advanced Search Techniques

### Phrase Search
Use quotes for exact phrases:
```bash
oboyu search '"project deadline march 15"'
```

### Combining Terms
Use natural language:
```bash
oboyu search "emails about contract renewal from legal department"
```

## Interactive Search Mode

For exploratory searching, use interactive mode:

```bash
oboyu search --interactive
```

This opens a search session where you can:
- Execute multiple searches in sequence
- Navigate through results easily
- Refine queries based on results

Example session:
```
Oboyu Interactive Search
Type 'help' for commands, 'quit' to exit

> project roadmap
Found 23 results

[Results displayed...]

> Q2 milestones
Found 8 results

[Results displayed...]

> quit
```


## Japanese Language Search

Oboyu excels at Japanese text search:

```bash
# Search in Japanese
oboyu search "会議議事録"

# Mixed language search
oboyu search "プロジェクト deadline"

# Vector search understands context
oboyu search "来週の予定について" --mode vector
```

## Search Performance Tips

### 1. Use Specific Keywords
Instead of: `"document"`
Try: `"Q4 financial report"`

### 2. Leverage Vector Search
For conceptual searches:
```bash
oboyu search "documents explaining our pricing strategy" --mode vector
```

### 3. Use Available Options
Improve your search with available options:
```bash
oboyu search "api documentation" --rerank --top-k 10
```

## Understanding Relevance Scores

Oboyu assigns relevance scores (0.0 to 1.0):
- **0.90-1.00**: Excellent match
- **0.70-0.89**: Good match
- **0.50-0.69**: Fair match
- **Below 0.50**: Weak match

## Advanced Options

### Custom Database Path
Search a specific database:
```bash
oboyu search "documents" --db-path /path/to/custom.db
```

### Fine-tune Hybrid Search
Adjust the RRF (Reciprocal Rank Fusion) parameter:
```bash
oboyu search "documents" --rrf-k 30
```

## Troubleshooting Search Issues

### No Results Found
- Check spelling and try variations
- Use semantic mode for conceptual searches
- Verify the index is up-to-date

### Too Many Results
- Add more specific keywords
- Use filters (date, file type)
- Try phrase search with quotes

### Irrelevant Results
- Switch search modes
- Use more descriptive terms
- Filter by recently modified files

## Search Best Practices

1. **Start Broad, Then Narrow**
   ```bash
   oboyu search "project"  # Too broad
   oboyu search "project alpha milestone 2"  # Better
   ```

2. **Use Natural Language**
   ```bash
   oboyu search "emails about the new product launch"
   ```

3. **Combine Search Modes**
   Start with hybrid mode, then refine with specific modes

4. **Regular Index Updates**
   Keep your search results current:
   ```bash
   oboyu manage diff && oboyu search "latest reports"
   ```

## Next Steps

Now that you can search effectively:
- Explore [Basic Workflows](../basic-usage/basic-workflow.md) for daily usage patterns
- Learn about [Search Patterns](../basic-usage/search-patterns.md) for advanced techniques
- Configure [Search Optimization](../reference/configuration.md) for better results