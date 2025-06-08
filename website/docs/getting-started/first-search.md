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
oboyu query "project deadline"
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

2. meeting-notes-jan.txt (Score: 0.87)
   Path: ~/Documents/meetings/meeting-notes-jan.txt
   Modified: 2024-01-10
   
   ...discussed the upcoming project deadline and resource allocation. 
   The team agreed that the March deadline is achievable if we...

Found 15 results in 0.23 seconds
```

## Search Modes

Oboyu offers three search modes to match your needs:

### 1. Keyword Search (Default)
Fast, traditional keyword matching:
```bash
oboyu query "budget report 2024"
```

### 2. Semantic Search
Understands meaning and context:
```bash
oboyu query "documents about financial planning" --mode semantic
```

### 3. Hybrid Search
Combines keyword and semantic search for best results:
```bash
oboyu query "quarterly revenue analysis" --mode hybrid
```

## Refining Your Search

### Limit Number of Results
```bash
oboyu query "meeting notes" --limit 5
```

### Search Specific Index
```bash
oboyu query "architecture design" --db-path ~/indexes/work.db
```

### Filter by File Type
```bash
oboyu query "configuration" --file-type yaml
```

### Filter by Date
```bash
# Documents modified in last 7 days
oboyu query "status update" --days 7

# Documents from specific date range
oboyu query "project plan" --from 2024-01-01 --to 2024-01-31
```

## Search Examples by Use Case

### Finding Meeting Notes
```bash
# Recent meeting about specific topic
oboyu query "budget discussion" --days 30 --file-type md

# All meetings with specific person
oboyu query "meeting with Sarah" --mode semantic
```

### Searching Technical Documentation
```bash
# Find API documentation
oboyu query "REST API authentication" --db-path ~/indexes/dev-docs.db

# Find configuration examples
oboyu query "database connection config" --file-type yaml
```

### Research and Academic Papers
```bash
# Find papers on specific topic
oboyu query "machine learning optimization" --db-path ~/indexes/research.db --file-type pdf

# Find recent research
oboyu query "neural networks 2024" --mode hybrid
```

## Advanced Search Techniques

### Phrase Search
Use quotes for exact phrases:
```bash
oboyu query '"project deadline march 15"'
```

### Combining Terms
Use natural language:
```bash
oboyu query "emails about contract renewal from legal department"
```

### Excluding Terms
Use minus sign to exclude:
```bash
oboyu query "python tutorial -beginner"
```

### Wildcard Search
Use asterisk for partial matches:
```bash
oboyu query "report_2024_*.pdf"
```

## Interactive Search Mode

For exploratory searching, use interactive mode:

```bash
oboyu query --interactive
```

This opens a search session where you can:
- Refine queries based on results
- Navigate through results easily
- Open documents directly
- Save search results

Example session:
```
Oboyu Interactive Search
Type 'help' for commands, 'quit' to exit

> search: project roadmap
Found 23 results

> show 1
[Displaying document content...]

> refine: Q2 milestones
Found 8 results (refined from previous search)

> save results roadmap-search.txt
Results saved to roadmap-search.txt
```

## Search Result Actions

### Open Document
Open a result directly:
```bash
oboyu query "design spec" --open 1
```

### Export Results
Save search results:
```bash
oboyu query "quarterly reports" --export results.txt
```

### Copy File Path
```bash
oboyu query "config file" --copy-path 1
```

## Japanese Language Search

Oboyu excels at Japanese text search:

```bash
# Search in Japanese
oboyu query "会議議事録"

# Mixed language search
oboyu query "プロジェクト deadline"

# Semantic search understands context
oboyu query "来週の予定について" --mode semantic
```

## Search Performance Tips

### 1. Use Specific Keywords
Instead of: `"document"`
Try: `"Q4 financial report"`

### 2. Leverage Semantic Search
For conceptual searches:
```bash
oboyu query "documents explaining our pricing strategy" --mode semantic
```

### 3. Combine Filters
Narrow down results:
```bash
oboyu query "api documentation" --file-type md --days 90 --limit 10
```

## Understanding Relevance Scores

Oboyu assigns relevance scores (0.0 to 1.0):
- **0.90-1.00**: Excellent match
- **0.70-0.89**: Good match
- **0.50-0.69**: Fair match
- **Below 0.50**: Weak match

## Saving Frequent Searches

Create aliases for common searches:

```bash
# Save a search
oboyu query save "weekly meeting notes" --db-path ~/indexes/meetings.db

# Run saved search
oboyu query run meetings
```

## Search History

View your search history:
```bash
oboyu query history

# Re-run a previous search
oboyu query history --run 3
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
   oboyu query "project"  # Too broad
   oboyu query "project alpha milestone 2"  # Better
   ```

2. **Use Natural Language**
   ```bash
   oboyu query "emails about the new product launch"
   ```

3. **Combine Search Modes**
   Start with hybrid mode, then refine with specific modes

4. **Regular Index Updates**
   Keep your search results current:
   ```bash
   oboyu index update && oboyu query "latest reports"
   ```

## Next Steps

Now that you can search effectively:
- Explore [Basic Workflows](../usage-examples/basic-workflow.md) for daily usage patterns
- Learn about [Search Patterns](../usage-examples/search-patterns.md) for advanced techniques
- Configure [Search Optimization](../configuration-optimization/search-optimization.md) for better results