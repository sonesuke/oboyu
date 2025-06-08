---
id: search-patterns
title: Search Pattern Collection
sidebar_position: 3
---

# Search Pattern Collection

Master the art of searching with these proven patterns and techniques. From basic keyword searches to advanced semantic queries, this guide covers all the search patterns you'll need.

## Keyword Search Patterns

### Basic Keywords
```bash
# Single keyword
oboyu query "budget"

# Multiple keywords (OR logic)
oboyu query "budget finance accounting"

# All keywords required (AND logic)
oboyu query "+budget +2024 +approved"
```

### Phrase Search
```bash
# Exact phrase
oboyu query '"project deadline march 15"'

# Partial phrase with wildcards
oboyu query '"project deadline *"'
```

### Boolean Operators
```bash
# AND operator
oboyu query "python AND tutorial"

# OR operator  
oboyu query "python OR java OR javascript"

# NOT operator
oboyu query "python NOT beginner"

# Complex boolean
oboyu query "(python OR java) AND tutorial NOT video"
```

## Semantic Search Patterns

### Concept-Based Search
```bash
# Find documents about a concept
oboyu query "documents explaining customer churn" --mode semantic

# Find similar documents
oboyu query "similar to our Q3 financial report" --mode semantic

# Abstract concepts
oboyu query "strategies for improving team morale" --mode semantic
```

### Question-Based Search
```bash
# Direct questions
oboyu query "what is our remote work policy?" --mode semantic

# How-to queries
oboyu query "how to configure database connections" --mode semantic

# Why queries
oboyu query "why did sales drop in December?" --mode semantic
```

### Context-Aware Search
```bash
# Find related documents
oboyu query "documents related to the Johnson account" --mode semantic

# Find follow-ups
oboyu query "follow-up actions from product launch meeting" --mode semantic

# Find prerequisites
oboyu query "what should I read before the strategy meeting?" --mode semantic
```

## Hybrid Search Patterns

### Balanced Search
```bash
# Combine precision and recall
oboyu query "machine learning python tutorial" --mode hybrid

# Technical + conceptual
oboyu query "REST API best practices security" --mode hybrid
```

### Weighted Search
```bash
# Emphasize keywords
oboyu query "error 404 nginx" --mode hybrid --keyword-weight 0.7

# Emphasize semantics
oboyu query "improving code quality" --mode hybrid --semantic-weight 0.7
```

## Pattern Search Examples

### Finding Patterns in Text
```bash
# Email patterns
oboyu query "[a-zA-Z]+@company\.com" --regex

# Date patterns
oboyu query "\d{4}-\d{2}-\d{2}" --regex

# Version numbers
oboyu query "v\d+\.\d+\.\d+" --regex
```

### Code Patterns
```bash
# Function definitions
oboyu query "def \w+\(.*\):" --regex --file-type py

# Import statements
oboyu query "^import|^from .* import" --regex --file-type py

# TODO comments
oboyu query "//\s*TODO:|#\s*TODO:" --regex
```

### Document Structure Patterns
```bash
# Markdown headers
oboyu query "^#{1,6}\s+.*Security" --regex --file-type md

# List items
oboyu query "^\s*[-*]\s+.*requirement" --regex

# Numbered items
oboyu query "^\s*\d+\.\s+.*step" --regex
```

## Advanced Search Techniques

### Proximity Search
```bash
# Words near each other
oboyu query "project NEAR/5 deadline"

# Same paragraph
oboyu query "budget SAME_PARA approval"

# Same sentence
oboyu query "meeting SAME_SENT cancelled"
```

### Fuzzy Search
```bash
# Spelling variations
oboyu query "organization~" # Also finds: organisation

# Typo tolerance
oboyu query "recieve~2" # Finds: receive

# Sound-alike (phonetic)
oboyu query "smith~phonetic" # Finds: Schmidt, Smythe
```

### Field-Specific Search
```bash
# Search in title/filename
oboyu query "title:report"

# Search in path
oboyu query "path:projects/alpha"

# Search by author (if metadata available)
oboyu query "author:john"

# Search by date
oboyu query "modified:2024-01-*"
```

## Search Patterns by Use Case

### Meeting Notes Patterns
```bash
# Action items
oboyu query "action:|assigned:|TODO:|@" --file-type md

# Decisions made
oboyu query "decided:|agreed:|approved:" --mode semantic

# Attendees
oboyu query "attendees:|present:|participants:"
```

### Project Documentation
```bash
# Requirements
oboyu query "must|shall|should|requirement" --file-type md,docx

# Technical specifications
oboyu query "spec:|specification:|interface:|API:"

# Diagrams and figures
oboyu query "figure|diagram|chart|graph" --mode semantic
```

### Research Papers
```bash
# Citations
oboyu query "\[\d+\]|\(\w+,\s*\d{4}\)" --regex --file-type pdf

# Abstract sections
oboyu query "abstract:|summary:" --file-type pdf

# Conclusions
oboyu query "conclusion:|in conclusion|we conclude" --mode semantic
```

## Language-Specific Patterns

### Japanese Search Patterns
```bash
# Hiragana only
oboyu query "[ぁ-ん]+" --regex

# Katakana only
oboyu query "[ァ-ヴ]+" --regex

# Kanji compounds
oboyu query "[一-龥]{2,}" --regex

# Mixed Japanese/English
oboyu query "プロジェクト.*deadline"
```

### Multi-language Patterns
```bash
# Documents in specific language
oboyu query "lang:ja 会議"

# Mixed language documents
oboyu query "meeting 会議" --mode hybrid

# Romanized Japanese
oboyu query "kaigi OR 会議"
```

## Performance Search Patterns

### Optimized Searches
```bash
# Limit scope early
oboyu query "error" --file-type log --days 1

# Use specific index
oboyu query "configuration" --db-path ~/indexes/technical-docs.db

# Combine filters
oboyu query "critical" --file-type log --path "**/errors/**" --days 7
```

### Batch Search Patterns
```bash
# Multiple related searches
for term in "error" "warning" "critical"; do
    oboyu query "$term" --file-type log --export "${term}-results.txt"
done

# Progressive refinement
oboyu query "project" --export all-projects.txt
oboyu query "project alpha" --export alpha-specific.txt
```

## Search Pattern Templates

### Daily Standup Template
```bash
# What I did yesterday
oboyu query "completed|done|finished" --days 1 --author me

# What I'm doing today  
oboyu query "TODO|planned|scheduled" --days 0

# Blockers
oboyu query "blocked|issue|problem" --mode semantic --days 7
```

### Code Review Template
```bash
# Find recent changes
oboyu query "modified|updated|changed" --file-type py,js --days 7

# Security concerns
oboyu query "password|secret|key|token" --file-type py,js

# TODO items
oboyu query "TODO|FIXME|HACK|XXX" --file-type py,js
```

### Research Template
```bash
# Find methodology
oboyu query "method|methodology|approach" --file-type pdf

# Find results
oboyu query "results|findings|outcomes" --file-type pdf

# Find limitations
oboyu query "limitation|caveat|assumption" --file-type pdf
```

## Combining Patterns

### Complex Queries
```bash
# Financial + Date + Format
oboyu query "(revenue OR profit) AND 2024" --file-type xlsx,pdf

# Technical + Semantic + Scope
oboyu query "database optimization techniques" --mode hybrid --path "**/docs/**"

# Multi-pattern search
oboyu query '(TODO|FIXME) AND "high priority"' --file-type md --days 30
```

## Search Pattern Best Practices

1. **Start Simple**: Begin with basic keywords, then add complexity
2. **Use the Right Mode**: Keyword for exact terms, semantic for concepts
3. **Filter Early**: Use file-type and date filters to narrow scope
4. **Test Patterns**: Use --limit 5 to test before full search
5. **Save Complex Patterns**: Create aliases for frequently used patterns

## Debugging Search Patterns

### Pattern Testing
```bash
# Test with small result set
oboyu query "your pattern" --limit 5 --verbose

# Explain query interpretation
oboyu query "complex query" --explain

# Debug regex patterns
oboyu query "pattern" --regex --debug
```

## Next Steps

- Apply these patterns in [Real-world Scenarios](../real-world-scenarios/technical-docs.md)
- Optimize search with [Search Optimization](../configuration-optimization/search-optimization.md)
- Automate patterns with [CLI Workflows](../integration/cli-workflows.md)