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
oboyu search "budget"

# Multiple keywords (OR logic)
oboyu search "budget finance accounting"

# All keywords required (AND logic)
oboyu search "+budget +2024 +approved"
```

### Phrase Search
```bash
# Exact phrase
oboyu search '"project deadline march 15"'

# Partial phrase with wildcards
oboyu search '"project deadline *"'
```

### Boolean Operators
```bash
# AND operator
oboyu search "python AND tutorial"

# OR operator  
oboyu search "python OR java OR javascript"

# NOT operator
oboyu search "python NOT beginner"

# Complex boolean
oboyu search "(python OR java) AND tutorial NOT video"
```

## Semantic Search Patterns

### Concept-Based Search
```bash
# Find documents about a concept
oboyu search "documents explaining customer churn" --mode vector

# Find similar documents
oboyu search "similar to our Q3 financial report" --mode vector

# Abstract concepts
oboyu search "strategies for improving team morale" --mode vector
```

### Question-Based Search
```bash
# Direct questions
oboyu search "what is our remote work policy?" --mode vector

# How-to queries
oboyu search "how to configure database connections" --mode vector

# Why queries
oboyu search "why did sales drop in December?" --mode vector
```

### Context-Aware Search
```bash
# Find related documents
oboyu search "documents related to the Johnson account" --mode vector

# Find follow-ups
oboyu search "follow-up actions from product launch meeting" --mode vector

# Find prerequisites
oboyu search "what should I read before the strategy meeting?" --mode vector
```

## Hybrid Search Patterns

### Balanced Search
```bash
# Combine precision and recall
oboyu search "machine learning python tutorial" --mode hybrid

# Technical + conceptual
oboyu search "REST API best practices security" --mode hybrid
```

### Weighted Search
```bash
# Emphasize keywords
oboyu search "error 404 nginx" --mode hybrid

# Emphasize semantics
oboyu search "improving code quality" --mode hybrid
```

## Pattern Search Examples

### Finding Patterns in Text
```bash
# Email patterns
oboyu search "[a-zA-Z]+@company\.com"

# Date patterns
oboyu search "\d{4}-\d{2}-\d{2}"

# Version numbers
oboyu search "v\d+\.\d+\.\d+"
```

### Code Patterns
```bash
# Function definitions
oboyu search "def \w+\(.*\):"

# Import statements
oboyu search "^import|^from .* import"

# TODO comments
oboyu search "//\s*TODO:|#\s*TODO:"
```

### Document Structure Patterns
```bash
# Markdown headers
oboyu search "^#{1,6}\s+.*Security"

# List items
oboyu search "^\s*[-*]\s+.*requirement"

# Numbered items
oboyu search "^\s*\d+\.\s+.*step"
```

## Advanced Search Techniques

### Proximity Search
```bash
# Words near each other
oboyu search "project NEAR/5 deadline"

# Same paragraph
oboyu search "budget SAME_PARA approval"

# Same sentence
oboyu search "meeting SAME_SENT cancelled"
```

### Fuzzy Search
```bash
# Spelling variations
oboyu search "organization~" # Also finds: organisation

# Typo tolerance
oboyu search "recieve~2" # Finds: receive

# Sound-alike (phonetic)
oboyu search "smith~phonetic" # Finds: Schmidt, Smythe
```

### Field-Specific Search
```bash
# Search in title/filename
oboyu search "title:report"

# Search in path
oboyu search "path:projects/alpha"

# Search by author (if metadata available)
oboyu search "author:john"

# Search by date
oboyu search "modified:2024-01-*"
```

## Search Patterns by Use Case

### Meeting Notes Patterns
```bash
# Action items
oboyu search "action:|assigned:|TODO:|@"

# Decisions made
oboyu search "decided:|agreed:|approved:" --mode vector

# Attendees
oboyu search "attendees:|present:|participants:"
```

### Project Documentation
```bash
# Requirements
oboyu search "must|shall|should|requirement"

# Technical specifications
oboyu search "spec:|specification:|interface:|API:"

# Diagrams and figures
oboyu search "figure|diagram|chart|graph" --mode vector
```

### Research Papers
```bash
# Citations
oboyu search "\[\d+\]|\(\w+,\s*\d{4}\)"

# Abstract sections
oboyu search "abstract:|summary:"

# Conclusions
oboyu search "conclusion:|in conclusion|we conclude" --mode vector
```

## Language-Specific Patterns

### Japanese Search Patterns
```bash
# Hiragana only
oboyu search "[ぁ-ん]+"

# Katakana only
oboyu search "[ァ-ヴ]+"

# Kanji compounds
oboyu search "[一-龥]{2,}"

# Mixed Japanese/English
oboyu search "プロジェクト.*deadline"
```

### Multi-language Patterns
```bash
# Documents in specific language
oboyu search "lang:ja 会議"

# Mixed language documents
oboyu search "meeting 会議" --mode hybrid

# Romanized Japanese
oboyu search "kaigi OR 会議"
```

## Performance Search Patterns

### Optimized Searches
```bash
# Limit scope early
oboyu search "error"

# Use specific index
oboyu search "configuration" --db-path ~/indexes/technical-docs.db

# Combine filters
oboyu search "critical"
```

### Batch Search Patterns
```bash
# Multiple related searches
for term in "error" "warning" "critical"; do
    oboyu search "$term" > "${term}-results.txt"
done

# Progressive refinement
oboyu search "project" > all-projects.txt
oboyu search "project alpha" > alpha-specific.txt
```

## Search Pattern Templates

### Daily Standup Template
```bash
# What I did yesterday
oboyu search "completed|done|finished"

# What I'm doing today  
oboyu search "TODO|planned|scheduled"

# Blockers
oboyu search "blocked|issue|problem" --mode vector
```

### Code Review Template
```bash
# Find recent changes
oboyu search "modified|updated|changed"

# Security concerns
oboyu search "password|secret|key|token"

# TODO items
oboyu search "TODO|FIXME|HACK|XXX"
```

### Research Template
```bash
# Find methodology
oboyu search "method|methodology|approach"

# Find results
oboyu search "results|findings|outcomes"

# Find limitations
oboyu search "limitation|caveat|assumption"
```

## Combining Patterns

### Complex Queries
```bash
# Financial + Date + Format
oboyu search "(revenue OR profit) AND 2024"

# Technical + Semantic + Scope
oboyu search "database optimization techniques" --mode hybrid

# Multi-pattern search
oboyu search '(TODO|FIXME) AND "high priority"'
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
oboyu search "your pattern" --top-k 5

# Explain query interpretation
oboyu search "complex query"

# Debug regex patterns
oboyu search "pattern"
```

## Next Steps

- Apply these patterns in [Real-world Scenarios](../use-cases/technical-docs.md)
- Optimize search with [Search Optimization](../reference/configuration.md)
- Automate patterns with [CLI Workflows](../integration-automation/cli-workflows.md)