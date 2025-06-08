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
oboyu query --query "budget"

# Multiple keywords (OR logic)
oboyu query --query "budget finance accounting"

# All keywords required (AND logic)
oboyu query --query "+budget +2024 +approved"
```

### Phrase Search
```bash
# Exact phrase
oboyu query --query '"project deadline march 15"'

# Partial phrase with wildcards
oboyu query --query '"project deadline *"'
```

### Boolean Operators
```bash
# AND operator
oboyu query --query "python AND tutorial"

# OR operator  
oboyu query --query "python OR java OR javascript"

# NOT operator
oboyu query --query "python NOT beginner"

# Complex boolean
oboyu query --query "(python OR java) AND tutorial NOT video"
```

## Semantic Search Patterns

### Concept-Based Search
```bash
# Find documents about a concept
oboyu query --query "documents explaining customer churn" --mode vector

# Find similar documents
oboyu query --query "similar to our Q3 financial report" --mode vector

# Abstract concepts
oboyu query --query "strategies for improving team morale" --mode vector
```

### Question-Based Search
```bash
# Direct questions
oboyu query --query "what is our remote work policy?" --mode vector

# How-to queries
oboyu query --query "how to configure database connections" --mode vector

# Why queries
oboyu query --query "why did sales drop in December?" --mode vector
```

### Context-Aware Search
```bash
# Find related documents
oboyu query --query "documents related to the Johnson account" --mode vector

# Find follow-ups
oboyu query --query "follow-up actions from product launch meeting" --mode vector

# Find prerequisites
oboyu query --query "what should I read before the strategy meeting?" --mode vector
```

## Hybrid Search Patterns

### Balanced Search
```bash
# Combine precision and recall
oboyu query --query "machine learning python tutorial" --mode hybrid

# Technical + conceptual
oboyu query --query "REST API best practices security" --mode hybrid
```

### Weighted Search
```bash
# Emphasize keywords
oboyu query --query "error 404 nginx" --mode hybrid

# Emphasize semantics
oboyu query --query "improving code quality" --mode hybrid
```

## Pattern Search Examples

### Finding Patterns in Text
```bash
# Email patterns
oboyu query --query "[a-zA-Z]+@company\.com"

# Date patterns
oboyu query --query "\d{4}-\d{2}-\d{2}"

# Version numbers
oboyu query --query "v\d+\.\d+\.\d+"
```

### Code Patterns
```bash
# Function definitions
oboyu query --query "def \w+\(.*\):"

# Import statements
oboyu query --query "^import|^from .* import"

# TODO comments
oboyu query --query "//\s*TODO:|#\s*TODO:"
```

### Document Structure Patterns
```bash
# Markdown headers
oboyu query --query "^#{1,6}\s+.*Security"

# List items
oboyu query --query "^\s*[-*]\s+.*requirement"

# Numbered items
oboyu query --query "^\s*\d+\.\s+.*step"
```

## Advanced Search Techniques

### Proximity Search
```bash
# Words near each other
oboyu query --query "project NEAR/5 deadline"

# Same paragraph
oboyu query --query "budget SAME_PARA approval"

# Same sentence
oboyu query --query "meeting SAME_SENT cancelled"
```

### Fuzzy Search
```bash
# Spelling variations
oboyu query --query "organization~" # Also finds: organisation

# Typo tolerance
oboyu query --query "recieve~2" # Finds: receive

# Sound-alike (phonetic)
oboyu query --query "smith~phonetic" # Finds: Schmidt, Smythe
```

### Field-Specific Search
```bash
# Search in title/filename
oboyu query --query "title:report"

# Search in path
oboyu query --query "path:projects/alpha"

# Search by author (if metadata available)
oboyu query --query "author:john"

# Search by date
oboyu query --query "modified:2024-01-*"
```

## Search Patterns by Use Case

### Meeting Notes Patterns
```bash
# Action items
oboyu query --query "action:|assigned:|TODO:|@"

# Decisions made
oboyu query --query "decided:|agreed:|approved:" --mode vector

# Attendees
oboyu query --query "attendees:|present:|participants:"
```

### Project Documentation
```bash
# Requirements
oboyu query --query "must|shall|should|requirement"

# Technical specifications
oboyu query --query "spec:|specification:|interface:|API:"

# Diagrams and figures
oboyu query --query "figure|diagram|chart|graph" --mode vector
```

### Research Papers
```bash
# Citations
oboyu query --query "\[\d+\]|\(\w+,\s*\d{4}\)"

# Abstract sections
oboyu query --query "abstract:|summary:"

# Conclusions
oboyu query --query "conclusion:|in conclusion|we conclude" --mode vector
```

## Language-Specific Patterns

### Japanese Search Patterns
```bash
# Hiragana only
oboyu query --query "[ぁ-ん]+"

# Katakana only
oboyu query --query "[ァ-ヴ]+"

# Kanji compounds
oboyu query --query "[一-龥]{2,}"

# Mixed Japanese/English
oboyu query --query "プロジェクト.*deadline"
```

### Multi-language Patterns
```bash
# Documents in specific language
oboyu query --query "lang:ja 会議"

# Mixed language documents
oboyu query --query "meeting 会議" --mode hybrid

# Romanized Japanese
oboyu query --query "kaigi OR 会議"
```

## Performance Search Patterns

### Optimized Searches
```bash
# Limit scope early
oboyu query --query "error"

# Use specific index
oboyu query --query "configuration" --db-path ~/indexes/technical-docs.db

# Combine filters
oboyu query --query "critical"
```

### Batch Search Patterns
```bash
# Multiple related searches
for term in "error" "warning" "critical"; do
    oboyu query --query "$term" > "${term}-results.txt"
done

# Progressive refinement
oboyu query --query "project" > all-projects.txt
oboyu query --query "project alpha" > alpha-specific.txt
```

## Search Pattern Templates

### Daily Standup Template
```bash
# What I did yesterday
oboyu query --query "completed|done|finished"

# What I'm doing today  
oboyu query --query "TODO|planned|scheduled"

# Blockers
oboyu query --query "blocked|issue|problem" --mode vector
```

### Code Review Template
```bash
# Find recent changes
oboyu query --query "modified|updated|changed"

# Security concerns
oboyu query --query "password|secret|key|token"

# TODO items
oboyu query --query "TODO|FIXME|HACK|XXX"
```

### Research Template
```bash
# Find methodology
oboyu query --query "method|methodology|approach"

# Find results
oboyu query --query "results|findings|outcomes"

# Find limitations
oboyu query --query "limitation|caveat|assumption"
```

## Combining Patterns

### Complex Queries
```bash
# Financial + Date + Format
oboyu query --query "(revenue OR profit) AND 2024"

# Technical + Semantic + Scope
oboyu query --query "database optimization techniques" --mode hybrid

# Multi-pattern search
oboyu query --query '(TODO|FIXME) AND "high priority"'
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
oboyu query --query "your pattern" --top-k 5

# Explain query interpretation
oboyu query --query "complex query"

# Debug regex patterns
oboyu query --query "pattern"
```

## Next Steps

- Apply these patterns in [Real-world Scenarios](../real-world-scenarios/technical-docs.md)
- Optimize search with [Search Optimization](../configuration-optimization/search-optimization.md)
- Automate patterns with [CLI Workflows](../integration/cli-workflows.md)