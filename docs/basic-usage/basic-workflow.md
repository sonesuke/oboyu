---
id: basic-workflow
title: Essential Daily Workflow
sidebar_position: 1
---

# Essential Daily Workflow

This guide shows you how to integrate Oboyu into your daily routine for maximum productivity. Learn the most common patterns and shortcuts that will make document search second nature.

## Morning Routine: Quick Status Check

Start your day by catching up on recent updates:

```bash
# Find all documents modified yesterday
oboyu search "status OR update OR report"

# Check meeting notes from the past week
oboyu search "meeting"
```

### Create a Morning Alias

Save time with a custom command:

```bash
# Create the alias
echo 'alias morning="oboyu search \"status OR update OR meeting\""' >> ~/.bashrc

# Use it daily
morning
```

## Common Daily Tasks

### 1. Finding Today's Meeting Notes

Before a follow-up meeting:
```bash
# Find previous meeting notes
oboyu search "team meeting with product"

# Find action items from last meeting
oboyu search "action items TODO assigned" --mode vector
```

### 2. Locating Project Documents

When you need specific project files:
```bash
# Find all documents for a project
oboyu search "Project Alpha" --top-k 20

# Find the latest version
oboyu search "Project Alpha specification"
```

### 3. Searching Email Exports

If you export emails to text/markdown:
```bash
# Find emails from specific sender
oboyu search "from: john.doe@company.com"

# Find emails about specific topic
oboyu search "contract renewal discussion"
```

## Weekly Workflows

### Monday: Week Planning
```bash
# Find last week's accomplishments
oboyu search "completed OR done OR finished"

# Find this week's priorities
oboyu search "priority OR urgent OR deadline this week" --mode vector
```

### Friday: Week Review
```bash
# Generate week summary
oboyu search "progress update status"

# Find unfinished tasks
oboyu search "TODO OR pending OR in progress"
```

## Document Organization Workflow

### Before: Scattered Documents
```
~/Documents/
├── meeting_notes_jan15.txt
├── proj_update_final_v2_FINAL.doc
├── notes-2024-01-16.md
└── TODO-urgent.txt
```

### After: Searchable with Oboyu
No need to reorganize! Just index and search:

```bash
# Index messy folder
oboyu index ~/Documents --db-path ~/indexes/messy-docs.db

# Find what you need instantly
oboyu search "final project update" --db-path ~/indexes/messy-docs.db
```

## Data Enrichment Workflow

### Enhancing CSV Data with Knowledge Base
When you have CSV data that needs enrichment with information from your documents:

```bash
# Start with basic CSV data
cat companies.csv
# company_name,industry
# Apple Inc.,Technology
# Microsoft Corporation,Technology

# Create enrichment schema
cat > enrichment-schema.json << 'EOF'
{
  "input_schema": {
    "columns": {
      "company_name": {"type": "string", "required": true},
      "industry": {"type": "string", "required": false}
    },
    "primary_keys": ["company_name"]
  },
  "enrichment_schema": {
    "columns": {
      "description": {
        "type": "string",
        "description": "Company description and business overview",
        "source_strategy": "search_content",
        "query_template": "{company_name} business overview description",
        "extraction_method": "summarize"
      },
      "founded_year": {
        "type": "integer",
        "description": "Year company was founded",
        "source_strategy": "search_content",
        "query_template": "{company_name} founded established year",
        "extraction_method": "pattern_match",
        "extraction_pattern": "\\b(19|20)\\d{2}\\b"
      }
    }
  }
}
EOF

# Enrich data with knowledge base information
oboyu enrich companies.csv enrichment-schema.json --output enriched-companies.csv

# View enriched results
cat enriched-companies.csv
# company_name,industry,description,founded_year
# Apple Inc.,Technology,"Technology company specializing in consumer electronics and software",1976
# Microsoft Corporation,Technology,"Software and cloud computing company",1975
```

### Daily Enrichment Tasks
Common enrichment scenarios:

```bash
# Enrich customer data with company information
oboyu enrich customers.csv customer-enrichment.json

# Enrich financial data with market information
oboyu enrich stocks.csv financial-enrichment.json --confidence 0.8

# Enrich research data with publication details
oboyu enrich research-papers.csv publication-enrichment.json --batch-size 5
```

For detailed enrichment workflows, see the [CSV Enrichment Use Case](../use-cases/csv-enrichment.md) guide.

## Quick Access Patterns

### Recent Files Workflow
```bash
# What did I work on yesterday?
oboyu search "*" --top-k 10

# Files modified this morning
oboyu search "*"
```

### Project Context Switching
```bash
# Save common search patterns in shell aliases
alias alpha-search="oboyu search --db-path ~/indexes/alpha-docs.db"
alias beta-search="oboyu search --db-path ~/indexes/beta-docs.db"

# Quick switch between projects
alpha-search "search term"  # When working on Alpha
beta-search "search term"   # When switching to Beta
```

## Collaborative Workflows

### Shared Document Search
```bash
# Find documents mentioning team members
oboyu search "reviewed by Sarah OR assigned to Sarah"

# Find collaborative documents
oboyu search "shared OR collaborative OR team"
```

### Meeting Preparation
```bash
# Before a meeting, find all related documents
oboyu search "Q4 planning"

# Find previous decisions
oboyu search "decided OR agreed OR approved" --mode vector
```

## Time-Saving Shortcuts

### 1. Quick Search Alias
```bash
# Add to your shell configuration
alias q="oboyu search"

# Usage
q "budget report"
q "meeting notes"
```

### 2. Project-Specific Commands
```bash
# Create project shortcuts
alias work="oboyu search --db-path ~/indexes/work.db"
alias personal="oboyu search --db-path ~/indexes/personal.db"

# Usage
work "performance review"
personal "tax documents"
```

### 3. Common Searches Function
```bash
# Add to .bashrc/.zshrc
today() {
    oboyu search "$1"
}

thisweek() {
    oboyu search "$1"
}

# Usage
today "meetings"
thisweek "reports"
```

## Integration with Other Tools

### Open in Editor
```bash
# Search and edit
oboyu search "config file"

# Search and view
oboyu search "report"
```

### Pipeline with Other Commands
```bash
# Find and count
oboyu search "error log" | wc -l

# Find and grep
oboyu search "configuration" | xargs grep "database"

# Find and backup
oboyu search "important" | xargs cp -t ~/backup/
```

## Mobile and Remote Workflows

### SSH Workflow
```bash
# Search remote documents via SSH
ssh server "oboyu search 'project status'"

# Sync search results
oboyu search "reports" | ssh server "cat > results.txt"
```

### Cloud Storage Integration
```bash
# Index cloud-synced folders
oboyu index ~/Dropbox/Documents --db-path ~/indexes/dropbox.db
oboyu index ~/Google\ Drive/Work --db-path ~/indexes/gdrive.db

# Search individual databases
oboyu search "presentation" --db-path ~/indexes/dropbox.db
oboyu search "presentation" --db-path ~/indexes/gdrive.db
```

## Productivity Tips

### 1. Index Regularly
Set up a cron job:
```bash
# Add to crontab - use --cleanup-deleted for incremental updates
0 */4 * * * oboyu index ~/Documents --cleanup-deleted
```

### 2. Regular Updates
Update your index periodically:
```bash
# Incremental update with cleanup of deleted files
oboyu index ~/Documents --cleanup-deleted
```

### 3. Search History Analysis
```bash
# See what you search for most
oboyu search "*" # (Note: history functionality not available)

# Optimize common searches
# (Note: history functionality not available)
```

## Emergency Workflows

### Can't Find a Document
```bash
# Broad semantic search
oboyu search "document about [topic]" --mode vector --top-k 50

# Search by content in PDF-like files (text content only)
oboyu search "partial" | grep -i "pdf"

# Search by date range when you last remember seeing it
oboyu search "*"
```

### Recovering Lost Work
```bash
# Find most recently modified files
oboyu search "*" --top-k 20

# Find documents with specific content
oboyu search "exact phrase from document"
```

## Daily Checklist

A productive day with Oboyu:

- [ ] Morning: Check updates with `oboyu search "update"`
- [ ] Before meetings: Find relevant docs with project search
- [ ] During work: Use quick searches for instant access
- [ ] End of day: Update index with `oboyu index ~/Documents --cleanup-deleted`
- [ ] Weekly: Review search patterns and optimize

## Next Steps

- Learn [Document Type Handling](document-types.md) for specific file formats
- Explore [Search Patterns](search-patterns.md) for advanced techniques
- Set up [Automation](../integration-automation/automation.md) for hands-free workflows