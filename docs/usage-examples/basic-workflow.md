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
oboyu query --query "status OR update OR report"

# Check meeting notes from the past week
oboyu query --query "meeting"
```

### Create a Morning Alias

Save time with a custom command:

```bash
# Create the alias
echo 'alias morning="oboyu query --query \"status OR update OR meeting\""' >> ~/.bashrc

# Use it daily
morning
```

## Common Daily Tasks

### 1. Finding Today's Meeting Notes

Before a follow-up meeting:
```bash
# Find previous meeting notes
oboyu query --query "team meeting with product"

# Find action items from last meeting
oboyu query --query "action items TODO assigned" --mode vector
```

### 2. Locating Project Documents

When you need specific project files:
```bash
# Find all documents for a project
oboyu query --query "Project Alpha" --top-k 20

# Find the latest version
oboyu query --query "Project Alpha specification"
```

### 3. Searching Email Exports

If you export emails to text/markdown:
```bash
# Find emails from specific sender
oboyu query --query "from: john.doe@company.com"

# Find emails about specific topic
oboyu query --query "contract renewal discussion"
```

## Weekly Workflows

### Monday: Week Planning
```bash
# Find last week's accomplishments
oboyu query --query "completed OR done OR finished"

# Find this week's priorities
oboyu query --query "priority OR urgent OR deadline this week" --mode vector
```

### Friday: Week Review
```bash
# Generate week summary
oboyu query --query "progress update status"

# Find unfinished tasks
oboyu query --query "TODO OR pending OR in progress"
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
oboyu query --query "final project update" --db-path ~/indexes/messy-docs.db
```

## Quick Access Patterns

### Recent Files Workflow
```bash
# What did I work on yesterday?
oboyu query --query "*" --top-k 10

# Files modified this morning
oboyu query --query "*"
```

### Project Context Switching
```bash
# Save common search patterns in shell aliases
alias alpha-search="oboyu query --query --db-path ~/indexes/alpha-docs.db"
alias beta-search="oboyu query --query --db-path ~/indexes/beta-docs.db"

# Quick switch between projects
alpha-search "search term"  # When working on Alpha
beta-search "search term"   # When switching to Beta
```

## Collaborative Workflows

### Shared Document Search
```bash
# Find documents mentioning team members
oboyu query --query "reviewed by Sarah OR assigned to Sarah"

# Find collaborative documents
oboyu query --query "shared OR collaborative OR team"
```

### Meeting Preparation
```bash
# Before a meeting, find all related documents
oboyu query --query "Q4 planning"

# Find previous decisions
oboyu query --query "decided OR agreed OR approved" --mode vector
```

## Time-Saving Shortcuts

### 1. Quick Search Alias
```bash
# Add to your shell configuration
alias q="oboyu query --query"

# Usage
q "budget report"
q "meeting notes"
```

### 2. Project-Specific Commands
```bash
# Create project shortcuts
alias work="oboyu query --query --db-path ~/indexes/work.db"
alias personal="oboyu query --query --db-path ~/indexes/personal.db"

# Usage
work "performance review"
personal "tax documents"
```

### 3. Common Searches Function
```bash
# Add to .bashrc/.zshrc
today() {
    oboyu query --query "$1"
}

thisweek() {
    oboyu query --query "$1"
}

# Usage
today "meetings"
thisweek "reports"
```

## Integration with Other Tools

### Open in Editor
```bash
# Search and edit
oboyu query --query "config file"

# Search and view
oboyu query --query "report"
```

### Pipeline with Other Commands
```bash
# Find and count
oboyu query --query "error log" | wc -l

# Find and grep
oboyu query --query "configuration" | xargs grep "database"

# Find and backup
oboyu query --query "important" | xargs cp -t ~/backup/
```

## Mobile and Remote Workflows

### SSH Workflow
```bash
# Search remote documents via SSH
ssh server "oboyu query --query 'project status'"

# Sync search results
oboyu query --query "reports" | ssh server "cat > results.txt"
```

### Cloud Storage Integration
```bash
# Index cloud-synced folders
oboyu index ~/Dropbox/Documents --db-path ~/indexes/dropbox.db
oboyu index ~/Google\ Drive/Work --db-path ~/indexes/gdrive.db

# Search individual databases
oboyu query --query "presentation" --db-path ~/indexes/dropbox.db
oboyu query --query "presentation" --db-path ~/indexes/gdrive.db
```

## Productivity Tips

### 1. Index Regularly
Set up a cron job:
```bash
# Add to crontab
0 */4 * * * oboyu index ~/Documents --update
```

### 2. Use Watch Mode
Monitor for new documents:
```bash
# Watch for new files and index automatically
oboyu index ~/Documents --watch
```

### 3. Search History Analysis
```bash
# See what you search for most
oboyu query --query "*" # (Note: history functionality not available)

# Optimize common searches
# (Note: history functionality not available)
```

## Emergency Workflows

### Can't Find a Document
```bash
# Broad semantic search
oboyu query --query "document about [topic]" --mode vector --top-k 50

# Search by content in PDF-like files (text content only)
oboyu query --query "partial" | grep -i "pdf"

# Search by date range when you last remember seeing it
oboyu query --query "*"
```

### Recovering Lost Work
```bash
# Find most recently modified files
oboyu query --query "*" --top-k 20

# Find documents with specific content
oboyu query --query "exact phrase from document"
```

## Daily Checklist

A productive day with Oboyu:

- [ ] Morning: Check updates with `oboyu query --query "update"`
- [ ] Before meetings: Find relevant docs with project search
- [ ] During work: Use quick searches for instant access
- [ ] End of day: Update index with `oboyu index --update`
- [ ] Weekly: Review search patterns and optimize

## Next Steps

- Learn [Document Type Handling](document-types.md) for specific file formats
- Explore [Search Patterns](search-patterns.md) for advanced techniques
- Set up [Automation](../integration/automation.md) for hands-free workflows