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
oboyu query "status OR update OR report" --days 1

# Check meeting notes from the past week
oboyu query "meeting" --days 7 --file-type md
```

### Create a Morning Alias

Save time with a custom command:

```bash
# Create the alias
echo 'alias morning="oboyu query \"status OR update OR meeting\" --days 1"' >> ~/.bashrc

# Use it daily
morning
```

## Common Daily Tasks

### 1. Finding Today's Meeting Notes

Before a follow-up meeting:
```bash
# Find previous meeting notes
oboyu query "team meeting with product" --days 30

# Find action items from last meeting
oboyu query "action items TODO assigned" --mode semantic
```

### 2. Locating Project Documents

When you need specific project files:
```bash
# Find all documents for a project
oboyu query "Project Alpha" --limit 20

# Find the latest version
oboyu query "Project Alpha specification" --days 7
```

### 3. Searching Email Exports

If you export emails to text/markdown:
```bash
# Find emails from specific sender
oboyu query "from: john.doe@company.com"

# Find emails about specific topic
oboyu query "contract renewal discussion" --file-type eml
```

## Weekly Workflows

### Monday: Week Planning
```bash
# Find last week's accomplishments
oboyu query "completed OR done OR finished" --from last-monday --to last-friday

# Find this week's priorities
oboyu query "priority OR urgent OR deadline this week" --mode semantic
```

### Friday: Week Review
```bash
# Generate week summary
oboyu query "progress update status" --days 5 --export weekly-summary.txt

# Find unfinished tasks
oboyu query "TODO OR pending OR in progress" --days 5
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
oboyu index ~/Documents --name messy-docs

# Find what you need instantly
oboyu query "final project update" --index messy-docs
```

## Quick Access Patterns

### Recent Files Workflow
```bash
# What did I work on yesterday?
oboyu query "*" --days 1 --limit 10

# Files modified this morning
oboyu query "*" --hours 4
```

### Project Context Switching
```bash
# Save project-specific searches
oboyu query save "Project Alpha specs requirements" --name alpha-docs
oboyu query save "Project Beta api documentation" --name beta-docs

# Quick switch between projects
oboyu query run alpha-docs  # When working on Alpha
oboyu query run beta-docs   # When switching to Beta
```

## Collaborative Workflows

### Shared Document Search
```bash
# Find documents mentioning team members
oboyu query "reviewed by Sarah" OR "assigned to Sarah"

# Find collaborative documents
oboyu query "shared OR collaborative OR team" --file-type gdoc
```

### Meeting Preparation
```bash
# Before a meeting, find all related documents
oboyu query "Q4 planning" --export meeting-prep.txt

# Find previous decisions
oboyu query "decided OR agreed OR approved" --mode semantic
```

## Time-Saving Shortcuts

### 1. Quick Search Alias
```bash
# Add to your shell configuration
alias q="oboyu query"

# Usage
q "budget report"
q "meeting notes" --days 7
```

### 2. Project-Specific Commands
```bash
# Create project shortcuts
alias work="oboyu query --index work"
alias personal="oboyu query --index personal"

# Usage
work "performance review"
personal "tax documents"
```

### 3. Common Searches Function
```bash
# Add to .bashrc/.zshrc
today() {
    oboyu query "$1" --days 1
}

thisweek() {
    oboyu query "$1" --days 7
}

# Usage
today "meetings"
thisweek "reports"
```

## Integration with Other Tools

### Open in Editor
```bash
# Search and edit
oboyu query "config file" --open-with "code"

# Search and view
oboyu query "report" --open-with "less"
```

### Pipeline with Other Commands
```bash
# Find and count
oboyu query "error log" --format paths | wc -l

# Find and grep
oboyu query "configuration" --format paths | xargs grep "database"

# Find and backup
oboyu query "important" --days 30 --format paths | xargs cp -t ~/backup/
```

## Mobile and Remote Workflows

### SSH Workflow
```bash
# Search remote documents via SSH
ssh server "oboyu query 'project status'"

# Sync search results
oboyu query "reports" --export - | ssh server "cat > results.txt"
```

### Cloud Storage Integration
```bash
# Index cloud-synced folders
oboyu index ~/Dropbox/Documents --name dropbox
oboyu index ~/Google\ Drive/Work --name gdrive

# Search across cloud storage
oboyu query "presentation" --index dropbox,gdrive
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
oboyu query history --stats

# Optimize common searches
oboyu query history --frequent | head -5
```

## Emergency Workflows

### Can't Find a Document
```bash
# Broad semantic search
oboyu query "document about [topic]" --mode semantic --limit 50

# Search by partial filename
oboyu query "*.pdf" --name-only | grep -i "partial"

# Search by date range when you last remember seeing it
oboyu query "*" --from 2024-01-01 --to 2024-01-15
```

### Recovering Lost Work
```bash
# Find most recently modified files
oboyu query "*" --days 1 --sort modified --limit 20

# Find documents with specific content
oboyu query "exact phrase from document" --mode exact
```

## Daily Checklist

A productive day with Oboyu:

- [ ] Morning: Check updates with `oboyu query "update" --days 1`
- [ ] Before meetings: Find relevant docs with project search
- [ ] During work: Use quick searches for instant access
- [ ] End of day: Update index with `oboyu index --update`
- [ ] Weekly: Review search patterns and optimize

## Next Steps

- Learn [Document Type Handling](document-types.md) for specific file formats
- Explore [Search Patterns](search-patterns.md) for advanced techniques
- Set up [Automation](../integration/automation.md) for hands-free workflows