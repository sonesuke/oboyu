---
id: personal-notes
title: Personal Note Search Examples
sidebar_position: 4
---

# Personal Note Search Examples

Transform your personal knowledge management with Oboyu. Learn how to effectively search through journals, ideas, learning notes, and personal documentation.

## Scenario: Digital Knowledge Worker

Managing personal notes, ideas, learning resources, and life documentation across various topics and formats.

### Setting Up Personal Knowledge Base

```bash
# Index all personal notes
oboyu index ~/Notes --db-path ~/indexes/personal.db \
    --include "*.md" \
    --include "*.txt" \
    --include "*.org"

# Add specific categories
oboyu index ~/Notes/Journal --db-path ~/indexes/journal.db --update
oboyu index ~/Notes/Learning --db-path ~/indexes/learning.db --update
oboyu index ~/Notes/Ideas --db-path ~/indexes/ideas.db --update
```

### Daily Note-Taking Workflow

#### Morning Pages
```bash
# Find today's journal entry
oboyu query "date: $(date +%Y-%m-%d)" --db-path ~/indexes/journal.db

# Review yesterday's thoughts
oboyu query "reflection thoughts" --days 1 --db-path ~/indexes/journal.db

# Find recurring themes
oboyu query "grateful thankful appreciation" --days 30 --mode semantic
```

#### Idea Capture
```bash
# Find related ideas
oboyu query "app idea mobile" --db-path ~/indexes/ideas.db --mode semantic

# Find unfinished ideas
oboyu query "TODO elaborate expand on this" --db-path ~/indexes/ideas.db

# Connect ideas across time
oboyu query "similar to [current idea]" --mode semantic --days 365
```

## Scenario: Lifelong Learner

Organizing study notes, course materials, book summaries, and skill development tracking.

### Learning Management
```bash
# Find notes on specific topic
oboyu query "python decorators" --db-path ~/indexes/learning.db

# Track learning progress
oboyu query "learned: understood: mastered:" --days 30

# Find practice exercises
oboyu query "exercise: practice: try this:" --db-path ~/indexes/learning.db
```

### Book Notes and Summaries
```bash
# Find book summaries
oboyu query "book: title: author:" --db-path ~/indexes/personal.db

# Search quotes and highlights
oboyu query "quote: highlight: important:" --context 200

# Find book recommendations
oboyu query "recommended by must read" --mode semantic
```

## Scenario: Personal Project Manager

Tracking personal projects, goals, habits, and life administration.

### Goal Tracking
```bash
# Find current goals
oboyu query "goal: objective: target:" --db-path ~/indexes/personal.db

# Track progress
oboyu query "progress: milestone: achieved:" --days 30

# Review quarterly goals
oboyu query "Q1 2024 goals" --mode semantic
```

### Habit Tracking
```bash
# Find habit logs
oboyu query "habit: tracker: streak:" --days 7

# Analyze patterns
oboyu query "missed skipped failed" --days 30 --db-path ~/indexes/journal.db

# Find motivation notes
oboyu query "why important motivation" --mode semantic
```

## Advanced Personal Search Patterns

### Memory and Recall
```bash
# Find memories by date
oboyu query "remember when that time" --mode semantic

# Search by people
oboyu query "with Mom Dad family" --db-path ~/indexes/journal.db

# Find by location
oboyu query "at Paris Tokyo vacation" --mode semantic
```

### Reflection and Insights
```bash
# Find insights and learnings
oboyu query "realized: learned: insight:" --db-path ~/indexes/journal.db

# Track personal growth
oboyu query "improved better than before" --mode semantic --days 365

# Find patterns in thoughts
oboyu query "always when every time pattern" --mode semantic
```

## Real-World Example: Annual Review

Creating a comprehensive year-in-review using personal notes:

```bash
# 1. Gather all monthly reviews
oboyu query "monthly review reflection" --from 2023-01-01 --to 2023-12-31

# 2. Find major accomplishments
oboyu query "accomplished achieved completed proud" --year 2023 --mode semantic

# 3. Extract learnings
oboyu query "learned lesson mistake growth" --year 2023 --mode semantic

# 4. Find memorable moments
oboyu query "memorable amazing wonderful special" --year 2023 --db-path ~/indexes/journal.db

# 5. Track goal completion
oboyu query "goal achieved completed ✓" --year 2023

# 6. Identify themes
oboyu query --word-frequency --year 2023 --db-path ~/indexes/journal.db --top 20

# 7. Export for review document
oboyu query "2023" --db-path ~/indexes/personal.db --export year-review-2023.txt
```

## Personal Productivity Patterns

### GTD (Getting Things Done) Workflow
```bash
# Inbox processing
oboyu query "inbox: capture: process:" --status pending

# Next actions
oboyu query "@next @action context:" --db-path ~/indexes/personal.db

# Waiting for
oboyu query "@waiting @blocked waiting for" 

# Weekly review
oboyu query "added: modified:" --days 7 --export weekly-review.txt
```

### Zettelkasten Method
```bash
# Find permanent notes
oboyu query "id: [[" --db-path ~/indexes/zettelkasten.db

# Find connections
oboyu query "[[.*]]" --regex --file current-note.md

# Find orphan notes
oboyu query "*" --no-links --db-path ~/indexes/zettelkasten.db
```

## Creative Writing and Ideas

### Story and Content Ideas
```bash
# Find story seeds
oboyu query "what if imagine suppose" --db-path ~/indexes/ideas.db

# Character development
oboyu query "character: personality: trait:" --mode semantic

# Plot elements
oboyu query "conflict twist revelation" --mode semantic
```

### Blog Post Management
```bash
# Find draft posts
oboyu query "draft: status: unpublished" --db-path ~/indexes/blog.db

# Research for posts
oboyu query "research: source: reference:" --related-to current-post.md

# Find evergreen content
oboyu query "timeless evergreen always relevant" --mode semantic
```

## Personal Database Searches

### Contact and Relationship Notes
```bash
# Find notes about people
oboyu query "met talked with discussed" --db-path ~/indexes/journal.db

# Birthday and important dates
oboyu query "birthday anniversary important date" 

# Gift ideas
oboyu query "likes enjoys interested in gift idea" --mode semantic
```

### Health and Fitness Logs
```bash
# Find workout logs
oboyu query "workout exercise gym" --days 30

# Track symptoms or health notes
oboyu query "feeling symptom doctor health" --db-path ~/indexes/journal.db

# Diet and nutrition
oboyu query "ate meal calories nutrition" --days 7
```

## Japanese Personal Notes

For bilingual note-takers:

```bash
# Japanese diary entries
oboyu query "日記 今日" --db-path ~/indexes/journal.db

# Mixed language notes
oboyu query "meeting 会議 notes メモ" --mode hybrid

# Study notes
oboyu query "勉強 学習 覚える" --db-path ~/indexes/learning.db

# Personal reminders
oboyu query "忘れない 重要 リマインダー"
```

## Note Organization Best Practices

### Effective Note Structure
```markdown
---
date: 2024-01-15
tags: [learning, python, programming]
type: study-note
---

# Python Decorators Study

## Key Concepts
- Decorators are functions that modify other functions
- Use @syntax for clean code

## Examples
```python
@timer
def slow_function():
    pass
```

## Questions
- How do parameterized decorators work?

## TODO
- [ ] Practice writing custom decorators
- [ ] Read Real Python decorator guide
```

### Searchable Patterns
- Use consistent date formats: `YYYY-MM-DD`
- Tag liberally: `#idea #todo #learning`
- Use markers: `TODO:`, `IDEA:`, `REMEMBER:`
- Include context: people, places, projects
- Cross-reference: `See also: [[other-note]]`

## Daily Routines with Oboyu

### Morning Routine
```bash
# Review yesterday and plan today
morning_notes() {
    echo "=== Yesterday's Highlights ==="
    oboyu query "accomplished done completed" --days 1
    
    echo "=== Today's Priorities ==="
    oboyu query "todo today priority" --days 0
    
    echo "=== Upcoming Deadlines ==="
    oboyu query "deadline due date" --days 7
}
```

### Evening Review
```bash
# Reflect on the day
evening_review() {
    echo "=== Today's Entries ==="
    oboyu query "*" --days 0 --db-path ~/indexes/journal.db
    
    echo "=== Gratitude ==="
    oboyu query "grateful thankful appreciate" --days 0
    
    echo "=== Tomorrow's Focus ==="
    oboyu query "tomorrow next day" --days 0
}
```

## Integration Tips

### Quick Capture
```bash
# Add to shell config for quick notes
note() {
    echo "$(date +%Y-%m-%d): $*" >> ~/Notes/quick-capture.md
    oboyu index ~/Notes/quick-capture.md --update
}

# Usage
note "Great idea for app: voice-controlled task manager"
```

### Template System
```bash
# Create note from template
new_note() {
    template=$1
    title=$2
    cp ~/Notes/templates/$template.md ~/Notes/$(date +%Y%m%d)-$title.md
    oboyu index ~/Notes --update
}

# Usage
new_note "meeting" "team-standup"
new_note "book" "atomic-habits"
```

## Personal Knowledge Metrics

### Note-Taking Analytics
```bash
# Most active topics
oboyu query "*" --db-path ~/indexes/personal.db --word-frequency --top 20

# Note creation frequency
oboyu query "*" --db-path ~/indexes/personal.db --timeline --group-by day

# Most connected notes
oboyu query "[[*]]" --regex --stats links
```

## Next Steps

- Set up [CLI Workflows](../integration/cli-workflows.md) for personal productivity
- Configure [Search Optimization](../configuration-optimization/search-optimization.md) for your note structure
- Explore [Automation](../integration/automation.md) for note management