---
id: meeting-notes
title: Meeting Minutes Search Examples
sidebar_position: 2
---

# Meeting Minutes Search Examples

Master the art of searching through meeting notes, agendas, and action items. This guide provides real-world examples for effectively managing and retrieving information from meetings.

## Scenario: Corporate Meeting Management

You attend multiple meetings daily and need to track decisions, action items, and follow-ups across various projects and teams.

### Setting Up Meeting Notes Index

```bash
# Index all meeting notes
oboyu index ~/Documents/meetings --db-path ~/indexes/meetings.db \
    --include "*.md" \
    --include "*.txt" \
    --include "*.docx"

# Add shared meeting folders
oboyu index ~/Shared/team-meetings --db-path ~/indexes/meetings.db --update
```

### Pre-Meeting Preparation

#### Finding Previous Meeting Notes
```bash
# Last meeting with specific person/team
oboyu query "meeting with Sarah marketing" --days 30

# All meetings about a project
oboyu query "Project Phoenix meeting" --db-path ~/indexes/meetings.db

# Meetings with specific agenda items
oboyu query "agenda budget discussion" --mode semantic
```

#### Gathering Context
```bash
# Find action items from last meeting
oboyu query "action: assigned: TODO:" --file-type md --days 7

# Check previous decisions
oboyu query "decided: agreed: approved:" --mode semantic

# Find related documents mentioned
oboyu query "refer to document attachment" --mode semantic
```

## Scenario: Project Manager's Workflow

Managing multiple projects with various stakeholders and tracking all commitments.

### Daily Meeting Review
```bash
# Morning review of yesterday's meetings
oboyu query "meeting" --days 1 --format summary

# Find all action items assigned to you
oboyu query "@yourname action OR assigned" --days 7

# Check upcoming deliverables mentioned
oboyu query "deadline due date deliverable" --days 7 --mode semantic
```

### Weekly Status Compilation
```bash
# Gather all project updates
oboyu query "status update progress" --days 7 --file-type md

# Find blockers and issues
oboyu query "blocker issue risk concern" --mode semantic --days 7

# Extract key decisions
oboyu query "decision: decided: approved:" --days 7 --export weekly-decisions.txt
```

## Scenario: Executive Assistant

Managing executive's schedule and ensuring all meeting outcomes are tracked and actioned.

### Meeting Follow-up Workflow
```bash
# Find all meetings from today
oboyu query "meeting" --days 0

# Extract action items for follow-up
oboyu query "follow up: next steps: action required:" --days 1

# Find meetings requiring minutes
oboyu query "meeting NOT minutes" --days 3 --mode semantic
```

### Stakeholder Tracking
```bash
# All meetings with specific client
oboyu query "meeting Acme Corp" --db-path ~/indexes/meetings.db

# Executive decisions tracker
oboyu query "CEO decided approved" --mode semantic

# Board meeting preparation
oboyu query "board meeting agenda item" --days 90
```

## Advanced Meeting Search Patterns

### Action Item Management
```bash
# Find open action items
oboyu query "action: -completed -done" --regex

# Action items by assignee
oboyu query "@sarah TODO|action|assigned" 

# Overdue items
oboyu query "due: &lt; 2024-01-15" --mode semantic

# High priority actions
oboyu query "high priority urgent action" --mode hybrid
```

### Decision Tracking
```bash
# Find all decisions made
oboyu query "DECISION:|Decided:|Approved:" --file-type md

# Decisions pending approval
oboyu query "pending approval decision" --mode semantic

# Strategic decisions
oboyu query "strategic decision long-term" --mode semantic
```

## Real-World Example: Quarterly Business Review

Preparing for a QBR by gathering all relevant meeting information:

```bash
# 1. Find all Q4 meetings
oboyu query "meeting" --from 2023-10-01 --to 2023-12-31 --db-path ~/indexes/meetings.db

# 2. Extract key metrics discussed
oboyu query "revenue profit margin KPI" --from 2023-10-01 --mode semantic

# 3. Find customer feedback
oboyu query "customer feedback complaint satisfaction" --mode semantic

# 4. Gather product decisions
oboyu query "product roadmap feature decision" --from 2023-10-01

# 5. Compile action items
oboyu query "action: assigned:" --from 2023-10-01 --export qbr-actions.txt

# 6. Find risks and mitigation
oboyu query "risk mitigation concern issue" --mode semantic --export qbr-risks.txt
```

## Meeting Templates and Patterns

### Stand-up Meeting Pattern
```bash
# Daily stand-up search
standup_search() {
    echo "=== Yesterday ==="
    oboyu query "completed done finished" --days 1 --author $1
    echo "=== Today ==="
    oboyu query "working on today will" --days 0 --author $1
    echo "=== Blockers ==="
    oboyu query "blocked waiting on dependency" --days 3 --author $1
}

# Usage
standup_search "john"
```

### One-on-One Meeting Tracking
```bash
# Find all 1:1s with manager
oboyu query "1:1 one-on-one Sarah" --db-path ~/indexes/meetings.db

# Career development discussions
oboyu query "career development growth goals" --mode semantic

# Performance feedback
oboyu query "feedback performance improvement" --mode semantic
```

## Meeting Analytics

### Meeting Efficiency Analysis
```bash
# Find meetings without clear outcomes
oboyu query "meeting" --not "action: decision: next steps:" 

# Long meetings
oboyu query "meeting duration: > 2 hours" --mode semantic

# Recurring meeting effectiveness
oboyu query "weekly meeting outcomes results" --mode semantic
```

### Participation Tracking
```bash
# Find who attended most meetings
oboyu query "attendees: participants:" --format stats

# Silent participants
oboyu query "attendees:" --not "said: mentioned: proposed:"

# Most active contributors
oboyu query "proposed: suggested: recommended:" --format stats
```

## Japanese Meeting Notes

For bilingual organizations:

```bash
# Search Japanese meeting notes
oboyu query "会議 議事録" --db-path ~/indexes/meetings.db

# Find action items in Japanese
oboyu query "アクション 担当 期限" 

# Mixed language meetings
oboyu query "meeting 会議 action アクション" --mode hybrid

# Japanese names and titles
oboyu query "山田さん 部長" --db-path ~/indexes/meetings.db
```

## Meeting Note Organization Tips

### Consistent Format
```markdown
# Meeting: [Title]
Date: 2024-01-15
Attendees: @john, @sarah, @mike
Type: Weekly Sync

## Agenda
1. Project Status
2. Budget Review
3. Next Steps

## Decisions
- DECISION: Approved Q1 budget
- DECISION: Delayed feature X to Q2

## Action Items
- [ ] @john: Update project timeline by Jan 20
- [ ] @sarah: Send budget report to finance
```

### Searchable Patterns
- Use `@name` for attendees and assignees
- Use `DECISION:` for decisions
- Use `[ ]` and `[x]` for action items
- Include dates in ISO format
- Tag with project names

## Integration with Calendar

### Meeting Preparation Automation
```bash
# Script to find related content before meetings
prepare_meeting() {
    meeting_topic="$1"
    echo "Searching for: $meeting_topic"
    
    # Previous meetings
    oboyu query "$meeting_topic meeting" --days 90
    
    # Related documents
    oboyu query "$meeting_topic" --mode semantic --limit 10
    
    # Open items
    oboyu query "$meeting_topic action: -done" 
}

# Add to calendar reminder
prepare_meeting "Project Phoenix Review"
```

## Best Practices

1. **Consistent Naming**: Use "YYYY-MM-DD-Meeting-Topic.md" format
2. **Immediate Capture**: Write notes during or immediately after meetings
3. **Action Tracking**: Always mark action items clearly
4. **Regular Review**: Search for open items weekly
5. **Decision Log**: Maintain a separate decision log for quick reference

## Next Steps

- Learn about [Research Paper Search](research-papers.md) for academic meetings
- Explore [Personal Notes Search](personal-notes.md) for individual productivity
- Set up [Automation](../integration/automation.md) for meeting workflows