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
oboyu query --query "meeting with Sarah marketing"

# All meetings about a project
oboyu query --query "Project Phoenix meeting" --db-path ~/indexes/meetings.db

# Meetings with specific agenda items
oboyu query --query "agenda budget discussion" --mode vector
```

#### Gathering Context
```bash
# Find action items from last meeting
oboyu query --query "action: assigned: TODO:"

# Check previous decisions
oboyu query --query "decided: agreed: approved:" --mode vector

# Find related documents mentioned
oboyu query --query "refer to document attachment" --mode vector
```

## Scenario: Project Manager's Workflow

Managing multiple projects with various stakeholders and tracking all commitments.

### Daily Meeting Review
```bash
# Morning review of yesterday's meetings
oboyu query --query "meeting"

# Find all action items assigned to you
oboyu query --query "@yourname action OR assigned"

# Check upcoming deliverables mentioned
oboyu query --query "deadline due date deliverable" --mode vector
```

### Weekly Status Compilation
```bash
# Gather all project updates
oboyu query --query "status update progress"

# Find blockers and issues
oboyu query --query "blocker issue risk concern" --mode vector

# Extract key decisions
oboyu query --query "decision: decided: approved:"
```

## Scenario: Executive Assistant

Managing executive's schedule and ensuring all meeting outcomes are tracked and actioned.

### Meeting Follow-up Workflow
```bash
# Find all meetings from today
oboyu query --query "meeting"

# Extract action items for follow-up
oboyu query --query "follow up: next steps: action required:"

# Find meetings requiring minutes
oboyu query --query "meeting NOT minutes" --mode vector
```

### Stakeholder Tracking
```bash
# All meetings with specific client
oboyu query --query "meeting Acme Corp" --db-path ~/indexes/meetings.db

# Executive decisions tracker
oboyu query --query "CEO decided approved" --mode vector

# Board meeting preparation
oboyu query --query "board meeting agenda item"
```

## Advanced Meeting Search Patterns

### Action Item Management
```bash
# Find open action items
oboyu query --query "action: -completed -done"

# Action items by assignee
oboyu query --query "@sarah TODO|action|assigned" 

# Overdue items
oboyu query --query "due: < 2024-01-15" --mode vector

# High priority actions
oboyu query --query "high priority urgent action" --mode hybrid
```

### Decision Tracking
```bash
# Find all decisions made
oboyu query --query "DECISION:|Decided:|Approved:"

# Decisions pending approval
oboyu query --query "pending approval decision" --mode vector

# Strategic decisions
oboyu query --query "strategic decision long-term" --mode vector
```

## Real-World Example: Quarterly Business Review

Preparing for a QBR by gathering all relevant meeting information:

```bash
# 1. Find all Q4 meetings
oboyu query --query "meeting" --db-path ~/indexes/meetings.db

# 2. Extract key metrics discussed
oboyu query --query "revenue profit margin KPI" --mode vector

# 3. Find customer feedback
oboyu query --query "customer feedback complaint satisfaction" --mode vector

# 4. Gather product decisions
oboyu query --query "product roadmap feature decision"

# 5. Compile action items
oboyu query --query "action: assigned:"

# 6. Find risks and mitigation
oboyu query --query "risk mitigation concern issue" --mode vector
```

## Meeting Templates and Patterns

### Stand-up Meeting Pattern
```bash
# Daily stand-up search
standup_search() {
    echo "=== Yesterday ==="
    oboyu query --query "completed done finished"
    echo "=== Today ==="
    oboyu query --query "working on today will"
    echo "=== Blockers ==="
    oboyu query --query "blocked waiting on dependency"
}

# Usage
standup_search "john"
```

### One-on-One Meeting Tracking
```bash
# Find all 1:1s with manager
oboyu query --query "1:1 one-on-one Sarah" --db-path ~/indexes/meetings.db

# Career development discussions
oboyu query --query "career development growth goals" --mode vector

# Performance feedback
oboyu query --query "feedback performance improvement" --mode vector
```

## Meeting Analytics

### Meeting Efficiency Analysis
```bash
# Find meetings without clear outcomes
oboyu query --query "meeting" --not "action: decision: next steps:" 

# Long meetings
oboyu query --query "meeting duration: > 2 hours" --mode vector

# Recurring meeting effectiveness
oboyu query --query "weekly meeting outcomes results" --mode vector
```

### Participation Tracking
```bash
# Find who attended most meetings
oboyu query --query "attendees: participants:"

# Silent participants
oboyu query --query "attendees:" --not "said: mentioned: proposed:"

# Most active contributors
oboyu query --query "proposed: suggested: recommended:"
```

## Japanese Meeting Notes

For bilingual organizations:

```bash
# Search Japanese meeting notes
oboyu query --query "会議 議事録" --db-path ~/indexes/meetings.db

# Find action items in Japanese
oboyu query --query "アクション 担当 期限" 

# Mixed language meetings
oboyu query --query "meeting 会議 action アクション" --mode hybrid

# Japanese names and titles
oboyu query --query "山田さん 部長" --db-path ~/indexes/meetings.db
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
    oboyu query --query "$meeting_topic meeting"
    
    # Related documents
    oboyu query --query "$meeting_topic" --mode vector --top-k 10
    
    # Open items
    oboyu query --query "$meeting_topic action: -done" 
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