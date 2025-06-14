---
id: github-issues-search
title: GitHub Issues Search Examples
sidebar_position: 6
---

# GitHub Issues Search Examples

Learn how to use Oboyu to search through GitHub Issues for effective software development project management, bug tracking, and development workflow optimization. This guide covers practical scenarios for developers, project managers, and DevOps engineers managing software projects.

## Scenario: Software Development Team Lead

Managing a growing software project with hundreds of GitHub Issues across multiple repositories, you need to quickly find relevant information for sprint planning, bug triage, and feature development. Your team exports issues regularly for analysis and knowledge management.

### Setting Up GitHub Issues Index

First, organize your GitHub Issues exports and create a searchable index:

```bash
# Create directory structure for GitHub Issues
mkdir -p ~/github-projects/
mkdir -p ~/github-projects/issues-export/
mkdir -p ~/github-projects/issues-archive/

# Index GitHub Issues (JSON, Markdown, and text formats)
oboyu index ~/github-projects/issues-export \
  --include-patterns "*.json" "*.md" "*.txt" \
  --db-path ~/indexes/github-issues.db
```

### Daily Bug Triage and Management

#### Finding Similar Bugs and Error Patterns
```bash
# Search for authentication-related bugs
oboyu query --query "authentication error login failed" \
  --mode hybrid --db-path ~/indexes/github-issues.db

# Find API rate limiting issues
oboyu query --query "rate limit exceeded API throttling" \
  --mode bm25 --db-path ~/indexes/github-issues.db

# Search for database connection problems
oboyu query --query "database connection timeout pool" \
  --mode hybrid --db-path ~/indexes/github-issues.db
```

#### Sprint Planning and Feature Research
```bash
# Find feature requests for upcoming sprint
oboyu query --query "feature request enhancement user story" \
  --mode vector --db-path ~/indexes/github-issues.db

# Research implementation discussions
oboyu query --query "implementation approach architecture design" \
  --mode vector --db-path ~/indexes/github-issues.db

# Find performance improvement opportunities
oboyu query --query "performance slow optimization memory CPU" \
  --mode hybrid --db-path ~/indexes/github-issues.db
```

### Advanced GitHub Issues Search Patterns

#### Cross-Repository Analysis
```bash
# Find security-related issues across all repositories
oboyu query --query "security vulnerability CVE authentication authorization" \
  --mode hybrid --db-path ~/indexes/github-issues.db --top-k 20

# Search for breaking changes and deprecations
oboyu query --query "breaking change deprecated migration upgrade" \
  --mode vector --db-path ~/indexes/github-issues.db

# Find documentation gaps and improvements
oboyu query --query "documentation missing unclear README tutorial" \
  --mode vector --db-path ~/indexes/github-issues.db
```

#### Label and Priority-Based Searches
```bash
# Find high-priority bugs
oboyu query --query "priority:high critical urgent blocker" \
  --mode bm25 --db-path ~/indexes/github-issues.db

# Search for good first issues for new contributors
oboyu query --query "good first issue beginner friendly easy" \
  --mode vector --db-path ~/indexes/github-issues.db

# Find technical debt items
oboyu query --query "technical debt refactor cleanup legacy" \
  --mode hybrid --db-path ~/indexes/github-issues.db
```

## Scenario: DevOps Engineer

As a DevOps engineer managing CI/CD pipelines and infrastructure, you need to track deployment issues, monitoring alerts, and system reliability problems across multiple services and environments.

### Setting Up Infrastructure Issues Index

```bash
# Index issues from multiple infrastructure repositories
oboyu index ~/devops-issues/ \
  --include-patterns "*.json" "*.md" "incidents/*.txt" "alerts/*.md" \
  --db-path ~/indexes/infrastructure-issues.db \
  --chunk-size 1536  # Larger chunks for detailed incident reports
```

### Infrastructure and Deployment Management

#### CI/CD Pipeline Issues
```bash
# Find build failures and pipeline issues
oboyu query --query "build failed pipeline error CI CD deployment" \
  --mode hybrid --db-path ~/indexes/infrastructure-issues.db

# Search for Docker and container issues
oboyu query --query "Docker container image build registry pull" \
  --mode bm25 --db-path ~/indexes/infrastructure-issues.db

# Find test failures and flaky tests
oboyu query --query "test failure flaky intermittent timeout" \
  --mode vector --db-path ~/indexes/infrastructure-issues.db
```

#### Monitoring and Alerting
```bash
# Search for performance alerts and monitoring issues
oboyu query --query "alert monitoring memory CPU disk performance" \
  --mode hybrid --db-path ~/indexes/infrastructure-issues.db

# Find networking and connectivity problems
oboyu query --query "network connection timeout firewall DNS" \
  --mode hybrid --db-path ~/indexes/infrastructure-issues.db

# Search for scaling and capacity issues
oboyu query --query "scaling capacity auto-scaling load balancer" \
  --mode vector --db-path ~/indexes/infrastructure-issues.db
```

## Real-World Example: Release Planning and Risk Assessment

A practical workflow for preparing quarterly releases:

### 1. Identify Release Blockers
```bash
echo "=== Finding Release Blockers ==="
oboyu query --query "blocker critical release milestone" \
  --mode bm25 --db-path ~/indexes/github-issues.db --top-k 15

echo "=== Security Issues for Review ==="
oboyu query --query "security vulnerability audit review" \
  --mode hybrid --db-path ~/indexes/github-issues.db
```

### 2. Analyze Historical Release Issues
```bash
echo "=== Previous Release Problems ==="
oboyu query --query "rollback hotfix post-release production issue" \
  --mode vector --db-path ~/indexes/github-issues.db

echo "=== Migration and Breaking Changes ==="
oboyu query --query "migration breaking change database schema" \
  --mode hybrid --db-path ~/indexes/github-issues.db
```

### 3. Feature Completion Assessment
```bash
echo "=== Feature Development Status ==="
oboyu query --query "feature complete ready testing QA" \
  --mode vector --db-path ~/indexes/github-issues.db

echo "=== Outstanding Bug Fixes ==="
oboyu query --query "bug fix resolved closed merged" \
  --mode hybrid --db-path ~/indexes/github-issues.db
```

## Advanced Integration Patterns

### Automated Issue Analysis Scripts

Create automation scripts for regular issue analysis:

```bash
#!/bin/bash
# daily-issue-analysis.sh

ISSUES_DB="~/indexes/github-issues.db"
DATE=$(date '+%Y-%m-%d')

echo "=== Daily Issue Analysis - $DATE ==="

echo "Critical Issues Requiring Attention:"
oboyu query --query "critical urgent blocker priority:high" \
  --mode bm25 --db-path "$ISSUES_DB" --top-k 10 --format json

echo -e "\nSecurity Issues for Review:"
oboyu query --query "security vulnerability CVE audit" \
  --mode hybrid --db-path "$ISSUES_DB" --top-k 5

echo -e "\nPerformance Issues:"
oboyu query --query "performance slow memory leak optimization" \
  --mode vector --db-path "$ISSUES_DB" --top-k 8
```

### Team Standup Preparation

```bash
# standup-prep.sh
echo "=== Issues assigned to current user ==="
oboyu query --query "assigned:$USER in-progress working" \
  --mode bm25 --db-path ~/indexes/github-issues.db

echo "=== Blocked issues requiring help ==="
oboyu query --query "blocked help needed assistance review" \
  --mode vector --db-path ~/indexes/github-issues.db
```

### Integration with Development Tools

#### GitHub CLI Integration
```bash
# Export recent issues and index them
gh issue list --limit 100 --json title,body,labels,createdAt \
  > ~/github-projects/issues-export/recent-issues.json

# Update index with new issues
oboyu index ~/github-projects/issues-export/ \
  --db-path ~/indexes/github-issues.db --change-detection smart
```

#### Slack/Teams Notifications
```bash
# Find urgent issues for team notification
URGENT_ISSUES=$(oboyu query --query "urgent critical blocker" \
  --mode bm25 --db-path ~/indexes/github-issues.db --format json --top-k 5)

# Send to Slack (example integration)
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"Urgent Issues Found: $URGENT_ISSUES\"}" \
  $SLACK_WEBHOOK_URL
```

## Best Practices and Organization Tips

### Issue Export and Organization
- Export issues in JSON format for complete metadata preservation
- Organize by repository, milestone, or date for better searchability
- Include issue comments and discussions for comprehensive context
- Use consistent naming conventions: `{repo}-{date}-issues.json`

### Search Strategy Guidelines
- **Use hybrid mode** for general issue searches combining keywords and semantics
- **Use BM25 mode** for exact label, user, or ID searches
- **Use vector mode** for concept-based searches and finding related issues
- **Increase top-k** to 15-25 for exploration and comprehensive analysis

### Index Configuration for GitHub Issues
```yaml
# ~/.config/oboyu/github-issues.yaml
indexer:
  db_path: "~/indexes/github-issues.db"
  chunk_size: 1536              # Good for issue content
  chunk_overlap: 384
  use_reranker: true            # Better ranking for mixed content

crawler:
  include_patterns:
    - "*.json"
    - "*.md"  
    - "issues/**/*"
    - "bugs/**/*"
    - "features/**/*"
  exclude_patterns:
    - "*/node_modules/*"
    - "*/.git/*"

query:
  default_mode: "hybrid"
  top_k: 15
```

### Automation and Maintenance
- Set up daily/weekly index updates with `--change-detection smart`
- Create custom search aliases for common queries
- Use output formatting (`--format json`) for integration with other tools
- Monitor index size and performance for large issue collections

### Team Collaboration
- Share common search patterns and queries across team members
- Create repository-specific search configurations
- Document common issue patterns and search strategies
- Use consistent tagging and labeling for better searchability

## Next Steps

- [Technical Documentation Search](technical-docs.md) - Search through code documentation and technical specs
- [Personal Knowledge Base](personal-notes.md) - Organize individual development notes and learnings

---

This use case demonstrates Oboyu's power for managing software development workflows through intelligent GitHub Issues search. The combination of semantic understanding and precise keyword matching makes it ideal for finding relevant issues, tracking bugs, and supporting agile development processes.