# Critical Bug Fix Workflow

Fast-track workflow for critical bugs requiring immediate attention.

## Prerequisites
- Ensure you have necessary permissions for expedited review
- Main branch should be in clean state

## Emergency Workflow

### 1. Create Critical Bug Issue
```bash
# Create high-priority issue
gh issue create \
  --title "CRITICAL: Bug - {bug_description}" \
  --label "bug,critical,high-priority" \
  --assignee "@me" \
  --body "## Critical Bug Report

**Severity**: CRITICAL
**Impact**: {describe user/system impact}
**Affected Systems**: {list affected components}

## Description
{detailed bug description}

## Immediate Mitigation
{any temporary workarounds if available}

## Root Cause (if known)
{preliminary analysis}"

# Note the issue number for branch creation
```

### 2. Rapid Branch & Worktree Setup
```bash
# Create hotfix branch directly from main
ISSUE_NUMBER={issue_number}
git checkout main
git pull origin main
git checkout -b hotfix-${ISSUE_NUMBER}-{short-description}

# Create isolated worktree
PROJECT_DIR=$(basename "$PWD")
git worktree add "../${PROJECT_DIR}-hotfix-${ISSUE_NUMBER}" $(git branch --show-current)
cd "../${PROJECT_DIR}-hotfix-${ISSUE_NUMBER}"
uv sync
```

### 3. Expedited Development
- Focus on minimal, targeted fix
- Add only essential tests for the fix
- Document any technical debt incurred

### 4. Fast-Track PR Creation
```bash
# Push and create PR with critical markers
git add .
git commit -m "HOTFIX: Critical bug fix for #{issue_number}

- Root cause: {brief explanation}
- Fix: {what was changed}
- Impact: {systems affected}
"
git push -u origin $(git branch --show-current)

# Create PR with expedited review request
gh pr create \
  --title "CRITICAL: Fix bug #{issue_number} - {bug_description}" \
  --label "bug,critical,needs-immediate-review" \
  --body "## 🚨 CRITICAL BUG FIX

**Issue**: #{issue_number}
**Severity**: CRITICAL
**Review Priority**: IMMEDIATE

## Problem
{concise problem description}

## Solution
{concise fix description}

## Testing
- [ ] Fix verified in isolated environment
- [ ] Regression test added
- [ ] No new issues introduced

## Deployment Notes
{any special deployment considerations}

**⚡ This PR requires expedited review and merge**"
```

### 5. Expedited Review Process
- Request immediate review from available team members
- Bypass normal review queue
- Merge upon approval (minimum 1 reviewer for critical fixes)

### 6. Post-Fix Actions
```bash
# Quick cleanup after merge
cd ../{main_project_directory}
.claude/scripts/cleanup-bug-worktree.sh ${ISSUE_NUMBER}

# Create follow-up issue for comprehensive fix if needed
gh issue create \
  --title "Follow-up: Comprehensive fix for critical bug #{issue_number}" \
  --label "bug,technical-debt" \
  --body "Follow-up to emergency fix in PR #xxx
  
## Tasks
- [ ] Add comprehensive test coverage
- [ ] Refactor quick fix if needed
- [ ] Update documentation
- [ ] Performance optimization if applicable"
```

## Critical Bug Criteria
Use this workflow ONLY when:
- System is down or severely impaired
- Data loss is occurring or imminent
- Security vulnerability is actively exploited
- Major feature is completely broken in production

## Key Differences from Standard Bug Fix
- Bypasses normal review queue
- Minimal fix approach (technical debt acceptable)
- Immediate reviewer assignment
- Shortened test requirements
- Mandatory follow-up issue creation