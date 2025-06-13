# Enhanced Solve Issue Workflow

This command implements an advanced GitHub issue resolution workflow with git worktree isolation, automated CI/CD monitoring, and intelligent PR management.

## 🎯 Enhanced Workflow Overview

1. **Extract Issue Information**
   - Extract issue number from branch name
   - Fetch issue details from GitHub using `gh issue view {issue_number}`
   - Understand the problem, requirements, and acceptance criteria

2. **Initial Setup & Planning**
   - Create a comprehensive todo list for the implementation
   - Break down the issue into manageable tasks
   - Plan the implementation approach

3. **⭐ Worktree Environment Creation**
   - Create isolated development environment using git worktree
   - Enable parallel issue work without branch conflicts
   - Set up clean dependency environment

4. **Development with Early PR Creation**
   - Start implementation of the solution in isolated worktree
   - After first meaningful commit, create a **Draft PR** immediately
   - Use WIP title format: `WIP: Fix #{issue_number} - {issue_title}`
   - Link the issue in PR description with `Fixes #{issue_number}` or `Closes #{issue_number}`
   - Add comment to original issue: `Working on this in PR #{pr_number}`

5. **⭐ Automated CI/CD Monitoring**
   - Launch automated PR monitoring with real-time feedback
   - Auto-detect and fix common CI/CD issues (linting, formatting, imports)
   - Continuous 2-minute interval monitoring until completion

6. **Progress Tracking with Real-time Feedback**
   - Update PR title to reflect development progress:
     - `WIP: Fix #{issue_number} - [30%] Initial setup complete`
     - `WIP: Fix #{issue_number} - [70%] Core functionality implemented`
     - `Ready: Fix #{issue_number} - {final_title}`
   - Make regular commits with descriptive messages
   - Update PR description with implementation details and progress
   - Receive automated notifications of CI/CD status changes

7. **⭐ Auto-promotion to Ready**
   - Automatic promotion from Draft to Ready when all checks pass
   - Intelligent PR title updates reflecting completion status
   - Automated reviewer assignment based on changed files

8. **⭐ Cleanup & Finalization**
   - Automated worktree and branch cleanup after PR merge
   - Clean environment management for future work
   - Final status reporting and next steps guidance

## 🛠️ Detailed Implementation Steps

### Step 1: Issue Analysis & Planning
```bash
# Extract issue number from current branch
ISSUE_NUMBER=$(git branch --show-current | grep -o '[0-9]\+' | head -1)

# Fetch detailed issue information
gh issue view $ISSUE_NUMBER

# Create comprehensive implementation plan
# - Break down requirements into specific tasks
# - Identify affected files and components
# - Plan testing strategy
# - Estimate implementation timeline
```

### Step 2: Worktree Environment Setup
```bash
# Get current project directory name
PROJECT_DIR=$(basename "$PWD")
ISSUE_NUMBER={issue_number}  # Replace with actual issue number
BRANCH_NAME=$(git branch --show-current)

# Create git worktree in parallel directory
git worktree add "../${PROJECT_DIR}-issue-${ISSUE_NUMBER}" "${BRANCH_NAME}"

# Navigate to worktree
cd "../${PROJECT_DIR}-issue-${ISSUE_NUMBER}"

# Install dependencies in the isolated environment
uv sync

# Verify worktree status
git worktree list
echo "✅ Isolated development environment ready"
```

### Step 3: Initial Development & PR Creation
```bash
# Start implementation in the worktree
# Make meaningful initial commits

# After first significant commit
git add .
git commit -m "feat: initial implementation for issue #${ISSUE_NUMBER}"
git push -u origin ${BRANCH_NAME}

# Create Draft PR with proper linking
gh pr create --draft \
  --title "WIP: Fix #${ISSUE_NUMBER} - {issue_title}" \
  --body "Fixes #${ISSUE_NUMBER}

## 🎯 Problem
{Brief description of the issue}

## 💡 Solution Approach
{High-level implementation strategy}

## 📋 Progress Checklist
- [ ] Core functionality implemented
- [ ] Tests written/updated
- [ ] Documentation updated
- [ ] Code quality checks passed

## 🔗 Related
- Issue: #${ISSUE_NUMBER}
- Branch: ${BRANCH_NAME}
- Worktree: ${PROJECT_DIR}-issue-${ISSUE_NUMBER}"

# Get PR number for monitoring
PR_NUMBER=$(gh pr list --head ${BRANCH_NAME} --json number --jq '.[0].number')
echo "📝 Draft PR #${PR_NUMBER} created"
```

### Step 4: Launch Automated Monitoring
```bash
# Start automated CI/CD monitoring in background
.claude/scripts/monitor-pr-checks.sh ${PR_NUMBER} &
MONITOR_PID=$!

echo "🤖 Automated monitoring started (PID: ${MONITOR_PID})"
echo "📊 Monitoring will:"
echo "  - Check PR status every 2 minutes"
echo "  - Auto-fix common CI/CD issues"
echo "  - Auto-promote to Ready when all checks pass"
echo "  - Run until manually stopped (Ctrl+C)"
```

### Step 5: Development with Real-time Feedback
```bash
# Continue development with regular commits
# The monitoring system will automatically:
# - Detect failing checks
# - Apply auto-fixes for common issues
# - Commit and push fixes
# - Update PR status

# Manual progress updates
gh pr edit ${PR_NUMBER} --title "WIP: Fix #${ISSUE_NUMBER} - [50%] Core implementation complete"

# Update PR description with progress
gh pr edit ${PR_NUMBER} --body "$(cat <<EOF
Fixes #${ISSUE_NUMBER}

## 🎯 Problem
{Brief description of the issue}

## 💡 Solution Approach
{Implementation details}

## ✅ Completed
- [x] Initial setup and planning
- [x] Core functionality implemented
- [x] Basic tests added

## 🔄 In Progress
- [ ] Edge case handling
- [ ] Performance optimization
- [ ] Documentation updates

## 📊 Current Status
- Implementation: 70% complete
- Tests: 60% complete
- Documentation: 30% complete

Last updated: $(date)
EOF
)"
```

### Step 6: Quality Assurance & Testing
```bash
# Run comprehensive quality checks
uv run ruff check --fix && uv run mypy
uv run pytest -m "not slow" -k "not integration"

# The monitoring system handles this automatically, but manual execution is also available:
.claude/scripts/fix-common-issues.sh
```

### Step 7: Automated Promotion (Handled Automatically)
The monitoring system will automatically:
- Detect when all CI/CD checks pass
- Update PR title from "WIP:" to "Ready:"
- Convert PR from Draft to Ready for Review
- Notify about completion

### Step 8: Cleanup After Merge
```bash
# After PR is merged, return to main project
cd ../{main_project_directory}

# Use automated cleanup script
.claude/scripts/cleanup-issue-worktree.sh ${ISSUE_NUMBER}

# Verify cleanup
git worktree list
echo "🧹 Cleanup completed"
```

## 📋 Enhanced Implementation Checklist

### Planning Phase
- [ ] Issue analysis completed
- [ ] Implementation plan created
- [ ] Worktree environment set up
- [ ] Dependencies installed in worktree

### Development Phase
- [ ] First meaningful commit made
- [ ] Draft PR created with proper linking
- [ ] Automated monitoring launched
- [ ] Progress tracking implemented
- [ ] Core functionality implemented
- [ ] Tests written/updated

### Quality Assurance Phase
- [ ] Automated CI/CD fixes applied
- [ ] Code quality checks passed (lint, type, tests)
- [ ] PR description updated with implementation details
- [ ] All checks passing consistently

### Completion Phase
- [ ] PR auto-promoted to Ready for Review
- [ ] Reviewers assigned
- [ ] PR merged successfully
- [ ] Worktree and branch cleanup completed

## 🚀 Automation Features

### Automated CI/CD Monitoring
- **Real-time Check Status**: Monitor PR checks every 2 minutes
- **Auto-fix Common Issues**: Automatically fix linting, formatting, and import issues
- **Intelligent Promotion**: Auto-promote Draft to Ready when all checks pass
- **Continuous Feedback**: Real-time notifications of status changes

### Automated Issue Fixes
- **Linting**: `ruff check --fix` for code style issues
- **Formatting**: `ruff format` for consistent code formatting
- **Import Ordering**: Automatic import organization
- **Type Checking**: MyPy validation with helpful error messages

### Automated Cleanup
- **Worktree Management**: Clean removal of isolated development environments
- **Branch Cleanup**: Automatic deletion of feature branches after merge
- **Status Reporting**: Final cleanup verification and next steps

## 🔗 Issue Linking Best Practices

- Always include `Fixes #{issue_number}` or `Closes #{issue_number}` in PR description
- Reference related issues and discussions
- Add meaningful commit messages that reference the issue
- Update issue with PR link for transparency
- Use consistent branch naming: `{issue_number}-{descriptive-name}`

## 💡 Enhanced Tips for Success

### Parallel Development
- Use worktrees to work on multiple issues simultaneously
- Keep each worktree focused on a single issue
- Maintain clean separation between different features

### Automation Benefits
- Let the monitoring system handle routine CI/CD issues
- Focus on implementation while automation handles quality gates
- Trust the auto-promotion system for consistent PR management

### Quality Management
- Rely on automated fixes for common issues
- Address complex problems that require manual intervention
- Monitor automated feedback for continuous improvement

### Team Collaboration
- Create PR early for visibility and collaboration opportunities
- Use descriptive commit messages and PR updates
- Keep team informed of progress through automated PR comments
- Address review feedback promptly after auto-promotion

## 📊 Monitoring Dashboard

When monitoring is active, you'll see:
```
[2024-01-15 10:30:00] 🚀 Starting automated monitoring for PR #123
[2024-01-15 10:30:00] 📝 Monitoring interval: 120s (2 minutes)
[2024-01-15 10:30:00] 🛑 Press Ctrl+C to stop monitoring

[2024-01-15 10:30:00] 📊 PR #123 Status:
  - State: OPEN
  - Draft: true
  - Checks: PENDING

  Individual Checks:
    lint: pending
    test: pending
    type-check: pending

[2024-01-15 10:32:00] ❌ Some checks failed. Analyzing issues...
[2024-01-15 10:32:00] 🔧 Detected linting/type issues. Running auto-fix...
[2024-01-15 10:32:30] ✅ Auto-fixes committed
[2024-01-15 10:34:00] ✅ All checks passed! (1/2)
[2024-01-15 10:36:00] ✅ All checks passed! (2/2)
[2024-01-15 10:36:00] 🎉 PR #123 promoted to Ready for Review!
```