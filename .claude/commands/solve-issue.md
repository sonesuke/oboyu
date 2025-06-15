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

4. **Development with Immediate PR Creation**
   - Create a **Draft PR** immediately after starting work in isolated worktree
   - Use WIP title format: `WIP: Fix #{issue_number} - {issue_title}`
   - Link the issue in PR description with `Fixes #{issue_number}` or `Closes #{issue_number}`
   - Add comment to original issue: `Working on this in PR #{pr_number}`
   - Begin implementation of the solution with full transparency

5. **⭐ Automated CI/CD Monitoring**
   - Launch automated PR monitoring with real-time feedback
   - Auto-detect and fix common CI/CD issues (linting, formatting, imports)
   - Continuous 2-minute interval monitoring until completion

6. **Progress Tracking with Real-time Feedback**
   - Update PR title to reflect development progress:
     - `WIP: Fix #{issue_number} - [30%] Initial setup complete`
     - `WIP: Fix #{issue_number} - [70%] Core functionality implemented`
     - `Fix #{issue_number} - {final_title}`
   - Make regular commits with descriptive messages
   - Update PR description with implementation details and progress
   - Receive automated notifications of CI/CD status changes

7. **⭐ Auto-promotion to Ready**
   - Automatic promotion from Draft to Ready when all checks pass
   - Intelligent PR title updates reflecting completion status
   - Automated reviewer assignment based on changed files

8. **⭐ Autonomous Completion**
   - Continuous monitoring until PR merge/close
   - Automated worktree and branch cleanup after completion
   - Final status reporting without user intervention
   - **Fully autonomous from monitoring start to completion**

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

Use `.worktree/` directory for isolated development environments. Create if it doesn't exist.

```bash
# Ensure you're on latest main branch
git checkout main && git pull origin main

# Create worktree with new branch
ISSUE_NUMBER={issue_number}  # Replace with actual issue number
mkdir -p .worktree
git worktree add ".worktree/issue-${ISSUE_NUMBER}" -b "${ISSUE_NUMBER}-{descriptive-branch-name}"

# Navigate to worktree and install dependencies
cd ".worktree/issue-${ISSUE_NUMBER}"
uv sync

# Verify setup
git worktree list
echo "✅ Isolated development environment ready"
```

### Step 3: Immediate PR Creation & Initial Development
```bash
# Create Draft PR immediately after worktree setup
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
echo "📝 Draft PR #${PR_NUMBER} created immediately for transparency"

# Now start implementation with full visibility
git add .
git commit -m "feat: initial setup for issue #${ISSUE_NUMBER}" --allow-empty
git push -u origin ${BRANCH_NAME}
echo "🚀 Ready to begin development with PR tracking active"
```

### Step 4: Launch Automated Monitoring

**ALWAYS** launch automated monitoring regardless of task complexity for consistency.

```bash
# Check if automation scripts exist
if [[ ! -f "scripts/ci/monitor-pr-checks.sh" ]]; then
    echo "⚠️  Automation scripts not found. Using manual workflow."
    exit 1
fi

# Launch automated monitoring in FOREGROUND (ALWAYS, regardless of task complexity)
# IMPORTANT: Run in foreground so Claude Code can properly wait for completion
scripts/ci/monitor-pr-checks.sh ${PR_NUMBER}

# The script will run until:
# - All CI/CD checks pass and PR is promoted to Ready
# - PR is merged or closed
# - Manual interruption (Ctrl+C)
```

**Note**: Automation provides consistent workflow regardless of change size

### 🤖 Claude Code Implementation Guide

**For Claude Code users**: The monitoring system runs in FOREGROUND for proper execution flow:

1. **RUN** monitoring in foreground (no `&` background operator)
2. **WAIT** for monitoring script to complete its full cycle
3. **CONTINUE** automatically when monitoring exits successfully
4. **HANDLE** any errors that cause monitoring to exit early
5. **PROCEED** to cleanup only after monitoring completes

**Critical**: Use foreground execution so Claude Code can properly wait for CI/CD completion and handle the workflow sequentially. The monitoring script will exit when:
- All checks pass and PR is promoted to Ready
- PR is merged/closed
- Fatal error occurs requiring intervention

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
scripts/dev/fix-common-issues.sh
```

### Step 7: Automated Promotion (Fully Automated)
The monitoring system will automatically:
- Wait for ALL CI/CD checks to complete (no PENDING or IN_PROGRESS states)
- Verify 2 consecutive successful check cycles for stability
- Update PR title from "WIP:" to final clean title
- Convert PR from Draft to Ready for Review
- **Continue automatically to cleanup phase without user intervention**

### Step 8: Automated Completion & Status Report

**IMPORTANT**: This step executes automatically when monitoring system detects PR completion.

The workflow will:
1. **Monitor PR until merge/close**: Automated system continues monitoring
2. **Auto-cleanup on completion**: Worktree and branches cleaned automatically  
3. **Status reporting**: Final summary provided
4. **No user intervention required**: Fully autonomous completion

```bash
# This runs automatically via monitoring system when PR is merged/closed:
if [[ -f "scripts/dev/cleanup-issue-worktree.sh" ]]; then
    scripts/dev/cleanup-issue-worktree.sh ${ISSUE_NUMBER}
else
    # Manual cleanup if script unavailable
    cd ../../  # Return to project root
    git worktree remove ".worktree/issue-${ISSUE_NUMBER}" --force
    rm -rf ".worktree/issue-${ISSUE_NUMBER}"
    git push origin --delete "${ISSUE_NUMBER}-{branch-name}"
    git checkout main && git pull origin main
    git branch -d "${ISSUE_NUMBER}-{branch-name}"
fi

echo "🎉 Issue #${ISSUE_NUMBER} workflow completed successfully!"
```

**Autonomous Operation**: Once monitoring starts, the entire workflow runs to completion without requiring user input.

## 📋 Enhanced Implementation Checklist

### Planning Phase
- [ ] Issue analysis completed
- [ ] Implementation plan created
- [ ] Worktree environment set up
- [ ] Dependencies installed in worktree

### Development Phase
- [ ] Draft PR created immediately with proper linking
- [ ] Automated monitoring launched
- [ ] Initial setup commit made
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
- **Real-time Check Status**: Monitor PR checks every 30 seconds
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

### Claude Code Specific Considerations
- **Directory Restrictions**: Use `.worktree/` instead of `../` directories
- **Script Verification**: Always check if automation scripts exist before using
- **Task Assessment**: Evaluate complexity before applying full workflow
- **Git Operations**: Be mindful of branch creation order (main → worktree → branch)

### Parallel Development
- Use worktrees to work on multiple issues simultaneously
- Keep each worktree focused on a single issue
- Maintain clean separation between different features
- Ensure `.worktree/` is properly ignored in git

### Automation Benefits
- Let the monitoring system handle routine CI/CD issues
- Focus on implementation while automation handles quality gates
- Trust the auto-promotion system for consistent PR management
- Skip automation for simple single-file changes

### Quality Management
- Rely on automated fixes for common issues
- Address complex problems that require manual intervention
- Monitor automated feedback for continuous improvement
- Always run manual checks if automation is unavailable

### Team Collaboration
- Create PR early for visibility and collaboration opportunities
- Use descriptive commit messages and PR updates
- Keep team informed of progress through automated PR comments
- Address review feedback promptly after auto-promotion

## ⚠️ Common Pitfalls and Solutions

### Worktree Issues
- **Problem**: `fatal: branch already checked out`
  - **Solution**: Create worktree from main branch, not feature branch
- **Problem**: Cannot navigate to worktree
  - **Solution**: Use `.worktree/` structure within project directory

### Script Dependencies
- **Problem**: Automation scripts not found
  - **Solution**: Verify scripts exist in `scripts/` before use
- **Problem**: Monitoring fails to start
  - **Solution**: Fall back to manual workflow, don't block on automation

### Complexity Assessment
- **Problem**: Over-engineering simple tasks
  - **Solution**: Use judgment - simple tasks don't need full automation
- **Problem**: Under-estimating complex tasks
  - **Solution**: Use full workflow for multi-file changes

## 📊 Monitoring Dashboard

When monitoring is active, you'll see:
```
[2024-01-15 10:30:00] 🚀 Starting automated monitoring for PR #123
[2024-01-15 10:30:00] 📝 Monitoring interval: 30s (30 seconds)
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
[2024-01-15 10:36:00] 🔚 Monitoring complete. Exiting...

# Script exits here, Claude Code continues to next step
```