# Bug Fix Workflow

Fix a bug using isolated git worktree environment.

## Prerequisites
- Ensure you're in the main project directory
- All changes in main branch should be committed or stashed

## Workflow Steps

### 1. Issue & Branch Creation
```bash
# Create GitHub issue with bug label
gh issue create --title "Bug: {bug_description}" --label "bug" --body "Bug description here"

# Note the issue number (e.g., 254)
# Create and checkout new branch
git checkout -b bug-{issue_number}-{descriptive-name}
# Example: git checkout -b bug-254-authentication-timeout
```

### 2. Worktree Setup
```bash
# Get the current directory name
PROJECT_DIR=$(basename "$PWD")
ISSUE_NUMBER={issue_number}  # Replace with actual issue number
BRANCH_NAME=$(git branch --show-current)

# Create git worktree in parallel directory
git worktree add "../${PROJECT_DIR}-bug-${ISSUE_NUMBER}" "${BRANCH_NAME}"

# Navigate to worktree
cd "../${PROJECT_DIR}-bug-${ISSUE_NUMBER}"

# Install dependencies in the worktree
uv sync

# Verify worktree status
git worktree list
```

### 3. Isolated Development
- Work exclusively in worktree environment
- Reproduce the bug first
- Implement fix with clear, incremental commits
- Add regression tests
- Run tests: `uv run pytest -m "not slow" -k "not integration"`

### 4. PR Creation & Management
```bash
# After first commit in the worktree
git add .
git commit -m "Initial bug fix for #{issue_number}"
git push -u origin $(git branch --show-current)

# Create Draft PR with proper linking
gh pr create --draft \
  --title "WIP: Fix bug #{issue_number} - {bug_description}" \
  --body "Fixes #{issue_number}

## Problem
{Brief description of the bug}

## Solution
{Brief description of the fix}

## Testing
- [ ] Bug reproduction test added
- [ ] Existing tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Fix implementation complete
- [ ] Tests added/updated
- [ ] Documentation updated if needed"

# The issue number from the branch name will automatically link the PR to the issue
```

### 5. Quality Assurance Checklist
- [ ] Bug reproduction confirmed
- [ ] Fix implementation complete
- [ ] Regression tests added
- [ ] All existing tests pass
- [ ] Lint and type checks pass: `uv run ruff check --fix && uv run mypy`
- [ ] Performance impact assessed
- [ ] Documentation updated if needed

### 6. Cleanup
```bash
# After PR is merged, return to main project directory
cd ../{main_project_directory}

# Use the cleanup script (recommended)
.claude/scripts/cleanup-bug-worktree.sh {issue_number}

# Or manually:
# Remove the worktree
git worktree remove "../${PROJECT_DIR}-bug-${ISSUE_NUMBER}"

# Delete the local branch
git branch -d bug-{issue_number}-{name}

# Verify cleanup
git worktree list
```

## Benefits

- **Isolation**: No interference with main development
- **Focus**: Dedicated environment for bug investigation
- **Parallel Work**: Continue main development while fixing bugs
- **Clean Process**: Structured workflow from bug to fix

## Worktree Best Practices

1. **One Bug, One Worktree**: Create a separate worktree for each bug fix to maintain isolation
2. **Clean State**: Always start from a clean main/master branch state
3. **Regular Commits**: Make small, focused commits in the worktree to track progress
4. **Test in Isolation**: Run full test suite in the worktree before creating PR
5. **Avoid Cross-Dependencies**: Don't reference files from main workspace in worktree
6. **Timely Cleanup**: Remove worktrees promptly after PR merge to avoid clutter
7. **Document Context**: Include bug reproduction steps in commit messages

## Common Issues & Solutions

- **Worktree Already Exists**: Remove it first with `git worktree remove`
- **Branch Already Exists**: Delete with `git branch -D` if needed
- **Dependencies Out of Sync**: Always run `uv sync` after creating worktree
- **Can't Remove Worktree**: Ensure you're not inside the worktree directory