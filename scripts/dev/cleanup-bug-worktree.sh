#!/bin/bash

# Script to clean up bug fix worktrees after PR merge

# Get the bug issue number from command line argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <bug-issue-number>"
    echo "Example: $0 253"
    exit 1
fi

ISSUE_NUMBER=$1
PROJECT_DIR=$(basename "$PWD")
WORKTREE_PATH="../${PROJECT_DIR}-bug-${ISSUE_NUMBER}"

# Check if worktree exists
if git worktree list | grep -q "$WORKTREE_PATH"; then
    echo "Removing worktree: $WORKTREE_PATH"
    git worktree remove "$WORKTREE_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ Worktree removed successfully"
    else
        echo "✗ Failed to remove worktree"
        exit 1
    fi
else
    echo "Worktree not found: $WORKTREE_PATH"
fi

# Find and delete the bug branch
BRANCH_PATTERN="bug-${ISSUE_NUMBER}-"
BRANCHES=$(git branch | grep "$BRANCH_PATTERN" | tr -d ' *')

if [ -n "$BRANCHES" ]; then
    for branch in $BRANCHES; do
        echo "Deleting branch: $branch"
        git branch -d "$branch"
        
        if [ $? -eq 0 ]; then
            echo "✓ Branch $branch deleted successfully"
        else
            echo "✗ Failed to delete branch $branch (may not be fully merged)"
            echo "  Use 'git branch -D $branch' to force delete if needed"
        fi
    done
else
    echo "No branches found matching pattern: $BRANCH_PATTERN"
fi

echo ""
echo "Current worktrees:"
git worktree list