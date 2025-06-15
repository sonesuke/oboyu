#!/bin/bash

# Script to clean up issue resolution worktrees after PR merge
# Based on cleanup-bug-worktree.sh but adapted for issue workflow patterns

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to display usage
usage() {
    echo "Usage: $0 <issue-number>"
    echo "Example: $0 260"
    echo ""
    echo "This script will clean up worktrees and branches created for issue resolution:"
    echo "  - Remove issue worktree directory"
    echo "  - Delete local branches matching the issue pattern"
    echo "  - Display final cleanup status"
    echo ""
    echo "Branch patterns cleaned up:"
    echo "  - issue-{number}-*"
    echo "  - {number}-*"
    echo "  - feature-{number}-*"
    echo "  - enhancement-{number}-*"
    exit 1
}

# Check if issue number is provided
if [ $# -eq 0 ]; then
    usage
fi

ISSUE_NUMBER=$1
PROJECT_DIR=$(basename "$PWD")
WORKTREE_PATH="../${PROJECT_DIR}-issue-${ISSUE_NUMBER}"

log "${BLUE}🧹 Starting cleanup for issue #${ISSUE_NUMBER}${NC}"

# Function to remove worktree
cleanup_worktree() {
    # Check if worktree exists
    if git worktree list | grep -q "$WORKTREE_PATH"; then
        log "${BLUE}📁 Removing worktree: ${WORKTREE_PATH}${NC}"
        
        if git worktree remove "$WORKTREE_PATH"; then
            log "${GREEN}✅ Worktree removed successfully${NC}"
        else
            log "${RED}❌ Failed to remove worktree${NC}"
            log "${YELLOW}💡 Try manually with: git worktree remove \"$WORKTREE_PATH\"${NC}"
            return 1
        fi
    else
        log "${YELLOW}⚠️  Worktree not found: ${WORKTREE_PATH}${NC}"
        
        # Check for alternative worktree patterns
        local alt_patterns=(
            "../${PROJECT_DIR}-${ISSUE_NUMBER}"
            "../${PROJECT_DIR}-feature-${ISSUE_NUMBER}"
            "../${PROJECT_DIR}-enhancement-${ISSUE_NUMBER}"
        )
        
        for pattern in "${alt_patterns[@]}"; do
            if git worktree list | grep -q "$pattern"; then
                log "${BLUE}📁 Found alternative worktree: ${pattern}${NC}"
                if git worktree remove "$pattern"; then
                    log "${GREEN}✅ Alternative worktree removed successfully${NC}"
                    return 0
                fi
            fi
        done
        
        log "${BLUE}ℹ️  No worktrees found for issue #${ISSUE_NUMBER}${NC}"
    fi
}

# Function to cleanup branches
cleanup_branches() {
    log "${BLUE}🌿 Searching for branches related to issue #${ISSUE_NUMBER}${NC}"
    
    # Define branch patterns to look for
    local patterns=(
        "issue-${ISSUE_NUMBER}-"
        "${ISSUE_NUMBER}-"
        "feature-${ISSUE_NUMBER}-"
        "enhancement-${ISSUE_NUMBER}-"
    )
    
    local branches_found=false
    
    for pattern in "${patterns[@]}"; do
        local branches
        branches=$(git branch | grep "$pattern" | tr -d ' *' || true)
        
        if [ -n "$branches" ]; then
            branches_found=true
            log "${BLUE}📋 Found branches matching pattern: ${pattern}*${NC}"
            
            for branch in $branches; do
                log "${BLUE}🗑️  Attempting to delete branch: ${branch}${NC}"
                
                if git branch -d "$branch" 2>/dev/null; then
                    log "${GREEN}✅ Branch ${branch} deleted successfully${NC}"
                elif git branch -D "$branch" 2>/dev/null; then
                    log "${YELLOW}⚠️  Force deleted unmerged branch: ${branch}${NC}"
                else
                    log "${RED}❌ Failed to delete branch: ${branch}${NC}"
                    log "${YELLOW}💡 You may need to delete it manually: git branch -D ${branch}${NC}"
                fi
            done
        fi
    done
    
    if [ "$branches_found" = false ]; then
        log "${BLUE}ℹ️  No branches found matching issue #${ISSUE_NUMBER} patterns${NC}"
    fi
}

# Function to cleanup remote branches (optional)
cleanup_remote_branches() {
    log "${BLUE}🌐 Checking for remote branches to clean up...${NC}"
    
    # Get the default remote (usually origin)
    local remote
    remote=$(git remote | head -n1)
    
    if [ -z "$remote" ]; then
        log "${YELLOW}⚠️  No remote repository configured${NC}"
        return
    fi
    
    # Fetch latest remote information
    if git fetch "$remote" --prune; then
        log "${GREEN}✅ Remote branches pruned${NC}"
    else
        log "${YELLOW}⚠️  Failed to prune remote branches${NC}"
    fi
}

# Function to display final status
display_final_status() {
    log "${BLUE}📊 Final cleanup status:${NC}"
    echo ""
    echo "Current worktrees:"
    git worktree list
    echo ""
    echo "Local branches:"
    git branch | head -10
    if [ "$(git branch | wc -l)" -gt 10 ]; then
        echo "... (showing first 10 branches)"
    fi
    echo ""
}

# Function to check PR status (optional information)
check_pr_status() {
    log "${BLUE}🔍 Checking PR status for issue #${ISSUE_NUMBER}...${NC}"
    
    # Try to find associated PR
    local pr_number
    pr_number=$(gh pr list --search "issue #${ISSUE_NUMBER}" --json number --jq '.[0].number' 2>/dev/null || echo "")
    
    if [ -n "$pr_number" ] && [ "$pr_number" != "null" ]; then
        local pr_state
        pr_state=$(gh pr view "$pr_number" --json state --jq '.state' 2>/dev/null || echo "UNKNOWN")
        
        case "$pr_state" in
            "MERGED")
                log "${GREEN}✅ Associated PR #${pr_number} is merged${NC}"
                ;;
            "CLOSED")
                log "${YELLOW}⚠️  Associated PR #${pr_number} is closed (not merged)${NC}"
                ;;
            "OPEN")
                log "${YELLOW}⚠️  Associated PR #${pr_number} is still open${NC}"
                log "${YELLOW}💡 You may want to wait until PR is merged before cleanup${NC}"
                ;;
            *)
                log "${BLUE}ℹ️  Could not determine PR status${NC}"
                ;;
        esac
    else
        log "${BLUE}ℹ️  No associated PR found for issue #${ISSUE_NUMBER}${NC}"
    fi
}

# Main execution
main() {
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log "${RED}❌ Not in a git repository${NC}"
        exit 1
    fi
    
    # Check PR status for context
    if command -v gh >/dev/null 2>&1; then
        check_pr_status
        echo ""
    fi
    
    # Perform cleanup operations
    cleanup_worktree
    cleanup_branches
    cleanup_remote_branches
    
    echo ""
    display_final_status
    
    log "${GREEN}🎉 Cleanup completed for issue #${ISSUE_NUMBER}!${NC}"
    echo ""
    log "${BLUE}💡 Next steps:${NC}"
    echo "  - Verify the issue is resolved and closed on GitHub"
    echo "  - Update any related documentation if needed"
    echo "  - Consider updating project roadmap or backlog"
}

# Handle help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
fi

# Validate issue number format
if ! [[ "$ISSUE_NUMBER" =~ ^[0-9]+$ ]]; then
    log "${RED}❌ Invalid issue number: $ISSUE_NUMBER${NC}"
    log "${YELLOW}💡 Issue number must be a positive integer${NC}"
    exit 1
fi

# Run main function
main