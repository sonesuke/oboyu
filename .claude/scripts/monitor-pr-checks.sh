#!/bin/bash

# Script to monitor PR checks with automated CI/CD monitoring and auto-promotion
# Monitors PR status and automatically promotes Draft PR to Ready when all checks pass

set -e

# Function to display usage
usage() {
    echo "Usage: $0 <pr-number>"
    echo "Example: $0 123"
    echo ""
    echo "This script will:"
    echo "- Monitor PR checks every 2 minutes"
    echo "- Auto-fix common CI/CD issues when detected"
    echo "- Auto-promote Draft PR to Ready when all checks pass"
    echo "- Continue monitoring until manually stopped (Ctrl+C)"
    exit 1
}

# Check if PR number is provided
if [ $# -eq 0 ]; then
    usage
fi

PR_NUMBER=$1
MONITOR_INTERVAL=120  # 2 minutes in seconds

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

# Function to check if PR exists
check_pr_exists() {
    if ! gh pr view "$PR_NUMBER" &>/dev/null; then
        log "${RED}❌ PR #$PR_NUMBER not found${NC}"
        exit 1
    fi
}

# Function to get PR status
get_pr_status() {
    gh pr view "$PR_NUMBER" --json isDraft,state,statusCheckRollupState --jq '{isDraft, state, checksState: .statusCheckRollupState}'
}

# Function to get detailed check information
get_check_details() {
    gh pr checks "$PR_NUMBER" --json name,state,conclusion,detailsUrl
}

# Function to promote draft PR to ready
promote_to_ready() {
    local pr_title
    pr_title=$(gh pr view "$PR_NUMBER" --json title --jq '.title')
    
    # Remove WIP prefix for clean final title
    if [[ "$pr_title" == WIP:* ]]; then
        new_title="${pr_title#WIP: }"
        gh pr edit "$PR_NUMBER" --title "$new_title"
        log "${GREEN}✅ Updated PR title: $new_title${NC}"
    fi
    
    # Mark as ready for review
    gh pr ready "$PR_NUMBER"
    log "${GREEN}🎉 PR #$PR_NUMBER promoted to Ready for Review!${NC}"
}

# Function to analyze and fix common issues
analyze_and_fix_issues() {
    local check_details
    check_details=$(get_check_details)
    
    # Check for specific failure patterns
    local has_lint_issues=false
    local has_type_issues=false
    local has_test_failures=false
    
    # Parse check results for common patterns
    if echo "$check_details" | jq -r '.[] | select(.state == "FAILURE" or .conclusion == "FAILURE") | .name' | grep -qi "lint\|ruff\|format"; then
        has_lint_issues=true
    fi
    
    if echo "$check_details" | jq -r '.[] | select(.state == "FAILURE" or .conclusion == "FAILURE") | .name' | grep -qi "type\|mypy"; then
        has_type_issues=true
    fi
    
    if echo "$check_details" | jq -r '.[] | select(.state == "FAILURE" or .conclusion == "FAILURE") | .name' | grep -qi "test\|pytest"; then
        has_test_failures=true
    fi
    
    # Attempt to fix issues if detected
    if [ "$has_lint_issues" = true ] || [ "$has_type_issues" = true ]; then
        log "${YELLOW}🔧 Detected linting/type issues. Running auto-fix...${NC}"
        .claude/scripts/fix-common-issues.sh
        return 0
    fi
    
    if [ "$has_test_failures" = true ]; then
        log "${YELLOW}⚠️  Test failures detected. Manual intervention may be required.${NC}"
        echo "$check_details" | jq -r '.[] | select(.state == "FAILURE" or .conclusion == "FAILURE") | select(.name | test("test|pytest"; "i")) | "  - " + .name + ": " + .detailsUrl'
    fi
    
    return 1
}

# Function to display current status
display_status() {
    local status_json
    status_json=$(get_pr_status)
    
    local is_draft
    local state
    local checks_state
    
    is_draft=$(echo "$status_json" | jq -r '.isDraft')
    state=$(echo "$status_json" | jq -r '.state')
    checks_state=$(echo "$status_json" | jq -r '.checksState')
    
    log "${BLUE}📊 PR #$PR_NUMBER Status:${NC}"
    echo "  - State: $state"
    echo "  - Draft: $is_draft"
    echo "  - Checks: $checks_state"
    
    # Show individual check status
    local check_details
    check_details=$(get_check_details)
    
    if [ "$check_details" != "null" ] && [ "$check_details" != "[]" ]; then
        echo ""
        echo "  Individual Checks:"
        echo "$check_details" | jq -r '.[] | "    " + .name + ": " + (.state // .conclusion // "pending")'
    fi
    
    echo ""
}

# Function to monitor PR
monitor_pr() {
    log "${BLUE}🚀 Starting automated monitoring for PR #$PR_NUMBER${NC}"
    log "${BLUE}📝 Monitoring interval: ${MONITOR_INTERVAL}s (2 minutes)${NC}"
    log "${BLUE}🛑 Press Ctrl+C to stop monitoring${NC}"
    echo ""
    
    local consecutive_successes=0
    local required_successes=2  # Require 2 consecutive successful checks before promotion
    
    while true; do
        display_status
        
        local status_json
        status_json=$(get_pr_status)
        
        local is_draft
        local checks_state
        
        is_draft=$(echo "$status_json" | jq -r '.isDraft')
        checks_state=$(echo "$status_json" | jq -r '.checksState')
        
        # Check if PR is closed/merged
        local state
        state=$(echo "$status_json" | jq -r '.state')
        if [ "$state" = "MERGED" ] || [ "$state" = "CLOSED" ]; then
            log "${GREEN}✅ PR #$PR_NUMBER is $state. Stopping monitoring.${NC}"
            break
        fi
        
        # Handle different check states
        case "$checks_state" in
            "SUCCESS")
                consecutive_successes=$((consecutive_successes + 1))
                log "${GREEN}✅ All checks passed! (${consecutive_successes}/${required_successes})${NC}"
                
                if [ "$consecutive_successes" -ge "$required_successes" ] && [ "$is_draft" = "true" ]; then
                    promote_to_ready
                    log "${GREEN}🎯 Monitoring complete! PR is ready for review.${NC}"
                    break
                elif [ "$is_draft" = "false" ]; then
                    log "${GREEN}✅ PR is already ready for review and all checks pass!${NC}"
                    break
                fi
                ;;
            "FAILURE")
                consecutive_successes=0
                log "${RED}❌ Some checks failed. Analyzing issues...${NC}"
                if analyze_and_fix_issues; then
                    log "${YELLOW}🔧 Auto-fix applied. Continuing monitoring...${NC}"
                else
                    log "${YELLOW}⚠️  Issues require manual intervention.${NC}"
                fi
                ;;
            "PENDING"|"null")
                consecutive_successes=0
                log "${YELLOW}⏳ Checks are still running...${NC}"
                ;;
            *)
                consecutive_successes=0
                log "${YELLOW}❓ Unknown check state: $checks_state${NC}"
                ;;
        esac
        
        log "${BLUE}💤 Waiting ${MONITOR_INTERVAL}s before next check...${NC}"
        echo "$(printf '=%.0s' {1..50})"
        sleep "$MONITOR_INTERVAL"
    done
}

# Main execution
main() {
    # Trap Ctrl+C to exit gracefully
    trap 'log "${YELLOW}🛑 Monitoring stopped by user${NC}"; exit 0' INT
    
    log "${BLUE}🔍 Checking if PR #$PR_NUMBER exists...${NC}"
    check_pr_exists
    
    log "${GREEN}✅ PR #$PR_NUMBER found. Starting monitoring...${NC}"
    monitor_pr
}

# Run main function
main