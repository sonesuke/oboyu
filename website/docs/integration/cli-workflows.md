---
id: cli-workflows
title: CLI Workflow Examples
sidebar_position: 2
---

# CLI Workflow Examples

Master Oboyu's command-line interface with these practical workflow examples. Learn how to create efficient scripts, automate common tasks, and integrate Oboyu into your daily workflow.

## CLI Basics

### Command Structure

```bash
oboyu [command] [subcommand] [options] [arguments]

# Examples:
oboyu index ~/Documents --db-path ~/indexes/personal.db
oboyu query "search term" --limit 10
oboyu config set query.top_k 20
```

### Global Options

```bash
# Specify config file
oboyu --config ~/.config/oboyu/work.yaml query "meeting"

# Set log level
oboyu --log-level DEBUG index ~/Documents

# Output format
oboyu --output json query "search"

# Help for any command
oboyu [command] --help
```

## Essential Workflows

### Daily Search Routine

```bash
#!/bin/bash
# daily-search.sh - Morning search routine

echo "=== Daily Search Routine ==="

# Update index with recent changes
echo "Updating index..."
oboyu index ~/Documents --update --quiet

# Search for today's tasks
echo -e "\n📋 Today's Tasks:"
oboyu query "TODO today OR deadline $(date +%Y-%m-%d)" --limit 5

# Recent meeting notes
echo -e "\n📝 Recent Meetings:"
oboyu query "meeting" --days 3 --limit 5

# Urgent items
echo -e "\n🚨 Urgent Items:"
oboyu query "urgent OR ASAP OR critical" --days 7 --limit 5
```

### Project Search Workflow

```bash
#!/bin/bash
# project-search.sh - Search within a specific project

PROJECT=${1:-"Project Alpha"}

# Function to search project documents
search_project() {
    local query="$1"
    local mode="${2:-hybrid}"
    
    oboyu query "$query" \
        --filter "path:*/$PROJECT/*" \
        --mode "$mode" \
        --limit 10 \
        --format colored
}

# Interactive project search
while true; do
    echo -n "Search $PROJECT (q to quit): "
    read query
    
    [[ "$query" == "q" ]] && break
    
    search_project "$query"
    echo -e "\n---\n"
done
```

### Document Discovery

```bash
#!/bin/bash
# discover-docs.sh - Find related documents

# Find documents similar to a reference
find_similar() {
    local reference="$1"
    echo "Finding documents similar to: $reference"
    
    oboyu query "similar to $reference" \
        --mode vector \
        --limit 10 \
        --exclude "$reference"
}

# Find documents by topic cluster
find_cluster() {
    local topic="$1"
    echo "Finding $topic cluster..."
    
    oboyu query "$topic" \
        --mode hybrid \
        --expand-query \
        --group-by similarity \
        --limit 20
}

# Usage
find_similar "architecture-design.md"
find_cluster "machine learning"
```

## Advanced CLI Patterns

### Batch Processing

```bash
#!/bin/bash
# batch-search.sh - Process multiple searches

# Read queries from file
while IFS= read -r query; do
    echo "Searching: $query"
    oboyu query "$query" \
        --limit 5 \
        --format json \
        >> results.jsonl
done < queries.txt

# Process search results
jq -s 'group_by(.query) | map({query: .[0].query, count: length, top_score: (map(.score) | max)})' results.jsonl
```

### Pipeline Integration

```bash
# Search and process results
oboyu query "configuration" --format json | \
    jq -r '.[] | select(.score > 0.8) | .path' | \
    xargs grep -l "database"

# Find and backup important documents
oboyu query "important confidential" --format paths | \
    tar -czf important-docs-$(date +%Y%m%d).tar.gz -T -

# Search and open in editor
oboyu query "TODO fix" --limit 1 --format path | \
    xargs -I {} code {}
```

### Interactive Search Session

```bash
#!/bin/bash
# interactive-search.sh - Enhanced interactive search

# Setup
HISTORY_FILE=~/.oboyu_search_history
touch "$HISTORY_FILE"

# Search with history
search_with_history() {
    local query="$1"
    echo "$query" >> "$HISTORY_FILE"
    
    oboyu query "$query" \
        --interactive \
        --save-session \
        --format rich
}

# Interactive loop with readline
while IFS= read -e -p "oboyu> " -i "$last_query" query; do
    [[ -z "$query" ]] && continue
    [[ "$query" == "exit" ]] && break
    
    case "$query" in
        "history")
            tail -20 "$HISTORY_FILE"
            ;;
        "clear")
            clear
            ;;
        "help")
            echo "Commands: history, clear, help, exit"
            echo "Search: any other text"
            ;;
        *)
            search_with_history "$query"
            last_query="$query"
            ;;
    esac
done

# Save readline history
history -w ~/.oboyu_readline_history
```

## Automation Scripts

### Index Maintenance

```bash
#!/bin/bash
# maintain-index.sh - Automated index maintenance

LOG_FILE=~/.oboyu/maintenance.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Daily maintenance
daily_maintenance() {
    log "Starting daily maintenance"
    
    # Update indices
    for index in $(oboyu index list --format names); do
        log "Updating index: $index"
        oboyu index update --db-path ~/indexes/example.db --quiet
    done
    
    # Clean old entries
    log "Cleaning old entries"
    oboyu index clean --older-than 90d
    
    # Optimize indices
    log "Optimizing indices"
    oboyu index optimize --all
    
    log "Daily maintenance complete"
}

# Weekly maintenance
weekly_maintenance() {
    log "Starting weekly maintenance"
    
    # Full integrity check
    oboyu index verify --all --fix
    
    # Generate statistics
    oboyu index stats --all > ~/.oboyu/weekly-stats.txt
    
    # Backup indices
    oboyu index backup --all --destination ~/.oboyu/backups/
    
    log "Weekly maintenance complete"
}

# Run appropriate maintenance
case "$1" in
    "daily")
        daily_maintenance
        ;;
    "weekly")
        weekly_maintenance
        ;;
    *)
        echo "Usage: $0 {daily|weekly}"
        exit 1
        ;;
esac
```

### Search Analytics

```bash
#!/bin/bash
# search-analytics.sh - Analyze search patterns

# Collect search metrics
analyze_searches() {
    local days="${1:-30}"
    
    echo "=== Search Analytics (Last $days days) ==="
    
    # Top searches
    echo -e "\n📊 Top Searches:"
    oboyu query history \
        --days "$days" \
        --format json | \
        jq -r '.[] | .query' | \
        sort | uniq -c | sort -rn | head -10
    
    # Search performance
    echo -e "\n⚡ Performance Metrics:"
    oboyu query history \
        --days "$days" \
        --format json | \
        jq -r 'map(.duration_ms) | {
            avg: (add/length),
            min: min,
            max: max,
            p95: (sort | .[length * 0.95 | floor])
        }'
    
    # Zero result queries
    echo -e "\n❌ Queries with No Results:"
    oboyu query history \
        --days "$days" \
        --no-results \
        --format list
}

# Generate report
generate_report() {
    local output="search-report-$(date +%Y%m%d).html"
    
    {
        echo "<html><head><title>Search Analytics Report</title></head><body>"
        echo "<h1>Oboyu Search Analytics</h1>"
        echo "<pre>"
        analyze_searches 30
        echo "</pre>"
        echo "</body></html>"
    } > "$output"
    
    echo "Report generated: $output"
}

# Run analytics
analyze_searches "${1:-30}"
[[ "$2" == "--report" ]] && generate_report
```

## Shell Integration

### Bash Aliases

```bash
# Add to ~/.bashrc or ~/.bash_aliases

# Quick search
alias q='oboyu query'
alias qs='oboyu query --mode semantic'
alias qk='oboyu query --mode bm25'

# Index management
alias oidx='oboyu index'
alias oidx-update='oboyu index update --all'
alias oidx-stats='oboyu index stats'

# Common searches
alias qtodo='oboyu query "TODO OR FIXME" --days 7'
alias qmeeting='oboyu query "meeting" --days 7'
alias qrecent='oboyu query "*" --days 1 --sort date'

# Project-specific
alias qwork='oboyu query --db-path ~/indexes/work.db'
alias qpersonal='oboyu query --db-path ~/indexes/personal.db'
```

### Zsh Functions

```zsh
# Add to ~/.zshrc

# Fuzzy search with fzf
osearch() {
    local query="$*"
    local selected=$(
        oboyu query "$query" --format json | \
        jq -r '.[] | "\(.score|tostring[0:4]) \(.path)"' | \
        fzf --preview 'echo {} | cut -d" " -f2- | xargs oboyu show'
    )
    
    [[ -n "$selected" ]] && echo "$selected" | cut -d" " -f2- | xargs open
}

# Search and edit
oedit() {
    local result=$(oboyu query "$*" --limit 1 --format path)
    [[ -n "$result" ]] && ${EDITOR:-vim} "$result"
}

# Index with progress
oindex() {
    oboyu index "$@" | \
    pv -l -s $(find "$1" -type f | wc -l) > /dev/null
}
```

### Fish Shell

```fish
# Add to ~/.config/fish/config.fish

# Abbreviations
abbr -a oq 'oboyu query'
abbr -a oi 'oboyu index'
abbr -a oc 'oboyu config'

# Functions
function osearch-interactive
    set query (read -P "Search: ")
    oboyu query $query --interactive
end

function oindex-watch
    watch -n 60 "oboyu index $argv --update"
end
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/index-docs.yml
name: Index Documentation

on:
  push:
    paths:
      - 'docs/**'
      - '*.md'

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Oboyu
        run: pip install oboyu
      
      - name: Index documentation
        run: |
          oboyu index ./docs --db-path ~/indexes/docs.db
          oboyu index ./*.md --db-path ~/indexes/readme.db --update
      
      - name: Test search
        run: |
          oboyu query "installation" --db-path ~/indexes/docs.db
          oboyu query "contributing" --db-path ~/indexes/readme.db
      
      - name: Upload index
        uses: actions/upload-artifact@v3
        with:
          name: search-index
          path: ~/.oboyu/*.db
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: update-search-index
        name: Update search index
        entry: bash -c 'oboyu index . --update --quiet'
        language: system
        pass_filenames: false
        always_run: true
```

## Error Handling

### Robust Scripts

```bash
#!/bin/bash
# robust-search.sh - Search with error handling

set -euo pipefail

# Error handler
error_handler() {
    echo "Error: Command failed at line $1"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Safe search function
safe_search() {
    local query="$1"
    local max_retries=3
    local retry=0
    
    while [[ $retry -lt $max_retries ]]; do
        if oboyu query "$query" --timeout 30; then
            return 0
        else
            echo "Search failed, retry $((retry + 1))/$max_retries"
            ((retry++))
            sleep 2
        fi
    done
    
    echo "Search failed after $max_retries attempts"
    return 1
}

# Usage with fallback
if ! safe_search "$1"; then
    echo "Falling back to basic search"
    oboyu query "$1" --mode bm25 --limit 5
fi
```

## Performance Scripts

### Parallel Search

```bash
#!/bin/bash
# parallel-search.sh - Search multiple indices in parallel

# Search function
search_index() {
    local index="$1"
    local query="$2"
    echo "[$index]"
    oboyu query "$query" --db-path ~/indexes/example.db --limit 3
}

export -f search_index

# Parallel execution
query="$1"
oboyu index list --format names | \
    parallel -j 4 search_index {} "$query"
```

### Benchmark Script

```bash
#!/bin/bash
# benchmark-search.sh - Measure search performance

benchmark_mode() {
    local mode="$1"
    local query="$2"
    local iterations=10
    
    echo "Benchmarking $mode mode..."
    
    total=0
    for i in $(seq 1 $iterations); do
        duration=$(oboyu query "$query" --mode "$mode" --format json | \
                  jq -r '.[0].search_duration_ms')
        total=$((total + duration))
    done
    
    avg=$((total / iterations))
    echo "$mode: ${avg}ms average"
}

# Run benchmarks
query="${1:-machine learning}"

for mode in bm25 vector hybrid; do
    benchmark_mode "$mode" "$query"
done
```

## Next Steps

- Set up [Automation](automation.md) for scheduled tasks
- Explore [MCP Integration](mcp-integration.md) for AI assistance
- Review [Configuration](../configuration-optimization/configuration.md) for optimization