# E2E Display Testing

This directory contains end-to-end display tests for Oboyu using the Claude Code SDK.

## Prerequisites

1. Install Claude Code SDK:
```bash
npm install -g @anthropic-ai/claude-code
```

2. Set your API key (optional - will prompt if needed):
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Running Tests

### Quick Start
```bash
uv run python e2e/run_tests.py
```

### Run Specific Tests
```bash
# Basic CLI display
uv run python e2e/run_tests.py --test basic

# Indexing progress
uv run python e2e/run_tests.py --test indexing

# Search results
uv run python e2e/run_tests.py --test search

# Error display
uv run python e2e/run_tests.py --test error
```

### Advanced Options
```bash
# Use custom oboyu command path
uv run python e2e/run_tests.py --oboyu-path /path/to/oboyu

# Save report to different location
uv run python e2e/run_tests.py --report my_report.md

# Keep test data for debugging
uv run python e2e/run_tests.py --no-cleanup
```

## Test Categories

1. **basic** - Basic CLI commands (help, version, health)
2. **indexing** - Indexing progress display
3. **search** - Search result formatting
4. **error** - Error message display

## How It Works

1. **Execute Commands**: Each test runs actual Oboyu commands and captures their output
2. **Claude Analysis**: Claude Code SDK analyzes the captured output for display issues
3. **Problem Detection**: Issues like formatting problems or broken features are flagged as test failures
4. **Detailed Reports**: Results are saved as Markdown reports with specific improvement suggestions

## Example Output

```
Running E2E display tests with oboyu command: oboyu
Test selection: all
------------------------------------------------------------

Running basic_cli_display...
✓ basic_cli_display completed

Running search_result_display...
✗ search_result_display failed: Display issues detected

============================================================
TEST SUMMARY
============================================================
Total tests: 2
Passed: 1
Failed: 1

Failed tests:
  - search_result_display: Display issues detected

Total API cost: $0.1234
```

## Notes

- Tests use Claude Code in non-interactive mode (`-p` flag)
- Real-time streaming shows Claude Code analysis progress
- API costs are tracked and reported
- Test data is automatically cleaned up unless `--no-cleanup` is used
- Test reports (`e2e_display_report.md`) are generated locally and not committed to git