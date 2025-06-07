# E2E Display Testing with Claude Code SDK

## Overview

Oboyu now includes comprehensive end-to-end (E2E) display testing using the Claude Code SDK. This testing framework automatically verifies that CLI output formatting, progress displays, and user interfaces are working correctly from a user's perspective.

## Features

The E2E display testing framework covers:

- **CLI Output Formatting**: Verifies help text, version info, and command outputs
- **Progress Display**: Tests HierarchicalLogger and progress bar rendering
- **Search Results**: Validates text and JSON output formats, snippet highlighting
- **Japanese Text**: Ensures proper rendering of Japanese characters
- **Error Messages**: Checks that errors are clear and user-friendly
- **Interactive Mode**: Tests the interactive query interface

## Prerequisites

1. **Claude Code SDK**: Install the SDK globally:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **API Key**: Set your Anthropic API key (optional - will prompt if needed):
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

3. **Oboyu**: Ensure Oboyu is installed and accessible:
   ```bash
   uv sync
   ```

## Running Tests

### Quick Start

Run all E2E display tests:

```bash
uv run python e2e/run_tests.py
```

### Running Specific Tests

You can run individual test categories:

```bash
# Run only basic CLI display tests
uv run python e2e/run_tests.py --test basic

# Run only search result display tests
uv run python e2e/run_tests.py --test search

# Available test options:
# - basic: Basic CLI commands (help, version, health)
# - indexing: Indexing progress display
# - search: Search result formatting
# - error: Error message display
```

### Advanced Options

```bash
# Use a custom oboyu command path
uv run python e2e/run_tests.py --oboyu-path /path/to/oboyu

# Save report to a different location
uv run python e2e/run_tests.py --report my_report.md

# Keep test data after running (for debugging)
uv run python e2e/run_tests.py --no-cleanup
```

## Test Implementation

The E2E tests are implemented as standalone Python scripts in the `e2e/` directory. The main test runner is `run_tests.py` which uses the `OboyuE2EDisplayTester` class from `display_tester.py`.

## Test Structure

### Test Categories

1. **Basic CLI Display** (`test_basic_cli_display`)
   - Tests `--help` output formatting
   - Verifies version display
   - Validates JSON output format

2. **Indexing Progress Display** (`test_indexing_progress_display`)
   - Creates temporary test files
   - Monitors HierarchicalLogger output
   - Checks progress bars and completion messages

3. **Search Result Display** (`test_search_result_display`)
   - Tests both text and JSON output formats
   - Verifies snippet highlighting
   - Checks Japanese text rendering

4. **Error Display** (`test_error_display`)
   - Tests various error scenarios
   - Verifies error message clarity
   - Checks formatting consistency


### Test Implementation

The main test class `OboyuE2EDisplayTester` uses Claude Code SDK in headless mode:

```python
def run_claude_check(self, prompt: str) -> dict[str, Any]:
    """Execute Claude Code to check display issues."""
    # Runs claude with -p flag for non-interactive mode
    # Uses --output-format json for structured results
```

## Test Reports

After running tests, a comprehensive Markdown report is generated containing:

- Test results for each category
- Specific issues identified
- UI/UX improvement suggestions
- Metadata (API costs, execution time)

Example report structure:

```markdown
# Oboyu E2E Display Test Report

## Summary
- Total tests run: 6
- Test environment: /path/to/oboyu
- Oboyu command: oboyu

## Test Results

### Basic CLI Display
[Claude Code's analysis of CLI display quality]

### Indexing Progress Display
[Analysis of progress display functionality]

...

## Metadata
- Total cost: $0.0234
- Total duration: 15234ms
- Total turns: 12
```

## Integration with CI/CD

You can integrate E2E display testing into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run E2E Display Tests
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    npm install -g @anthropic-ai/claude-code
    uv run python e2e/run_tests.py --test basic
```

## Cost Considerations

- Each test run consumes API tokens based on the complexity of checks
- The framework reports total API costs in the test report
- Consider running subset of tests during development
- Use comprehensive tests for release validation

## Troubleshooting

### Common Issues

1. **Claude Code not found**
   - Ensure Claude Code SDK is installed globally
   - Check PATH includes npm global bin directory

2. **API Key errors**
   - Verify ANTHROPIC_API_KEY is set correctly
   - Check API key has sufficient credits

3. **Test failures**
   - Review the generated report for specific issues
   - Use `--no-cleanup` to inspect test data
   - Check oboyu command path is correct

### Debug Mode

For debugging, you can run the test module directly:

```bash
# Run with Python directly for debugging
cd e2e
uv run python run_tests.py --test basic --no-cleanup
```

## Best Practices

1. **Regular Testing**: Run E2E display tests before releases
2. **Selective Testing**: Use specific test categories during development
3. **Report Review**: Always review generated reports for UI/UX insights
4. **Cost Management**: Monitor API costs and adjust test frequency accordingly
5. **Continuous Improvement**: Update tests when adding new display features