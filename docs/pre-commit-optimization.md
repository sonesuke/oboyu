# Pre-commit Hook Performance Optimization Guide

## Overview

This guide documents the optimizations made to improve pre-commit hook performance, reducing commit times from 30-60+ seconds to under 10 seconds for typical commits.

## Problem Analysis

The original pre-commit configuration had several performance bottlenecks:

1. **Ruff**: Ran on entire codebase regardless of changes
2. **MyPy**: Full type checking without caching
3. **Pytest**: Ran all tests (excluding slow/integration) even for small changes
4. **Pylint**: Checked all files instead of just changed ones

## Implemented Optimizations

### 1. Incremental File Processing

All hooks now process only changed files:
- `pass_filenames: true` - Pass changed files to tools
- `types: [python]` - Only run on Python files
- Removed `always_run: true` flags

### 2. MyPy Caching

Enabled incremental mode with persistent cache:
```yaml
entry: uv run mypy --incremental --cache-dir=.mypy_cache
```

### 3. Smart Test Runner

Created `.claude/scripts/run-affected-tests.sh` that:
- Detects changed Python files
- Maps source files to their test files
- Only runs relevant tests
- Skips testing entirely if no Python files changed

### 4. Focused Tool Execution

Each tool now:
- Only processes files relevant to its purpose
- Skips execution when no relevant files exist
- Uses caching where available

## Performance Comparison

| Scenario | Before | After |
|----------|--------|-------|
| Single file change | 30-60s | <5s |
| Documentation only | 30-60s | <1s |
| Multiple file changes | 60s+ | 10-15s |
| No Python changes | 30-60s | <1s |

## Usage Guidelines

### For Developers

1. **Normal commits**: Just use `git commit` as usual - hooks are now fast
2. **Large changes**: For 20+ file changes, consider:
   ```bash
   # Run checks manually first
   uv run ruff check --fix
   uv run mypy
   uv run pytest -m "not slow"
   
   # Then commit with hooks disabled if needed
   git commit --no-verify
   ```

3. **Debugging slow commits**:
   ```bash
   # Check what files are staged
   git status
   
   # Run hooks manually to see timing
   pre-commit run --all-files
   ```

### CI/CD Considerations

The full test suite and comprehensive checks still run in CI/CD:
- GitHub Actions run all tests including slow/integration
- Full codebase linting and type checking
- Coverage analysis

This ensures code quality while keeping local development fast.

## Troubleshooting

### MyPy still slow?
- Ensure `.mypy_cache/` exists
- Try clearing cache: `rm -rf .mypy_cache/`
- Check if many type stubs need downloading

### Tests running when they shouldn't?
- Check the test detection logic in `.claude/scripts/run-affected-tests.sh`
- Ensure test files follow naming convention: `test_*.py`

### Hooks not running?
- Verify pre-commit is installed: `pre-commit install`
- Check hook configuration: `pre-commit run --all-files --verbose`

## Future Improvements

1. **Parallel execution**: Run independent hooks concurrently
2. **Better test mapping**: Use AST analysis to find truly affected tests
3. **Distributed caching**: Share caches across team members
4. **Progressive testing**: Run most-likely-to-fail tests first

## Configuration Reference

The optimized configuration in `.pre-commit-config.yaml`:
- Uses local hooks for better control
- Passes filenames to all tools
- Filters by file types
- Leverages tool-specific optimizations

## Rollback Instructions

If issues arise, revert to the original configuration:
```bash
git checkout main -- .pre-commit-config.yaml
rm -rf .claude/scripts/run-affected-tests.sh
pre-commit install
```