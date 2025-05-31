# Development Guide

This guide provides comprehensive information for developers working on Oboyu.

## Development Workflow

### Code Quality

Oboyu uses several tools to maintain high code quality:

```bash
# Format and lint code
uv run ruff check --fix

# Type checking
uv run mypy

# Run tests with coverage
uv run pytest --cov=src

# Run fast tests (excluding slow integration tests)
uv run pytest -m "not slow" -k "not integration"
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- **Never use `--no-verify`** when committing
- Always fix lint errors and test failures before committing
- Pre-commit hooks maintain code quality automatically

## Dependency Management with UV

[UV](https://github.com/astral-sh/uv) is used for fast, reliable Python package management:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv sync --upgrade
```

### Why UV?

- **Speed**: 10-100x faster than pip
- **Reliability**: Consistent dependency resolution
- **Modern**: Built-in virtual environment management
- **Compatibility**: Drop-in replacement for pip

## Testing

### Test Categories

Oboyu has different test categories for efficient development:

```bash
# Fast tests (recommended for development)
uv run pytest -m "not slow" -k "not integration"

# All tests with coverage
uv run pytest --cov=src

# Only slow tests (ML model loading, large datasets)
uv run pytest -m "slow"

# Only integration tests
uv run pytest -k "integration"

# Specific test file
uv run pytest tests/test_file.py::test_function
```

### Test Structure

```
tests/
├── cli/              # CLI command tests
├── common/           # Common utilities tests
├── crawler/          # Document crawling tests
├── indexer/          # Indexing and search tests
├── integration/      # End-to-end integration tests
└── mcp/              # MCP server tests
```

### Writing Tests

Follow these guidelines when writing tests:

1. **Use descriptive test names** that explain what is being tested
2. **Mark slow tests** with `@pytest.mark.slow` decorator
3. **Use fixtures** for common test data and setup
4. **Test edge cases** and error conditions
5. **Keep tests focused** on a single functionality

Example:
```python
import pytest
from oboyu.indexer.indexer import Indexer

@pytest.mark.slow
def test_indexer_with_japanese_content():
    """Test that indexer properly handles Japanese text with MeCab tokenization."""
    # Test implementation
    pass

def test_indexer_configuration_validation():
    """Test that indexer validates configuration parameters correctly."""
    # Test implementation  
    pass
```

## Architecture Overview

### Component Structure

Oboyu follows a modular architecture with clear separation of concerns:

```
src/oboyu/
├── cli/              # Command-line interface
├── common/           # Shared utilities and configuration
├── crawler/          # Document discovery and extraction
├── indexer/          # Core indexing and search functionality
└── mcp/              # Model Context Protocol server
```

### Key Design Principles

1. **Modularity**: Each component has a well-defined responsibility
2. **Testability**: Components are designed for easy unit testing
3. **Extensibility**: New features can be added without breaking existing functionality
4. **Performance**: Optimized for Japanese text processing and large document collections
5. **Type Safety**: Comprehensive type hints throughout the codebase

## Development Environment Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Git for version control
git --version
```

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sonesuke/oboyu.git
   cd oboyu
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Install development dependencies**:
   ```bash
   uv sync --dev
   ```

4. **Run tests to verify setup**:
   ```bash
   uv run pytest -m "not slow"
   ```

## Code Analysis with Repomix

[Repomix](https://github.com/yamadashy/repomix) is used for comprehensive codebase analysis.

### Configuration

The project includes a pre-configured `repomix.config.json` that:
- Includes all Python files, Markdown docs, and key configuration files
- Respects `.gitignore` patterns
- Enables compression for efficient analysis
- Performs security checks

### Usage

```bash
# Install repomix globally
npm install -g repomix

# Generate codebase analysis
repomix

# Use with Claude Code for AI-assisted development
# The generated repomix-output.xml provides comprehensive context
```

### Benefits

- **AI Assistance**: Provides context for Claude Code and other AI tools
- **Code Review**: Comprehensive overview for code reviews
- **Documentation**: Automatic documentation of codebase structure
- **Onboarding**: Helps new developers understand the codebase

## Debugging

### Logging

Oboyu uses Python's standard logging module with different levels:

```python
import logging

# Enable debug logging for specific components
logging.getLogger("oboyu.indexer").setLevel(logging.DEBUG)
logging.getLogger("oboyu.crawler").setLevel(logging.INFO)

# Enable verbose CLI output
oboyu query "test" --verbose
```

### Common Debug Scenarios

1. **Indexing Issues**:
   ```bash
   # Run with verbose output
   oboyu index ./docs --verbose
   
   # Check specific file processing
   oboyu index ./docs --include-patterns "specific_file.txt" --verbose
   ```

2. **Search Issues**:
   ```bash
   # Interactive mode for testing
   oboyu query --interactive --verbose
   
   # Test different search modes
   oboyu query "test" --mode vector --explain
   ```

3. **MCP Server Issues**:
   ```bash
   # Run with debug logging
   oboyu mcp --verbose --debug
   ```

## Contributing Guidelines

### Code Style

- **Follow PEP 8** with Ruff enforcement
- **Use type hints** for all function signatures
- **Write docstrings** for public functions and classes
- **Keep functions focused** and maintain single responsibility
- **Use descriptive variable names**

### Commit Message Format

Follow conventional commit format:

```
type(scope): description

body (optional)

footer (optional)
```

Examples:
```
feat(indexer): add support for PDF document processing
fix(cli): resolve issue with Japanese encoding detection
docs(readme): update installation instructions
test(indexer): add tests for incremental indexing
```

### Pull Request Process

1. **Create a feature branch** from main
2. **Make focused changes** with clear commit messages
3. **Add or update tests** for new functionality
4. **Update documentation** if needed
5. **Ensure all tests pass** and code quality checks succeed
6. **Create a pull request** with clear description

### Review Criteria

Pull requests are reviewed for:

- **Functionality**: Does the code work as intended?
- **Test Coverage**: Are there adequate tests?
- **Code Quality**: Does it follow project standards?
- **Documentation**: Is it properly documented?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?

## Performance Optimization

### Profiling

Use Python's built-in profiling tools:

```bash
# Profile indexing performance
python -m cProfile -o indexing.prof -c "from oboyu.cli.main import app; app()"

# Analyze profile results
python -c "import pstats; pstats.Stats('indexing.prof').sort_stats('tottime').print_stats(20)"
```

### Memory Optimization

For large document collections:

```python
# Use smaller batch sizes
indexer_config = {
    "batch_size": 4,  # Reduce from default 8
    "chunk_size": 512,  # Smaller chunks for memory efficiency
}

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### Japanese Text Processing

Optimization tips for Japanese content:

- **Use MeCab** for accurate tokenization (vs. simple character splitting)
- **Enable ONNX optimization** for embedding models (2-4x speedup)
- **Tune chunk sizes** for Japanese text density (typically smaller than English)
- **Consider reranker trade-offs** (accuracy vs. speed)

## Release Process

### Version Management

Oboyu follows semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `src/oboyu/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite** including slow tests
4. **Verify benchmark performance** hasn't regressed
5. **Create release tag** and GitHub release
6. **Publish to PyPI** (automated via CI/CD)

## Troubleshooting

### Common Development Issues

1. **Import Errors**:
   ```bash
   # Ensure oboyu is installed in development mode
   uv pip install -e .
   ```

2. **Test Failures**:
   ```bash
   # Check for missing test dependencies
   uv sync --dev
   
   # Run tests in verbose mode
   uv run pytest -v
   ```

3. **Type Check Failures**:
   ```bash
   # Run mypy with detailed output
   uv run mypy --show-error-codes
   ```

4. **MeCab Issues** (Japanese tokenization):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install mecab mecab-ipadic-utf8
   
   # On macOS
   brew install mecab mecab-ipadic
   ```

### Getting Help

- **Documentation**: Check relevant docs in `docs/` directory
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request review from maintainers

## Future Development

### Planned Features

- [ ] Support for additional document formats (PDF, DOCX)
- [ ] Distributed indexing for large document collections
- [ ] Real-time document synchronization
- [ ] Advanced Japanese linguistic features
- [ ] Custom embedding model fine-tuning
- [ ] REST API server mode

### Architecture Evolution

The codebase is designed to support future enhancements:

- **Plugin System**: For custom document processors
- **Distributed Architecture**: For scaling to larger deployments
- **Model Registry**: For managing multiple embedding models
- **Advanced Analytics**: For search quality metrics

For the latest development roadmap, see the project's GitHub issues and milestones.