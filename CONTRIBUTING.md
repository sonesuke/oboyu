# Contributing to Oboyu

First off, thank you for considering contributing to Oboyu! It's people like you that make Oboyu such a great tool for Japanese semantic search.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [What we are looking for](#what-we-are-looking-for)
- [How to contribute](#how-to-contribute)
- [Development Process](#development-process)
- [Style Guides](#style-guides)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to [iamsonesuke@gmail.com](mailto:iamsonesuke@gmail.com).

## What we are looking for

Oboyu is an open source project and we love to receive contributions from our community! There are many ways to contribute:

### Types of Contributions

#### 🐛 Report Bugs

Report bugs at https://github.com/sonesuke/oboyu/issues.

If you are reporting a bug, please include:

* Your operating system name and version
* Python version
* Detailed steps to reproduce the bug
* Any error messages or stack traces
* What you expected to happen
* What actually happened

#### 🔧 Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### ✨ Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

#### 📝 Write Documentation

Oboyu could always use more documentation, whether as part of the official Oboyu docs, in docstrings, or even on the web in blog posts, articles, and such.

#### 💡 Submit Feedback

The best way to send feedback is to file an issue at https://github.com/sonesuke/oboyu/issues.

If you are proposing a feature:

* Explain in detail how it would work
* Keep the scope as narrow as possible, to make it easier to implement
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

## How to contribute

### First Time Contributors

Unsure where to begin contributing to Oboyu? You can start by looking through these `beginner` and `help-wanted` issues:

* [Beginner issues](https://github.com/sonesuke/oboyu/labels/beginner) - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues](https://github.com/sonesuke/oboyu/labels/help%20wanted) - issues which should be a bit more involved than beginner issues.

### Pull Request Process

1. **Fork the repo** and create your branch from `main`.

2. **Set up your development environment**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/oboyu.git
   cd oboyu
   uv sync
   ```

3. **Make your changes** in a new git branch:
   ```bash
   git checkout -b name-of-your-bugfix-or-feature
   ```

4. **Make sure your code passes all tests and style checks**:
   ```bash
   # Run fast tests during development
   uv run pytest -m "not slow" -k "not integration"
   
   # Run all tests before submitting
   uv run pytest --cov=src
   
   # Check code style
   uv run ruff check --fix
   
   # Type checking
   uv run mypy
   ```

5. **Add tests** for your changes. Make sure your changes are covered by tests and that all tests pass.

6. **Update documentation** if you are changing behavior or adding features.

7. **Commit your changes** using a descriptive commit message that follows our [commit message conventions](#commit-messages).

8. **Push your branch** to GitHub:
   ```bash
   git push origin name-of-your-bugfix-or-feature
   ```

9. **Submit a pull request** through the GitHub website.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for Python 3.13+ on Linux and macOS.
4. Check that all tests pass in the GitHub Actions CI.

## Development Process

### Development Workflow

#### Code Quality

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

#### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- **Never use `--no-verify`** when committing
- Always fix lint errors and test failures before committing
- Pre-commit hooks maintain code quality automatically

### Testing

#### Test Categories

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

#### Writing Tests

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

### Debugging

#### Logging

Enable debug logging for specific components:

```python
import logging

# Enable debug logging for specific components
logging.getLogger("oboyu.indexer").setLevel(logging.DEBUG)
logging.getLogger("oboyu.crawler").setLevel(logging.INFO)
```

#### Common Debug Scenarios

1. **Indexing Issues**:
   ```bash
   # Run with verbose output
   oboyu index ./docs --debug
   ```

2. **Search Issues**:
   ```bash
   # Interactive mode for testing
   oboyu query --interactive --debug
   ```

3. **MCP Server Issues**:
   ```bash
   # Run with debug logging
   oboyu mcp --debug
   ```

## Style Guides

### Python Code Style

- **Follow PEP 8** with Ruff enforcement
- **Use type hints** for all function signatures
- **Write docstrings** for all public functions and classes
- **Keep functions focused** and maintain single responsibility
- **Use descriptive variable names**
- **Maximum line length**: 100 characters
- **Use f-strings** for string formatting

Example:
```python
def process_japanese_text(
    text: str, 
    encoding: str = "utf-8"
) -> str:
    """Process Japanese text with proper normalization.
    
    Args:
        text: Input text to process
        encoding: Text encoding (default: utf-8)
        
    Returns:
        Normalized Japanese text
        
    Raises:
        ValueError: If text cannot be decoded
    """
    # Implementation
    pass
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

body (optional)

footer (optional)
```

#### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

#### Examples

```
feat(indexer): add support for PDF document processing

- Implement PDF text extraction using pypdf
- Add tests for PDF processing
- Update documentation

Closes #123
```

```
fix(cli): resolve issue with Japanese encoding detection

The crawler was failing to detect Shift-JIS encoding in some cases.
This fix adds additional heuristics for encoding detection.
```

### Documentation Style

- Use **Markdown** for all documentation
- Include **code examples** where appropriate
- Keep documentation **up to date** with code changes
- Use **clear and concise** language
- Include **Japanese examples** where relevant

### Documentation Site

The documentation site is built with [Docusaurus](https://docusaurus.io/) and hosted on GitHub Pages.

#### Local Development

```bash
# Navigate to website directory
cd website

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Test production build locally
npm run serve
```

#### Adding Documentation

1. Add new markdown files to `website/docs/`
2. Update `website/sidebars.ts` to include new pages
3. Follow the existing frontmatter format:
   ```markdown
   ---
   id: page-id
   title: Page Title
   sidebar_position: 10
   ---
   ```

#### Deployment

Documentation is automatically deployed when changes are pushed to the `main` branch via GitHub Actions.

## Code Analysis

### Using Repomix

When working on larger changes, use Repomix for codebase analysis:

```bash
# Install repomix globally
npm install -g repomix

# Generate codebase analysis
repomix

# Use with Claude Code for AI-assisted development
# The generated repomix-output.xml provides comprehensive context
```

When using Claude Code, include the compress option to reduce token usage:
```bash
# In Claude Code
mcp__repomix__pack_codebase(directory="/path/to/oboyu", compress=true)
```

## Security

### Reporting Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them to [iamsonesuke@gmail.com](mailto:iamsonesuke@gmail.com). You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the requested information listed below (as much as you can provide) to help us better understand the nature and scope of the possible issue:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

## Community

### Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Search [existing issues](https://github.com/sonesuke/oboyu/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/sonesuke/oboyu/discussions) for questions

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Pull Requests**: For code contributions

## Recognition

Contributors who submit accepted pull requests will be added to the [AUTHORS](AUTHORS.md) file.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions to open source, large or small, make projects like Oboyu possible. Thank you for taking the time to contribute.