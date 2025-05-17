## Development Workflow

- **Format and lint code**:
  ```
  uv run ruff check --fix
  ```

- **Run tests with coverage**:
  ```
  uv run pytest --cov=src
  ```

- **Type checking**:
  ```
  uv run mypy
  ```

## Using UV for Dependency Management

[UV](https://github.com/astral-sh/uv) is a fast, reliable Python package installer and resolver that significantly speeds up dependency management:

- **Install dependencies**: `uv sync`
- **Add a new dependency**: `uv add package-name`

## Using Repomix for Code Analysis

[Repomix](https://github.com/yamadashy/repomix) is a tool to analyze your codebase and generate a comprehensive overview.

### Setup

The project comes with a pre-configured `repomix.config.json` file that:
- Includes all Python files, Markdown documents, and key configuration files
- Respects `.gitignore` patterns
- Enables compression for more efficient analysis
- Checks for security issues

### Generating Code Analysis

To generate a code analysis file:

```bash
# Install repomix
npm install -g repomix

# Generate analysis (creates repomix-output.xml)
repomix
```

### Working with Claude Code

The generated analysis is particularly useful with Claude Code:

1. Use `repomix` to generate the analysis file
2. Open the project with Claude Code
3. Claude will analyze the `repomix-output.xml` to better understand the codebase

For more details, see the [Repomix documentation](https://github.com/yamadashy/repomix).