
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

