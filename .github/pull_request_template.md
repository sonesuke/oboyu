# Pull Request

## Summary

**Brief description of the changes:**

**Motivation for these changes:**

## Type of Change

Please check all that apply:

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test improvements
- [ ] Configuration changes

## Related Issues

- Fixes # (issue number)
- Related to # (issue number)

## Changes Made

**Detailed list of changes:**
- 
- 

**New files added/removed:**
- 
- 

**Modified functionality:**
- 
- 

## Testing

- [ ] I have tested my changes locally
- [ ] All existing tests pass (`uv run pytest -m "not slow" -k "not integration"`)
- [ ] I have added tests for new functionality
- [ ] I have updated documentation as needed
- [ ] Tests pass with coverage (`uv run pytest --cov=src`)

## Documentation

- [ ] Documentation has been updated (if applicable)
- [ ] Code comments have been added/updated
- [ ] README updated (if applicable)
- [ ] Architecture documentation updated (if applicable)

## Code Quality

- [ ] Code has been linted and formatted (`uv run ruff check --fix`)
- [ ] Type checking passes (`uv run mypy`)
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] My changes generate no new warnings

## Performance Considerations

**Any performance impacts:**

**Memory usage considerations:**

**Search index size changes:**

**Benchmark results (if applicable):**

## Japanese Language Considerations

Since oboyu has specialized Japanese language support, please verify:

- [ ] Changes maintain Japanese text processing capability
- [ ] Encoding detection still works properly
- [ ] Japanese tokenization is not affected
- [ ] No regression in Japanese search quality
- [ ] Japanese documentation updated (if applicable)

## Checklist

- [ ] Any dependent changes have been merged
- [ ] I have made corresponding changes to the documentation
- [ ] Pre-commit hooks pass (do not use `--no-verify`)
- [ ] All CI checks are passing
- [ ] I have reviewed the [Contributing Guidelines](CONTRIBUTING.md)

## Additional Notes

**Screenshots (if applicable):**

**Special deployment considerations:**

**Breaking changes that require version updates:**

**Additional context:**

---

<!-- 
Guidelines for contributors:
- Keep your PR focused and atomic - one feature/fix per PR
- Write clear commit messages following conventional commits
- Update tests and documentation
- Consider Japanese language users and functionality
- Run all code quality checks before submitting
-->