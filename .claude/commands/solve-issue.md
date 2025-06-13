# Solve Issue Workflow

This command implements an enhanced GitHub issue resolution workflow with early PR creation and progress tracking.

## 🎯 Workflow Overview

1. **Extract Issue Information**
   - Extract issue number from branch name
   - Fetch issue details from GitHub using `gh issue view {issue_number}`
   - Understand the problem, requirements, and acceptance criteria

2. **Initial Setup & Planning**
   - Create a comprehensive todo list for the implementation
   - Break down the issue into manageable tasks
   - Plan the implementation approach

3. **Development with Early PR Creation**
   - Start implementation of the solution
   - After first meaningful commit, create a **Draft PR** immediately
   - Use WIP title format: `WIP: Fix #{issue_number} - {issue_title}`
   - Link the issue in PR description with `Fixes #{issue_number}` or `Closes #{issue_number}`
   - Add comment to original issue: `Working on this in PR #{pr_number}`

4. **Progress Tracking**
   - Update PR title to reflect development progress:
     - `WIP: Fix #{issue_number} - [30%] Initial setup complete`
     - `WIP: Fix #{issue_number} - [70%] Core functionality implemented`
     - `Ready: Fix #{issue_number} - {final_title}`
   - Make regular commits with descriptive messages
   - Update PR description with implementation details and progress

5. **Quality Gates & Completion**
   - Run linting and type checking: `uv run ruff check --fix && uv run mypy`
   - Run fast tests: `uv run pytest -m "not slow" -k "not integration"`
   - Ensure all pre-commit hooks pass
   - Update PR from Draft to Ready for Review
   - Request appropriate reviewers based on changed files

## 📋 Implementation Checklist

- [ ] Issue analysis completed
- [ ] Implementation plan created
- [ ] First meaningful commit made
- [ ] Draft PR created with proper linking
- [ ] Progress tracking implemented
- [ ] Core functionality implemented
- [ ] Tests written/updated
- [ ] Code quality checks passed (lint, type, tests)
- [ ] PR description updated with implementation details
- [ ] PR converted from Draft to Ready for Review

## 🔗 Issue Linking Best Practices

- Always include `Fixes #{issue_number}` or `Closes #{issue_number}` in PR description
- Reference related issues and discussions
- Add meaningful commit messages that reference the issue
- Update issue with PR link for transparency

## 💡 Tips for Success

- Create PR early for visibility and collaboration opportunities
- Use descriptive commit messages and PR updates
- Keep team informed of progress through PR comments
- Address review feedback promptly
- Ensure CI/CD checks pass before requesting final review 