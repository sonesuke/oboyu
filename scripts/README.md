# Development Scripts

This directory contains development and CI/CD automation scripts for the oboyu project.

## Directory Structure

```
scripts/
├── dev/                        # Development automation scripts
│   ├── cleanup-bug-worktree.sh    # Clean up bug fix worktrees
│   ├── cleanup-issue-worktree.sh  # Clean up issue worktrees
│   └── fix-common-issues.sh       # Auto-fix common code quality issues
├── ci/                         # CI/CD automation scripts
│   ├── monitor-pr-checks.sh       # Monitor PR CI/CD status with auto-fixes
│   └── run-affected-tests.sh      # Run tests affected by changes
└── README.md                   # This file
```

## Development Scripts (`dev/`)

### cleanup-bug-worktree.sh
Automates cleanup of git worktrees created for bug fixes.

**Usage:**
```bash
scripts/dev/cleanup-bug-worktree.sh <bug-number>
```

**What it does:**
- Removes the worktree directory for the specified bug
- Deletes the feature branch locally and remotely
- Returns to main branch and updates it
- Cleans up any temporary files

### cleanup-issue-worktree.sh
Automates cleanup of git worktrees created for GitHub issues.

**Usage:**
```bash
scripts/dev/cleanup-issue-worktree.sh <issue-number>
```

**What it does:**
- Removes the worktree directory for the specified issue
- Deletes the feature branch locally and remotely
- Returns to main branch and updates it
- Cleans up any temporary files

### fix-common-issues.sh
Automatically fixes common code quality issues like linting, formatting, and imports.

**Usage:**
```bash
scripts/dev/fix-common-issues.sh
```

**What it does:**
- Runs `ruff check --fix` for linting issues
- Runs `ruff format` for code formatting
- Organizes import statements
- Commits fixes automatically if changes are made

## CI/CD Scripts (`ci/`)

### monitor-pr-checks.sh
Monitors Pull Request CI/CD checks and automatically fixes common issues.

**Usage:**
```bash
scripts/ci/monitor-pr-checks.sh <pr-number>
```

**What it does:**
- Monitors PR status every 30 seconds
- Detects failing CI/CD checks
- Automatically applies fixes for common issues
- Promotes draft PRs to ready when all checks pass
- Runs until PR is merged or closed

**Features:**
- Real-time status monitoring
- Automatic linting and formatting fixes
- Intelligent PR promotion
- Continuous feedback loop

### run-affected-tests.sh
Intelligently runs only tests affected by code changes.

**Usage:**
```bash
scripts/ci/run-affected-tests.sh
```

**What it does:**
- Analyzes changed files in the current branch
- Identifies affected test files
- Runs only relevant tests to save time
- Reports test results and coverage

## Usage in Development Workflow

### With Claude Code
These scripts are integrated with the Claude Code solve-issue workflow:

```bash
# Scripts are referenced in .claude/commands/solve-issue.md
# and automatically used during issue resolution
```

### Manual Usage
Scripts can be run independently from the project root:

```bash
# Fix code quality issues
./scripts/dev/fix-common-issues.sh

# Monitor a PR (run in foreground)
./scripts/ci/monitor-pr-checks.sh 123

# Clean up after issue completion
./scripts/dev/cleanup-issue-worktree.sh 343
```

### Integration with Other Tools
Scripts are designed to work with:
- **Git worktrees**: For isolated development environments
- **GitHub CLI (`gh`)**: For PR and issue management
- **UV**: For Python dependency management
- **Ruff**: For linting and formatting
- **MyPy**: For type checking
- **Pytest**: For testing

## Prerequisites

Before using these scripts, ensure you have:
- **Git**: Version control
- **GitHub CLI (`gh`)**: For GitHub operations
- **UV**: Python package manager
- **Ruff**: Linting and formatting tool
- **MyPy**: Type checking tool
- **Pytest**: Testing framework

Install prerequisites:
```bash
# Install GitHub CLI
brew install gh

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Other tools are installed via: uv sync
```

## Configuration

Scripts use the following configuration:
- **Monitoring interval**: 30 seconds (in monitor-pr-checks.sh)
- **Auto-fix timeout**: 5 minutes per attempt
- **Branch patterns**: `<issue-number>-<description>` format
- **Worktree location**: `.worktree/issue-<number>/`

## Error Handling

All scripts include comprehensive error handling:
- **Graceful failures**: Scripts won't leave the repository in a broken state
- **Rollback capability**: Failed operations are automatically reverted
- **Detailed logging**: Clear error messages and debugging information
- **Exit codes**: Proper exit codes for CI/CD integration

## Contributing

When adding new scripts:
1. Place them in the appropriate subdirectory (`dev/` or `ci/`)
2. Make them executable: `chmod +x scripts/path/to/script.sh`
3. Add documentation to this README
4. Include proper error handling and logging
5. Test scripts thoroughly before committing

## Troubleshooting

### Common Issues

**Script not executable:**
```bash
chmod +x scripts/path/to/script.sh
```

**GitHub CLI not authenticated:**
```bash
gh auth login
```

**Missing dependencies:**
```bash
uv sync
```

**Worktree conflicts:**
```bash
git worktree prune
```

For more help, check individual script help:
```bash
scripts/path/to/script.sh --help
```