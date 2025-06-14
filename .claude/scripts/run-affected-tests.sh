#!/bin/bash
# Script to run only tests affected by changed files

# Get list of changed files (staged and unstaged)
CHANGED_FILES=$(git diff --name-only HEAD; git diff --cached --name-only)

# If no files changed, exit early
if [ -z "$CHANGED_FILES" ]; then
    echo "No files changed, skipping tests"
    exit 0
fi

# Filter for Python files only
PYTHON_FILES=$(echo "$CHANGED_FILES" | grep -E "\.py$" || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python files changed, skipping tests"
    exit 0
fi

# Determine which test files to run based on changed files
TEST_FILES=""

for file in $PYTHON_FILES; do
    # If a test file was changed, run it
    if [[ $file == tests/* ]]; then
        TEST_FILES="$TEST_FILES $file"
    # If a source file was changed, find corresponding test
    elif [[ $file == src/* ]]; then
        # Convert src/oboyu/foo.py to tests/test_foo.py
        test_file=$(echo "$file" | sed 's|src/oboyu/|tests/test_|' | sed 's|\.py$|.py|')
        if [ -f "$test_file" ]; then
            TEST_FILES="$TEST_FILES $test_file"
        fi
        
        # Also check for module-level tests
        module_dir=$(dirname "$file" | sed 's|src/oboyu|tests|')
        if [ -d "$module_dir" ]; then
            module_tests=$(find "$module_dir" -name "test_*.py" 2>/dev/null || true)
            if [ -n "$module_tests" ]; then
                TEST_FILES="$TEST_FILES $module_tests"
            fi
        fi
    fi
done

# Remove duplicates and trim whitespace
TEST_FILES=$(echo "$TEST_FILES" | tr ' ' '\n' | sort -u | tr '\n' ' ' | xargs)

# Filter out non-existent files (e.g., deleted files)
EXISTING_TEST_FILES=""
for file in $TEST_FILES; do
    if [ -f "$file" ]; then
        EXISTING_TEST_FILES="$EXISTING_TEST_FILES $file"
    fi
done
EXISTING_TEST_FILES=$(echo "$EXISTING_TEST_FILES" | xargs)

if [ -z "$EXISTING_TEST_FILES" ]; then
    echo "No relevant test files found for changed files"
    exit 0
fi

echo "Running tests for changed files: $EXISTING_TEST_FILES"
uv run pytest -m "not slow" -k "not integration" --no-cov --tb=short -q $EXISTING_TEST_FILES