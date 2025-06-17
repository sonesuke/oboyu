#!/bin/bash
# Verification script to test all oboyu CLI --help commands

echo "=== Verifying Oboyu CLI Help Commands ==="
echo "Testing all commands with --help flag..."
echo ""

# Define all commands to test
commands=(
    "--help"
    "version --help"
    "index --help"
    "search --help"
    "enrich --help"
    "build-kg --help"
    "deduplicate --help"
    "mcp --help"
    "clear --help"
    "status --help"
)

# Test each command
for cmd in "${commands[@]}"; do
    echo "========================================="
    echo "Testing: oboyu $cmd"
    echo "========================================="
    oboyu $cmd
    if [ $? -ne 0 ]; then
        echo "ERROR: Command failed: oboyu $cmd"
        exit 1
    fi
    echo ""
    echo ""
done

echo "✅ All commands successfully displayed help information!"