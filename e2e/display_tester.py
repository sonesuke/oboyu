"""E2E Display Testing with Claude Code SDK for Oboyu."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class OboyuE2EDisplayTester:
    """Claude Code SDK-based display testing for Oboyu."""

    def __init__(self, oboyu_path: str = "oboyu") -> None:
        """Initialize the E2E display tester.
        
        Args:
            oboyu_path: Path to the oboyu command

        """
        self.oboyu_path = oboyu_path
        self.test_data_dir: Path | None = None

    def setup(self) -> None:
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="oboyu_e2e_"))
        
        # Create test database path (separate from user's database)
        self.test_db_path = self.test_data_dir / "test_index.db"
        
        # Create minimal test files for faster testing
        test_files = {
            "test.txt": "This is a simple test document for Oboyu.",
            "日本語.txt": "これは日本語のテストです。",
        }
        
        for filename, content in test_files.items():
            (self.test_data_dir / filename).write_text(content, encoding="utf-8")

    def teardown(self) -> None:
        """Clean up test environment."""
        if self.test_data_dir and self.test_data_dir.exists():
            import shutil
            shutil.rmtree(self.test_data_dir)

    def run_oboyu_command(self, command_args: list[str], show_output: bool = True) -> tuple[int, str, str]:
        """Run an oboyu command and optionally show its output.
        
        Args:
            command_args: Arguments to pass to oboyu command
            show_output: Whether to show command output in real-time
            
        Returns:
            Tuple of (return_code, stdout, stderr)

        """
        cmd = [self.oboyu_path]
        
        # Always use test database to avoid contaminating user's database
        # (unless --db-path is already specified in command_args)
        if hasattr(self, 'test_db_path') and "--db-path" not in command_args:
            cmd.extend(["--db-path", str(self.test_db_path)])
        
        cmd.extend(command_args)
        
        if show_output:
            print(f"\n🔍 Executing: {' '.join(cmd)}")
            print("-" * 60)
        
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        
        if show_output:
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            print("-" * 60)
            print(f"Return code: {result.returncode}\n")
        
        return result.returncode, result.stdout, result.stderr

    def run_claude_check(self, prompt: str, context_files: list[str] | None = None) -> dict[str, Any]:
        """Execute Claude Code to check display issues.
        
        Args:
            prompt: The prompt for Claude Code
            context_files: Optional list of context files to include
            
        Returns:
            Dict containing Claude Code's response
            
        Raises:
            RuntimeError: If Claude Code execution fails

        """
        claude_prompt = f"""
        You are an e2e tester for Oboyu, a Python-based search and indexing tool.
        I have already executed the commands and captured their output.
        Your task is to analyze the output that was already captured and displayed.

        DO NOT attempt to run any commands - just analyze the output I've shown you.

        Check for display abnormalities from these perspectives:
        1. CLI command output formatting is normal
        2. Progress display works properly
        3. Error messages are displayed appropriately
        4. Japanese text is not garbled
        5. JSON output format is correct

        Specific task:
        {prompt}

        Provide a brief assessment of the display quality based on the output shown above.
        If there are problems, report them specifically and include fix suggestions.
        """
        
        # Build command
        cmd = ["claude", "-p", claude_prompt, "--output-format", "json"]
        
        # Add context files if provided
        if context_files:
            for file in context_files:
                cmd.extend(["-f", file])
        
        # Execute Claude Code with streaming output
        print("\n--- Executing Claude Code ---")
        print(f"Command: {' '.join(cmd[:2])} [prompt] {' '.join(cmd[3:])}")
        print("--- Claude Code Output ---")
        
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd(),
            env={**os.environ, "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Read output in real-time
        import sys
        
        while True:
            # Check if process is still running
            if process.poll() is not None:
                # Process finished, read remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    print(remaining_stdout, end='')
                    stdout_lines.append(remaining_stdout)
                if remaining_stderr:
                    print(remaining_stderr, end='', file=sys.stderr)
                    stderr_lines.append(remaining_stderr)
                break
            
            # Read available output
            if process.stdout and process.stdout.readable():
                line = process.stdout.readline()
                if line:
                    print(line, end='')
                    stdout_lines.append(line)
            
            if process.stderr and process.stderr.readable():
                line = process.stderr.readline()
                if line:
                    print(line, end='', file=sys.stderr)
                    stderr_lines.append(line)
        
        stdout_text = ''.join(stdout_lines)
        stderr_text = ''.join(stderr_lines)
        
        print("\n--- Claude Code Finished ---\n")
        
        # Create a result object similar to subprocess.run
        class Result:
            def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        result = Result(process.returncode, stdout_text, stderr_text)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if "ANTHROPIC_API_KEY" in error_msg:
                error_msg += "\nPlease set your API key with: export ANTHROPIC_API_KEY=your_key"
            raise RuntimeError(f"Claude Code failed: {error_msg}")
            
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            # Handle stream-json format
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    if data.get('type') == 'result':
                        return data
                except json.JSONDecodeError:
                    continue
            raise RuntimeError(f"Could not parse Claude Code output: {result.stdout}")

    def test_basic_cli_display(self) -> dict[str, Any]:
        """Test basic CLI display functionality."""
        print("\n=== BASIC CLI DISPLAY TEST ===")
        
        # Run actual oboyu commands and capture output
        _, help_output, help_error = self.run_oboyu_command(["--help"])
        _, version_output, version_error = self.run_oboyu_command(["version"])
        _, health_output, health_error = self.run_oboyu_command(["health", "--help"])
        
        prompt = f"""
        I just ran these Oboyu CLI commands. Here is the actual output:

        COMMAND 1: `{self.oboyu_path} --help`
        STDOUT:
        {help_output}
        STDERR:
        {help_error}

        COMMAND 2: `{self.oboyu_path} version`
        STDOUT:
        {version_output}
        STDERR:
        {version_error}

        COMMAND 3: `{self.oboyu_path} health --help`
        STDOUT:
        {health_output}
        STDERR:
        {health_error}

        Based on this actual output, check that:
        - Text is properly aligned and formatted
        - No garbled or unexpected characters
        - Help text is clear and well-structured
        - Commands complete successfully (no error output)
        
        Provide a brief assessment of the display quality and any issues found.
        """
        
        return self.run_claude_check(prompt)

    def test_indexing_progress_display(self) -> dict[str, Any]:
        """Test indexing progress display."""
        if not self.test_data_dir:
            raise RuntimeError("Test data directory not set up")
            
        print("\n=== INDEXING PROGRESS DISPLAY TEST ===")
        
        # Run actual indexing command and capture output
        _, index_out, index_err = self.run_oboyu_command(["index", str(self.test_data_dir)])
        
        prompt = f"""
        I just ran the Oboyu indexing command. Here is the actual output:

        COMMAND: `{self.oboyu_path} index {self.test_data_dir}`
        STDOUT:
        {index_out}
        STDERR:
        {index_err}

        Based on this actual output, check that:
        - Progress bars display correctly
        - Hierarchical structure is clear
        - File processing status is visible
        - Completion messages are properly formatted
        - No display artifacts or glitches
        
        Provide a brief assessment of the indexing progress display quality.
        """
        
        return self.run_claude_check(prompt)

    def test_search_result_display(self) -> dict[str, Any]:
        """Test search result display."""
        if not self.test_data_dir:
            raise RuntimeError("Test data directory not set up")
            
        print("\n=== SEARCH RESULT DISPLAY TEST ===")
        
        # Ensure data is indexed first (may already be done)
        self.run_oboyu_command(["index", str(self.test_data_dir)])
        
        # Run search commands to show output
        _, search_text_out, search_text_err = self.run_oboyu_command(["query", "--query", "test", "--format", "text"])
        _, search_json_out, search_json_err = self.run_oboyu_command(["query", "--query", "test", "--format", "json"])
        _, search_jp_out, search_jp_err = self.run_oboyu_command(["query", "--query", "日本語", "--format", "text"])
        
        prompt = f"""
        I just ran these Oboyu search commands. Here is the actual output:

        COMMAND 1: `{self.oboyu_path} query --query "test" --format text`
        STDOUT:
        {search_text_out}
        STDERR:
        {search_text_err}

        COMMAND 2: `{self.oboyu_path} query --query "test" --format json`
        STDOUT:
        {search_json_out}
        STDERR:
        {search_json_err}

        COMMAND 3: `{self.oboyu_path} query --query "日本語" --format text`
        STDOUT:
        {search_jp_out}
        STDERR:
        {search_jp_err}

        Based on this actual output, check that:
        - Search results are properly formatted
        - Snippets are highlighted correctly
        - Scores are displayed clearly
        - Japanese text renders without issues
        - JSON format is valid and parseable
        
        Provide a brief assessment of the search result display quality.
        """
        
        return self.run_claude_check(prompt)


    def test_error_display(self) -> dict[str, Any]:
        """Test error message display."""
        print("\n=== ERROR DISPLAY TEST ===")
        
        # Run commands that should produce errors and capture output
        _, error1_out, error1_err = self.run_oboyu_command(["index", "/non/existent/path"])
        _, error2_out, error2_err = self.run_oboyu_command(["index", ".", "--config", "/invalid/config.yaml"])
        _, error3_out, error3_err = self.run_oboyu_command(["query", "--query", "test", "--db-path", "/non/existent.db"])
        _, error4_out, error4_err = self.run_oboyu_command(["query", "--invalid-option"])
        
        prompt = f"""
        I just ran these Oboyu commands that should produce errors. Here is the actual output:

        COMMAND 1: `{self.oboyu_path} index /non/existent/path`
        STDOUT:
        {error1_out}
        STDERR:
        {error1_err}

        COMMAND 2: `{self.oboyu_path} index . --config /invalid/config.yaml`
        STDOUT:
        {error2_out}
        STDERR:
        {error2_err}

        COMMAND 3: `{self.oboyu_path} query --query "test" --db-path /non/existent.db`
        STDOUT:
        {error3_out}
        STDERR:
        {error3_err}

        COMMAND 4: `{self.oboyu_path} query --invalid-option`
        STDOUT:
        {error4_out}
        STDERR:
        {error4_err}

        Based on this actual error output, check that:
        - Error messages are clear and helpful
        - Formatting is consistent
        - Stack traces (if shown) are readable
        - Exit codes are appropriate (commands should fail)
        - User gets actionable information
        
        Provide a brief assessment of the error display quality.
        """
        
        return self.run_claude_check(prompt)


    def generate_report(self, results: dict[str, dict[str, Any]]) -> str:
        """Generate a comprehensive test report.
        
        Args:
            results: Dict mapping test names to their results
            
        Returns:
            Markdown formatted report

        """
        report = ["# Oboyu E2E Display Test Report", ""]
        
        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- Total tests run: {len(results)}")
        report.append(f"- Test environment: {Path.cwd()}")
        report.append(f"- Oboyu command: {self.oboyu_path}")
        report.append("")
        
        # Test Results
        report.append("## Test Results")
        report.append("")
        
        for test_name, result in results.items():
            report.append(f"### {test_name.replace('_', ' ').title()}")
            report.append("")
            
            if isinstance(result, dict):
                if "result" in result:
                    report.append(result["result"])
                if "error" in result:
                    report.append(f"**Error:** {result['error']}")
            else:
                report.append(str(result))
            
            report.append("")
        
        # Metadata
        report.append("## Metadata")
        report.append("")
        
        total_cost = sum(r.get("cost_usd", 0) for r in results.values() if isinstance(r, dict))
        total_duration = sum(r.get("duration_ms", 0) for r in results.values() if isinstance(r, dict))
        total_turns = sum(r.get("num_turns", 0) for r in results.values() if isinstance(r, dict))
        
        report.append(f"- Total cost: ${total_cost:.4f}")
        report.append(f"- Total duration: {total_duration}ms")
        report.append(f"- Total turns: {total_turns}")
        report.append("")
        
        return "\n".join(report)

