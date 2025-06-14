"""Enhanced E2E Testing with ttyd + Playwright integration for Oboyu."""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright

    ENHANCED_DEPENDENCIES_AVAILABLE = True
except ImportError:
    ENHANCED_DEPENDENCIES_AVAILABLE = False
    # Create stub classes for type checking
    Page = Any
    Browser = Any
    BrowserContext = Any


class TtydIntegratedOboyuTester:
    """Enhanced E2E tester with ttyd + Playwright integration using time-based completion detection."""

    def __init__(self, oboyu_path: str = "oboyu", ttyd_port: int = 7681) -> None:
        """Initialize the enhanced E2E tester."""
        if not ENHANCED_DEPENDENCIES_AVAILABLE:
            raise ImportError("Enhanced E2E testing dependencies not available. Install with: uv sync --group e2e-enhanced")

        self.oboyu_path = oboyu_path
        self.ttyd_port = ttyd_port
        self.test_data_dir: Path | None = None
        self.ttyd_process: subprocess.Popen[bytes] | None = None
        self.playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.screenshots_dir: Path | None = None

    def setup(self) -> None:
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="oboyu_e2e_"))
        self.test_db_path = self.test_data_dir / "test_index.db"

        # Create minimal test files
        test_files = {
            "test.txt": "This is a simple test document for Oboyu.",
            "日本語.txt": "これは日本語のテストです。",
        }

        for filename, content in test_files.items():
            (self.test_data_dir / filename).write_text(content, encoding="utf-8")

        # Create screenshots directory
        if self.test_data_dir:
            self.screenshots_dir = self.test_data_dir / "screenshots"
            self.screenshots_dir.mkdir(exist_ok=True)
            (self.screenshots_dir / "cli_commands").mkdir(exist_ok=True)

    def teardown(self) -> None:
        """Clean up test environment."""
        asyncio.run(self._cleanup_async_resources())

        if self.test_data_dir and self.test_data_dir.exists():
            import shutil

            shutil.rmtree(self.test_data_dir)

    async def _cleanup_async_resources(self) -> None:
        """Clean up async resources."""
        if self.page:
            try:
                await self.page.close()
            except Exception as e:
                print(f"Warning: Failed to close page: {e}")
            self.page = None

        if self.context:
            try:
                await self.context.close()
            except Exception as e:
                print(f"Warning: Failed to close context: {e}")
            self.context = None

        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                print(f"Warning: Failed to close browser: {e}")
            self.browser = None

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                print(f"Warning: Failed to stop playwright: {e}")
            self.playwright = None

        # Stop ttyd process
        if self.ttyd_process:
            try:
                self.ttyd_process.terminate()
                self.ttyd_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    self.ttyd_process.kill()
                    self.ttyd_process.wait(timeout=5)
                except Exception as e:
                    print(f"Warning: Failed to kill ttyd process: {e}")
            self.ttyd_process = None

    async def setup_ttyd_environment_once(self) -> None:
        """Set up ttyd and browser environment once for multiple tests."""
        print("🚀 Setting up ttyd environment for the first time...")

        # Find available port
        actual_port = await self._find_available_port()
        self.ttyd_port = actual_port

        # Start ttyd server
        print(f"Starting ttyd server on port {self.ttyd_port}...")
        self.ttyd_process = subprocess.Popen(  # noqa: S603
            ["ttyd", "--port", str(self.ttyd_port), "--writable", "bash"]  # noqa: S607
        )

        await self._wait_for_ttyd_ready()

        # Launch browser
        print("Launching Playwright browser...")
        print("Browser launch configuration:")
        print("- headless: False")
        print("- slow_mo: 200ms")
        print("- args: --no-sandbox, --disable-dev-shm-usage")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            slow_mo=200,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--start-maximized", "--no-first-run", "--disable-default-apps"],
        )
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

        print("✅ Browser launched successfully!")
        print(f"Browser process ID: {self.browser}")

        # Navigate to ttyd
        print(f"🌐 Navigating to ttyd terminal at http://localhost:{self.ttyd_port}")
        await self.page.goto(f"http://localhost:{self.ttyd_port}", timeout=30000)

        # Bring browser to front and make it visible
        await self.page.bring_to_front()
        print("Browser brought to front")
        print("Browser should be visible now!")

        # Set viewport size for consistent screenshots
        await self.page.set_viewport_size({"width": 1200, "height": 800})
        print("Viewport set to 1200x800")

        # Wait for terminal to be ready
        await asyncio.sleep(2)
        print("Terminal ready!")

    async def _find_available_port(self) -> int:
        """Find an available port starting from ttyd_port."""
        import socket

        for port in range(self.ttyd_port, self.ttyd_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                continue

        # If original port is available, use it
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", self.ttyd_port))
                return self.ttyd_port
        except OSError:
            print(f"⚠️  Port {self.ttyd_port} is already in use. Attempting to find free port...")
            # Try to find an alternative port
            for port in range(57000, 58000):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("localhost", port))
                        print(f"Using alternative port: {port}")
                        return port
                except OSError:
                    continue

        raise RuntimeError("Could not find available port")

    async def _wait_for_ttyd_ready(self, timeout: int = 30) -> None:
        """Wait for ttyd server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.ttyd_port}") as response:
                        if response.status == 200:
                            print("ttyd server is ready!")
                            return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(1)

        raise RuntimeError(f"ttyd server failed to start within {timeout} seconds")

    async def _focus_terminal(self) -> None:
        """Focus on the terminal for input."""
        try:
            # Try to click on xterm-screen first
            await self.page.click(".xterm-screen", timeout=5000)
            print("✅ Terminal focused using selector: .xterm-screen")
        except Exception:
            try:
                # Fallback to clicking on terminal div
                await self.page.click(".terminal", timeout=5000)
                print("✅ Terminal focused using selector: .terminal")
            except Exception as e:
                print(f"⚠️ Could not focus terminal: {e}")

    async def _send_terminal_command(self, command: str, wait_ms: int = 1000) -> None:
        """Send command to terminal with simple time-based completion detection."""
        print(f"Executing terminal command: {command}")

        # Focus on terminal
        await self._focus_terminal()

        # Type the command
        await self.page.keyboard.type(command)
        await self.page.keyboard.press("Enter")

        # Simple time-based waiting - proven to work reliably
        wait_time = min(wait_ms / 1000, 2.5)  # Maximum 2.5 seconds for efficiency
        print(f"⏱️ Waiting {wait_time}s for command completion...")
        await asyncio.sleep(wait_time)
        print(f"✅ Command completed: {command}")

    async def _capture_screenshot(self, name: str, category: str = "general") -> Path:
        """Capture a screenshot of the current terminal state."""
        if not self.page or not self.screenshots_dir:
            raise RuntimeError("Page or screenshots directory not initialized")

        category_dir = self.screenshots_dir / category
        category_dir.mkdir(exist_ok=True)

        timestamp = int(time.time() * 1000)
        screenshot_path = category_dir / f"{name}_{timestamp}.png"

        await self.page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Screenshot saved: {screenshot_path}")
        return screenshot_path

    def run_claude_check(self, prompt: str, context_files: list[str] | None = None) -> dict[str, Any]:
        """Execute Claude Code to check display issues."""
        if not context_files:
            context_files = []

        print("🔍 Starting Claude Code visual analysis with screenshots...")

        # Copy screenshots to current directory for Claude Code access
        local_screenshots = []
        for i, screenshot_path in enumerate(context_files, 1):
            if Path(screenshot_path).exists():
                local_name = f"screenshot_{i}_{Path(screenshot_path).stem}.png"
                local_path = Path.cwd() / local_name
                import shutil

                shutil.copy2(screenshot_path, local_path)
                local_screenshots.append(local_name)
                print(f"Copied screenshot: {local_name}")

        if not local_screenshots:
            return {"error": "No screenshots available for analysis", "result": "No visual analysis possible"}

        try:
            print("\n--- Executing Claude Code ---")
            print(f"Including {len(local_screenshots)} screenshots for analysis")

            # Build Claude Code command
            cmd = ["claude", "--output-format", "json"] + local_screenshots + ["-p", prompt]
            print(f"Command: {' '.join(cmd[:3])} {' '.join(local_screenshots)} -p {prompt}")

            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=300
            )

            print("--- Claude Code Output ---")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Errors: {result.stderr}")
            print("--- Claude Code Finished ---")

            # Clean up local screenshots
            for screenshot in local_screenshots:
                try:
                    Path(screenshot).unlink()
                    print(f"Cleaned up: {screenshot}")
                except Exception as e:
                    print(f"Warning: Could not clean up {screenshot}: {e}")

            if result.returncode != 0:
                return {"error": f"Claude Code failed with return code {result.returncode}", "stderr": result.stderr}

            # Parse Claude Code output
            try:
                data = json.loads(result.stdout)
                return data
            except json.JSONDecodeError:
                # Handle stream-json format
                lines = result.stdout.strip().split("\n")
                for line in reversed(lines):
                    try:
                        data = json.loads(line)
                        if data.get("type") == "result":
                            return data
                    except json.JSONDecodeError:
                        continue
                raise RuntimeError(f"Could not parse Claude Code output: {result.stdout}")

        except subprocess.TimeoutExpired:
            return {"error": "Claude Code execution timed out", "result": "Analysis timeout"}
        except Exception as e:
            return {"error": f"Claude Code execution failed: {e}", "result": "Analysis failed"}

    async def test_visual_cli_commands(self) -> dict[str, Any]:
        """Test visual CLI commands display."""
        print("\n=== VISUAL CLI COMMANDS TEST ===")

        await self.setup_ttyd_environment_once()

        try:
            results = {"screenshots": [], "commands_tested": []}

            # Change to test directory
            await self._send_terminal_command(f"cd {self.test_data_dir}")

            # Test basic CLI commands with visual verification
            commands = [
                ("oboyu --help", "help_command"),
                ("oboyu version", "version_command"),
                ("oboyu health --help", "health_command"),
            ]

            for command, screenshot_name in commands:
                await self._send_terminal_command(command)
                screenshot = await self._capture_screenshot(screenshot_name, "cli_commands")
                results["screenshots"].append(str(screenshot))
                results["commands_tested"].append(command)

            # Analyze with Claude Code
            claude_prompt = """Please analyze the 3 terminal screenshots I've provided.

These screenshots show Oboyu CLI commands executed in a browser terminal:
1. Help command output (--help)
2. Version command output (version)
3. Health command help (health --help)

For each screenshot, evaluate:
- Text readability and clarity
- Terminal formatting and alignment
- Colors and contrast
- Any visual issues or artifacts

Provide an assessment of the overall visual quality and note any problems you observe across all screenshots."""

            claude_result = self.run_claude_check(claude_prompt, context_files=results["screenshots"])
            results["claude_analysis"] = claude_result
            print("✅ Claude Code visual analysis completed")

            return results

        except Exception as e:
            print(f"⚠️ Visual CLI commands test failed: {e}")
            return {"error": str(e), "test": "visual_cli_commands"}

    def generate_report(self, results: dict[str, dict[str, Any]]) -> str:
        """Generate test report."""
        report = ["# Enhanced Oboyu E2E Test Report", ""]
        report.append("*Generated with ttyd + Playwright integration (time-based completion detection)*")
        report.append("")

        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- Total tests run: {len(results)}")
        report.append(f"- Test environment: {Path.cwd()}")
        report.append(f"- Oboyu command: {self.oboyu_path}")
        report.append(f"- ttyd port: {self.ttyd_port}")
        if self.screenshots_dir:
            report.append(f"- Screenshots directory: {self.screenshots_dir}")
        report.append("")

        # Enhancement Features
        report.append("## Enhancement Features")
        report.append("")
        report.append("- ✅ Browser-based terminal testing with ttyd")
        report.append("- ✅ Real-time visual verification with Playwright")
        report.append("- ✅ Screenshot capture at multiple stages")
        report.append("- ✅ Time-based reliable completion detection")
        report.append("- ✅ Claude Code visual analysis integration")
        report.append("")

        # Test Results
        report.append("## Test Results")
        report.append("")

        for test_name, result in results.items():
            report.append(f"### {test_name.replace('_', ' ').title()}")
            report.append("")

            if isinstance(result, dict):
                if "error" in result:
                    report.append("❌ **Status:** Failed")
                    report.append(f"**Error:** {result['error']}")
                else:
                    report.append("✅ **Status:** Completed")

                    # Add screenshots info
                    if "screenshots" in result:
                        report.append(f"**Screenshots captured:** {len(result['screenshots'])}")

                    # Add Claude analysis
                    if "claude_analysis" in result:
                        claude_result = result["claude_analysis"]
                        if "result" in claude_result:
                            report.append("**Claude Code Analysis:**")
                            report.append(f"> {claude_result['result']}")

                        if "total_cost_usd" in claude_result:
                            report.append(f"**Analysis cost:** ${claude_result['total_cost_usd']:.4f}")

            report.append("")

        return "\n".join(report)


if __name__ == "__main__":
    # Simple test
    tester = TtydIntegratedOboyuTester()
    tester.setup()

    try:
        result = asyncio.run(tester.test_visual_cli_commands())
        print("Test completed successfully!")
        print(f"Result: {result}")
    finally:
        tester.teardown()
