"""Enhanced E2E Testing with ttyd + Playwright integration for Oboyu."""

import asyncio
import subprocess
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

from display_tester import OboyuE2EDisplayTester


class TtydIntegratedOboyuTester(OboyuE2EDisplayTester):
    """Enhanced E2E tester with ttyd + Playwright integration for visual CLI testing."""

    def __init__(self, oboyu_path: str = "oboyu", ttyd_port: int = 7681) -> None:
        """Initialize the enhanced E2E tester.

        Args:
            oboyu_path: Path to the oboyu command
            ttyd_port: Port for ttyd server

        Raises:
            ImportError: If enhanced dependencies are not available

        """
        if not ENHANCED_DEPENDENCIES_AVAILABLE:
            raise ImportError("Enhanced E2E testing dependencies not available. Install with: uv sync --group e2e-enhanced")

        super().__init__(oboyu_path)
        self.ttyd_port = ttyd_port
        self.ttyd_process: subprocess.Popen[bytes] | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.screenshots_dir: Path | None = None

    def setup(self) -> None:
        """Set up test environment including ttyd and browser."""
        super().setup()

        # Create screenshots directory
        if self.test_data_dir:
            self.screenshots_dir = self.test_data_dir / "screenshots"
            self.screenshots_dir.mkdir(exist_ok=True)

            # Create subdirectories for organized screenshot storage
            (self.screenshots_dir / "cli_commands").mkdir(exist_ok=True)
            (self.screenshots_dir / "progress_monitoring").mkdir(exist_ok=True)
            (self.screenshots_dir / "mcp_integration").mkdir(exist_ok=True)

    def teardown(self) -> None:
        """Clean up test environment including ttyd and browser."""
        asyncio.run(self._cleanup_async_resources())
        super().teardown()

    async def _cleanup_async_resources(self) -> None:
        """Clean up async resources (browser, ttyd process)."""
        if self.page:
            await self.page.close()
            self.page = None

        if self.context:
            await self.context.close()
            self.context = None

        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.ttyd_process:
            self.ttyd_process.terminate()
            try:
                self.ttyd_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ttyd_process.kill()
            self.ttyd_process = None

    async def setup_ttyd_environment(self) -> None:
        """Start ttyd server and launch Playwright browser."""
        # Check if ttyd is available
        try:
            subprocess.run(["ttyd", "--version"], capture_output=True, check=True)  # noqa: S603, S607
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("ttyd not found. Please install ttyd first. On macOS: brew install ttyd, On Ubuntu: apt install ttyd") from e

        # Start ttyd server with bash shell
        print(f"Starting ttyd server on port {self.ttyd_port}...")
        self.ttyd_process = subprocess.Popen(  # noqa: S603
            [  # noqa: S607
                "ttyd",
                "--port",
                str(self.ttyd_port),
                "--writable",  # Allow input
                "--once",  # Close after one session
                "bash",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for ttyd to start
        await self._wait_for_ttyd_ready()

        # Launch Playwright browser
        print("Launching Playwright browser...")
        playwright = await async_playwright().start()
        # Explicitly configure for visible browser with slow motion for debugging
        self.browser = await playwright.chromium.launch(
            headless=False,  # Make browser visible
            slow_mo=1000,  # Slow down actions for visibility (1 second delay)
            args=["--no-sandbox", "--disable-dev-shm-usage"],  # For compatibility
        )
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

        print("✅ Browser launched successfully!")
        print(f"🌐 Navigating to ttyd terminal at http://localhost:{self.ttyd_port}")

        # Navigate to ttyd terminal
        await self.page.goto(f"http://localhost:{self.ttyd_port}")

        # Wait for terminal to be ready
        await self.page.wait_for_selector(".xterm-screen", timeout=10000)
        print("Terminal ready!")

        # Setup working directory in terminal
        if self.test_data_dir:
            await self._send_terminal_command(f"cd {self.test_data_dir}")

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

    async def _send_terminal_command(self, command: str, wait_ms: int = 1000) -> None:
        """Send a command to the terminal and wait for execution."""
        if not self.page:
            raise RuntimeError("Browser page not initialized")

        # Focus on terminal
        await self.page.click(".xterm-screen")

        # Type command
        await self.page.keyboard.type(command)
        await self.page.keyboard.press("Enter")

        # Wait for command to execute
        await asyncio.sleep(wait_ms / 1000)

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

    async def test_visual_cli_commands(self) -> dict[str, Any]:
        """Test CLI commands with visual verification in browser terminal."""
        print("\n=== VISUAL CLI COMMANDS TEST ===")

        await self.setup_ttyd_environment()

        try:
            results = {"screenshots": [], "commands_tested": [], "visual_analysis": {}}

            # Test help command
            await self._send_terminal_command(f"{self.oboyu_path} --help")
            help_screenshot = await self._capture_screenshot("help_command", "cli_commands")
            results["screenshots"].append(str(help_screenshot))
            results["commands_tested"].append("--help")

            # Test version command
            await self._send_terminal_command(f"{self.oboyu_path} version")
            version_screenshot = await self._capture_screenshot("version_command", "cli_commands")
            results["screenshots"].append(str(version_screenshot))
            results["commands_tested"].append("version")

            # Test health command
            await self._send_terminal_command(f"{self.oboyu_path} health --help")
            health_screenshot = await self._capture_screenshot("health_command", "cli_commands")
            results["screenshots"].append(str(health_screenshot))
            results["commands_tested"].append("health --help")

            # Analyze with Claude Code using screenshots
            visual_analysis_prompt = f"""
            I have executed several Oboyu CLI commands in a browser terminal using ttyd and captured screenshots.
            Please analyze these screenshots for display quality:
            
            Commands tested: {", ".join(results["commands_tested"])}
            Screenshots captured: {len(results["screenshots"])} images
            
            Check for:
            1. Text readability in browser terminal
            2. Proper formatting and alignment
            3. No visual artifacts or rendering issues
            4. Consistent styling across commands
            5. Appropriate terminal colors and contrast
            
            Provide assessment of the visual CLI display quality in the browser environment.
            """

            claude_result = self.run_claude_check(visual_analysis_prompt, context_files=results["screenshots"])
            results["visual_analysis"] = claude_result

            return results

        finally:
            await self._cleanup_async_resources()

    async def test_interactive_indexing_progress(self) -> dict[str, Any]:
        """Test indexing with real-time progress monitoring and screenshots."""
        if not self.test_data_dir:
            raise RuntimeError("Test data directory not set up")

        print("\n=== INTERACTIVE INDEXING PROGRESS TEST ===")

        await self.setup_ttyd_environment()

        try:
            results = {"screenshots": [], "progress_stages": [], "timing_analysis": {}}

            # Start indexing command
            start_time = time.time()
            await self._send_terminal_command(f"{self.oboyu_path} --db-path {self.test_db_path} index {self.test_data_dir}", wait_ms=500)

            # Capture progress at different stages
            progress_stages = [("start", 1000), ("progress_1", 2000), ("progress_2", 3000), ("completion", 5000)]

            for stage_name, wait_ms in progress_stages:
                await asyncio.sleep(wait_ms / 1000)
                screenshot = await self._capture_screenshot(f"indexing_{stage_name}", "progress_monitoring")
                results["screenshots"].append(str(screenshot))
                results["progress_stages"].append(stage_name)

            end_time = time.time()
            results["timing_analysis"]["total_duration"] = end_time - start_time
            results["timing_analysis"]["stages_captured"] = len(progress_stages)

            # Analyze progress display with Claude Code
            progress_analysis_prompt = f"""
            I monitored an Oboyu indexing operation in real-time using a browser terminal.
            Captured {len(results["screenshots"])} screenshots showing progress at different stages:
            {", ".join(results["progress_stages"])}
            
            Total operation time: {results["timing_analysis"]["total_duration"]:.2f} seconds
            
            Please analyze the progress display quality:
            1. Progress bars render correctly in browser
            2. Real-time updates are visually smooth
            3. Hierarchical logging structure is clear
            4. Completion status is properly indicated
            5. Terminal handles dynamic content updates well
            
            Provide assessment of the interactive progress monitoring experience.
            """

            claude_result = self.run_claude_check(progress_analysis_prompt, context_files=results["screenshots"])
            results["progress_analysis"] = claude_result

            return results

        finally:
            await self._cleanup_async_resources()

    def test_enhanced_basic_cli_display(self) -> dict[str, Any]:
        """Enhanced version of basic CLI test with visual components."""
        # First run the traditional test
        traditional_result = self.test_basic_cli_display()

        # Then run the visual test
        visual_result = asyncio.run(self.test_visual_cli_commands())

        # Combine results
        return {"traditional_analysis": traditional_result, "visual_analysis": visual_result, "enhancement_type": "ttyd_playwright_integration"}

    def test_enhanced_indexing_progress_display(self) -> dict[str, Any]:
        """Enhanced version of indexing test with real-time visual monitoring."""
        # First run the traditional test
        traditional_result = self.test_indexing_progress_display()

        # Then run the interactive visual test
        interactive_result = asyncio.run(self.test_interactive_indexing_progress())

        # Combine results
        return {"traditional_analysis": traditional_result, "interactive_analysis": interactive_result, "enhancement_type": "real_time_progress_monitoring"}

    def generate_enhanced_report(self, results: dict[str, dict[str, Any]]) -> str:  # noqa: C901
        """Generate enhanced test report with visual analysis and screenshots."""
        report = ["# Enhanced Oboyu E2E Display Test Report", ""]
        report.append("*Generated with ttyd + Playwright integration*")
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

        # Enhanced Features
        report.append("## Enhancement Features")
        report.append("")
        report.append("- ✅ Browser-based terminal testing with ttyd")
        report.append("- ✅ Real-time visual verification with Playwright")
        report.append("- ✅ Screenshot capture at multiple stages")
        report.append("- ✅ Interactive progress monitoring")
        report.append("- ✅ Combined traditional + visual analysis")
        report.append("")

        # Test Results
        report.append("## Test Results")
        report.append("")

        for test_name, result in results.items():
            report.append(f"### {test_name.replace('_', ' ').title()}")
            report.append("")

            if isinstance(result, dict):
                # Handle enhanced test results
                if "enhancement_type" in result:
                    report.append(f"**Enhancement Type:** {result['enhancement_type']}")
                    report.append("")

                    if "traditional_analysis" in result:
                        report.append("#### Traditional Analysis")
                        if isinstance(result["traditional_analysis"], dict) and "result" in result["traditional_analysis"]:
                            report.append(result["traditional_analysis"]["result"])
                        report.append("")

                    if "visual_analysis" in result:
                        report.append("#### Visual Analysis")
                        if isinstance(result["visual_analysis"], dict):
                            visual = result["visual_analysis"]
                            if "commands_tested" in visual:
                                report.append(f"**Commands tested:** {', '.join(visual['commands_tested'])}")
                            if "screenshots" in visual:
                                report.append(f"**Screenshots captured:** {len(visual['screenshots'])}")
                                for screenshot in visual["screenshots"]:
                                    report.append(f"  - {screenshot}")
                            if "visual_analysis" in visual and "result" in visual["visual_analysis"]:
                                report.append("**Analysis:**")
                                report.append(visual["visual_analysis"]["result"])
                        report.append("")

                    if "interactive_analysis" in result:
                        report.append("#### Interactive Analysis")
                        if isinstance(result["interactive_analysis"], dict):
                            interactive = result["interactive_analysis"]
                            if "progress_stages" in interactive:
                                report.append(f"**Progress stages captured:** {', '.join(interactive['progress_stages'])}")
                            if "timing_analysis" in interactive:
                                timing = interactive["timing_analysis"]
                                if "total_duration" in timing:
                                    report.append(f"**Total duration:** {timing['total_duration']:.2f} seconds")
                            if "progress_analysis" in interactive and "result" in interactive["progress_analysis"]:
                                report.append("**Analysis:**")
                                report.append(interactive["progress_analysis"]["result"])
                        report.append("")

                # Handle standard test results
                elif "result" in result:
                    report.append(result["result"])
                elif "error" in result:
                    report.append(f"**Error:** {result['error']}")
            else:
                report.append(str(result))

            report.append("")

        # Visual Assets Summary
        if self.screenshots_dir and self.screenshots_dir.exists():
            report.append("## Visual Assets")
            report.append("")

            # Count screenshots by category
            categories = ["cli_commands", "progress_monitoring", "mcp_integration"]
            for category in categories:
                category_dir = self.screenshots_dir / category
                if category_dir.exists():
                    screenshots = list(category_dir.glob("*.png"))
                    report.append(f"- **{category.replace('_', ' ').title()}:** {len(screenshots)} screenshots")
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
        report.append("- Enhanced testing: ttyd + Playwright")
        report.append("")

        return "\n".join(report)
