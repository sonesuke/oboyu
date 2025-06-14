#!/usr/bin/env python3
"""Run enhanced E2E tests for Oboyu using ttyd + Playwright integration."""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    from ttyd_integrated_tester import TtydIntegratedOboyuTester
    from visual_validation import VisualValidationHelper

    ENHANCED_TESTING_AVAILABLE = True
except ImportError as e:
    ENHANCED_TESTING_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Fallback to standard testing if enhanced dependencies not available
from display_tester import OboyuE2EDisplayTester


def main() -> None:  # noqa: C901
    """Run enhanced E2E tests with various options."""
    parser = argparse.ArgumentParser(description="Run enhanced E2E tests for Oboyu using ttyd + Playwright integration")
    parser.add_argument(
        "--test",
        choices=[
            "all",
            "basic",
            "indexing",
            "search",
            "error",
            "visual-cli",
            "interactive-progress",
            "enhanced-basic",
            "enhanced-indexing",
        ],
        default="all",
        help="Which test(s) to run (default: all)",
    )
    parser.add_argument(
        "--oboyu-path",
        default="oboyu",
        help="Path to oboyu command (default: oboyu)",
    )
    parser.add_argument(
        "--report",
        default="e2e/e2e_enhanced_report.md",
        help="Path to save the test report (default: e2e/e2e_enhanced_report.md)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up test data after running",
    )
    parser.add_argument(
        "--ttyd-port",
        type=int,
        default=7681,
        help="Port for ttyd server (default: 7681)",
    )
    parser.add_argument(
        "--fallback-mode",
        action="store_true",
        help="Force fallback to standard testing (no ttyd/Playwright)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install enhanced dependencies and exit",
    )

    args = parser.parse_args()

    # Handle dependency installation
    if args.install_deps:
        print("Installing enhanced E2E testing dependencies...")
        import subprocess

        try:
            subprocess.run(  # noqa: S603
                ["uv", "sync", "--group", "e2e-enhanced"],  # noqa: S607
                check=True,
                cwd=Path.cwd(),
            )
            print("✓ Enhanced dependencies installed successfully!")
            print("You can now run enhanced tests without --fallback-mode")
            return
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            sys.exit(1)

    # Determine testing mode
    use_enhanced = ENHANCED_TESTING_AVAILABLE and not args.fallback_mode

    if not use_enhanced:
        if not ENHANCED_TESTING_AVAILABLE and not args.fallback_mode:
            print("⚠️  Enhanced testing dependencies not available.")
            print(f"Import error: {IMPORT_ERROR}")
            print("Run with --install-deps to install them, or use --fallback-mode for standard testing.")
            print("Falling back to standard E2E testing...\n")
        elif args.fallback_mode:
            print("🔄 Fallback mode enabled - using standard E2E testing...\n")

    # Create appropriate tester
    if use_enhanced:
        print("🚀 Starting enhanced E2E testing with ttyd + Playwright")
        print(f"ttyd port: {args.ttyd_port}")
        tester = TtydIntegratedOboyuTester(oboyu_path=args.oboyu_path, ttyd_port=args.ttyd_port)
        is_enhanced = True
    else:
        print("📋 Starting standard E2E testing")
        tester = OboyuE2EDisplayTester(oboyu_path=args.oboyu_path)
        is_enhanced = False

    tester.setup()

    try:
        results = {}

        # Define test mapping based on tester type
        if is_enhanced:
            test_mapping = {
                "basic": ("basic_cli_display", tester.test_basic_cli_display),
                "indexing": ("indexing_progress_display", tester.test_indexing_progress_display),
                "search": ("search_result_display", tester.test_search_result_display),
                "error": ("error_display", tester.test_error_display),
                "visual-cli": ("visual_cli_commands", lambda: asyncio.run(tester.test_visual_cli_commands())),
                "interactive-progress": ("interactive_indexing_progress", lambda: asyncio.run(tester.test_interactive_indexing_progress())),
                "enhanced-basic": ("enhanced_basic_cli_display", tester.test_enhanced_basic_cli_display),
                "enhanced-indexing": ("enhanced_indexing_progress_display", tester.test_enhanced_indexing_progress_display),
            }
        else:
            test_mapping = {
                "basic": ("basic_cli_display", tester.test_basic_cli_display),
                "indexing": ("indexing_progress_display", tester.test_indexing_progress_display),
                "search": ("search_result_display", tester.test_search_result_display),
                "error": ("error_display", tester.test_error_display),
            }

        # Determine which tests to run
        if args.test == "all":
            if is_enhanced:
                # For enhanced mode, run enhanced versions of core tests
                tests_to_run = [
                    ("enhanced_basic_cli_display", tester.test_enhanced_basic_cli_display),
                    ("enhanced_indexing_progress_display", tester.test_enhanced_indexing_progress_display),
                    ("search_result_display", tester.test_search_result_display),
                    ("error_display", tester.test_error_display),
                ]
            else:
                tests_to_run = list(test_mapping.values())
        else:
            if args.test in test_mapping:
                test_name, test_func = test_mapping[args.test]
                tests_to_run = [(test_name, test_func)]
            else:
                if not is_enhanced and args.test in ["visual-cli", "interactive-progress", "enhanced-basic", "enhanced-indexing"]:
                    print(f"✗ Test '{args.test}' requires enhanced mode. Install dependencies or remove --fallback-mode.")
                    sys.exit(1)
                else:
                    print(f"✗ Unknown test: {args.test}")
                    sys.exit(1)

        print(f"Running E2E tests with oboyu command: {args.oboyu_path}")
        print(f"Test selection: {args.test}")
        print(f"Mode: {'Enhanced (ttyd + Playwright)' if is_enhanced else 'Standard'}")
        print("-" * 60)

        failed_tests = []

        # Run selected tests
        for test_name, test_func in tests_to_run:
            print(f"\nRunning {test_name}...")
            try:
                result = test_func()
                results[test_name] = result

                # Enhanced failure detection
                has_critical_issue = False

                if isinstance(result, dict):
                    # Check traditional analysis results
                    if "traditional_analysis" in result:
                        trad_result = result["traditional_analysis"]
                        if isinstance(trad_result, dict) and "result" in trad_result:
                            result_text = trad_result["result"].lower()
                            critical_keywords = ["critical issue", "fix suggestion", "needs to be fixed", "not working"]
                            has_critical_issue = any(keyword in result_text for keyword in critical_keywords)

                    # Check standard result format
                    elif "result" in result:
                        result_text = result["result"].lower()
                        critical_keywords = ["critical issue", "fix suggestion", "needs to be fixed", "not working"]
                        has_critical_issue = any(keyword in result_text for keyword in critical_keywords)

                    # Check for failure emojis
                    result_str = str(result)
                    has_failure_emoji = "❌" in result_str

                    if has_critical_issue or has_failure_emoji:
                        has_critical_issue = True

                if has_critical_issue:
                    failed_tests.append(test_name)
                    print(f"✗ {test_name} failed: Issues detected")
                else:
                    print(f"✓ {test_name} completed")

            except Exception as e:
                results[test_name] = {"error": str(e)}
                failed_tests.append(test_name)
                print(f"✗ {test_name} failed: {e}")

        # Generate and save report
        if results:
            print("\nGenerating test report...")

            if is_enhanced and hasattr(tester, "generate_enhanced_report"):
                report = tester.generate_enhanced_report(results)
            else:
                report = tester.generate_report(results)

            report_path = Path(args.report)
            report_path.write_text(report, encoding="utf-8")
            print(f"Report saved to: {report_path}")

            # Add visual validation summary for enhanced mode
            if is_enhanced and hasattr(tester, "screenshots_dir") and tester.screenshots_dir:
                print("\nGenerating visual assets summary...")
                manifest = VisualValidationHelper.create_screenshot_manifest(tester.screenshots_dir)
                if manifest.get("exists", False):
                    print(f"📸 Screenshots: {manifest['total_screenshots']} files")
                    print(f"💾 Total size: {manifest['total_size_bytes'] / (1024 * 1024):.2f} MB")
                    for category, info in manifest["categories"].items():
                        print(f"  - {category.replace('_', ' ').title()}: {info['count']} files")

            # Print summary
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            total_tests = len(results)
            failed_count = len(failed_tests)
            passed_tests = total_tests - failed_count

            print(f"Total tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {failed_count}")
            print(f"Mode: {'Enhanced' if is_enhanced else 'Standard'}")

            if failed_count > 0:
                print("\nFailed tests:")
                for test_name in failed_tests:
                    result = results[test_name]
                    if isinstance(result, dict) and "error" in result:
                        print(f"  - {test_name}: {result['error']}")
                    else:
                        print(f"  - {test_name}: Issues detected")

            # Calculate total cost
            total_cost = sum(r.get("cost_usd", 0) for r in results.values() if isinstance(r, dict))
            if total_cost > 0:
                print(f"\nTotal API cost: ${total_cost:.4f}")

            # Enhanced mode specific output
            if is_enhanced:
                print("\n🎯 Enhanced testing completed")
                if hasattr(tester, "screenshots_dir") and tester.screenshots_dir:
                    print(f"📁 Visual assets: {tester.screenshots_dir}")

    finally:
        if not args.no_cleanup:
            tester.teardown()
        else:
            print(f"\nTest data preserved at: {tester.test_data_dir}")
            if is_enhanced and hasattr(tester, "screenshots_dir") and tester.screenshots_dir:
                print(f"Screenshots preserved at: {tester.screenshots_dir}")


if __name__ == "__main__":
    main()
