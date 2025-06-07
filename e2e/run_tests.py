#!/usr/bin/env python3
"""Run E2E display tests for Oboyu using Claude Code SDK."""

import argparse
from pathlib import Path

from display_tester import OboyuE2EDisplayTester


def main() -> None:
    """Run E2E display tests with various options."""
    parser = argparse.ArgumentParser(
        description="Run E2E display tests for Oboyu using Claude Code SDK"
    )
    parser.add_argument(
        "--test",
        choices=[
            "all",
            "basic",
            "indexing",
            "search",
            "error",
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
        default="e2e_display_report.md",
        help="Path to save the test report (default: e2e_display_report.md)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up test data after running",
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = OboyuE2EDisplayTester(oboyu_path=args.oboyu_path)
    tester.setup()
    
    try:
        results = {}
        
        # Define test mapping
        test_mapping = {
            "basic": ("basic_cli_display", tester.test_basic_cli_display),
            "indexing": ("indexing_progress_display", tester.test_indexing_progress_display),
            "search": ("search_result_display", tester.test_search_result_display),
            "error": ("error_display", tester.test_error_display),
        }
        
        # Determine which tests to run
        if args.test == "all":
            tests_to_run = list(test_mapping.values())
        else:
            test_name, test_func = test_mapping[args.test]
            tests_to_run = [(test_name, test_func)]
        
        print(f"Running E2E display tests with oboyu command: {args.oboyu_path}")
        print(f"Test selection: {args.test}")
        print("-" * 60)
        
        failed_tests = []
        
        # Run selected tests
        for test_name, test_func in tests_to_run:
            print(f"\nRunning {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                
                # Check if this test found critical issues
                result_text = result.get("result", "").lower()
                critical_keywords = ["critical issue", "fix suggestion", "needs to be fixed", "not working"]
                has_critical_issue = any(keyword in result_text for keyword in critical_keywords)
                has_failure_emoji = "❌" in result.get("result", "")
                
                if has_critical_issue or has_failure_emoji:
                    failed_tests.append(test_name)
                    print(f"✗ {test_name} failed: Display issues detected")
                else:
                    print(f"✓ {test_name} completed")
                    
            except Exception as e:
                results[test_name] = {"error": str(e)}
                failed_tests.append(test_name)
                print(f"✗ {test_name} failed: {e}")
        
        # Generate and save report
        if results:
            print("\nGenerating test report...")
            report = tester.generate_report(results)
            report_path = Path(args.report)
            report_path.write_text(report, encoding="utf-8")
            print(f"Report saved to: {report_path}")
            
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
            
            if failed_count > 0:
                print("\nFailed tests:")
                for test_name in failed_tests:
                    result = results[test_name]
                    if "error" in result:
                        print(f"  - {test_name}: {result['error']}")
                    else:
                        print(f"  - {test_name}: Display issues detected")
            
            # Calculate total cost
            total_cost = sum(
                r.get("cost_usd", 0) for r in results.values() if isinstance(r, dict)
            )
            if total_cost > 0:
                print(f"\nTotal API cost: ${total_cost:.4f}")
        
    finally:
        if not args.no_cleanup:
            tester.teardown()
        else:
            print(f"\nTest data preserved at: {tester.test_data_dir}")


if __name__ == "__main__":
    main()
