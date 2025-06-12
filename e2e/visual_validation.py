"""Visual validation helpers for enhanced E2E testing."""

from pathlib import Path
from typing import Any, Dict, List


class VisualValidationHelper:
    """Helper class for visual validation and screenshot analysis."""

    @staticmethod
    def validate_terminal_rendering(screenshot_path: Path) -> Dict[str, Any]:
        """Validate terminal rendering quality from screenshot.

        This is a placeholder for future computer vision-based validation.
        Currently returns basic file validation.

        Args:
            screenshot_path: Path to the screenshot file

        Returns:
            Validation results dictionary

        """
        if not screenshot_path.exists():
            return {"valid": False, "error": f"Screenshot file not found: {screenshot_path}", "checks": {}}

        # Basic file validation
        file_size = screenshot_path.stat().st_size
        if file_size == 0:
            return {"valid": False, "error": "Screenshot file is empty", "checks": {"file_size": file_size}}

        return {"valid": True, "checks": {"file_exists": True, "file_size": file_size, "file_extension": screenshot_path.suffix, "readable": True}}

    @staticmethod
    def compare_screenshots(before_path: Path, after_path: Path) -> Dict[str, Any]:
        """Compare two screenshots for visual differences.

        This is a placeholder for future image comparison functionality.

        Args:
            before_path: Path to the "before" screenshot
            after_path: Path to the "after" screenshot

        Returns:
            Comparison results dictionary

        """
        before_valid = VisualValidationHelper.validate_terminal_rendering(before_path)
        after_valid = VisualValidationHelper.validate_terminal_rendering(after_path)

        if not before_valid["valid"] or not after_valid["valid"]:
            return {"comparable": False, "error": "One or both screenshots are invalid", "before_validation": before_valid, "after_validation": after_valid}

        # Basic file size comparison (placeholder for real image diff)
        before_size = before_path.stat().st_size
        after_size = after_path.stat().st_size
        size_diff = abs(before_size - after_size)

        return {
            "comparable": True,
            "size_difference": size_diff,
            "size_change_percentage": (size_diff / before_size) * 100 if before_size > 0 else 0,
            "before_size": before_size,
            "after_size": after_size,
            "identical_size": before_size == after_size,
        }

    @staticmethod
    def analyze_progress_screenshots(screenshot_paths: List[Path]) -> Dict[str, Any]:
        """Analyze a sequence of screenshots for progress indication.

        Args:
            screenshot_paths: List of screenshot paths in chronological order

        Returns:
            Progress analysis results

        """
        if not screenshot_paths:
            return {"valid_sequence": False, "error": "No screenshots provided"}

        # Validate all screenshots
        validations = []
        for i, path in enumerate(screenshot_paths):
            validation = VisualValidationHelper.validate_terminal_rendering(path)
            validation["index"] = i
            validation["timestamp"] = path.stem.split("_")[-1] if "_" in path.stem else "unknown"
            validations.append(validation)

        valid_count = sum(1 for v in validations if v["valid"])

        return {
            "valid_sequence": valid_count == len(screenshot_paths),
            "total_screenshots": len(screenshot_paths),
            "valid_screenshots": valid_count,
            "invalid_screenshots": len(screenshot_paths) - valid_count,
            "validations": validations,
            "sequence_completeness": valid_count / len(screenshot_paths) if screenshot_paths else 0,
        }

    @staticmethod
    def generate_visual_context_for_claude(results: Dict[str, Any]) -> str:
        """Generate visual context description for Claude analysis.

        Args:
            results: Test results containing screenshot information

        Returns:
            Formatted context string for Claude prompts

        """
        context_parts = []

        if "screenshots" in results:
            screenshot_count = len(results["screenshots"])
            context_parts.append(f"📸 {screenshot_count} screenshots captured during testing")

            # Group screenshots by category if possible
            categories = {}
            for screenshot in results["screenshots"]:
                path = Path(screenshot)
                if path.parent.name in ["cli_commands", "progress_monitoring", "mcp_integration"]:
                    category = path.parent.name
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(path.name)

            if categories:
                context_parts.append("\nScreenshot categories:")
                for category, files in categories.items():
                    category_name = category.replace("_", " ").title()
                    context_parts.append(f"  - {category_name}: {len(files)} screenshots")

        if "commands_tested" in results:
            commands = ", ".join(results["commands_tested"])
            context_parts.append(f"\n🔧 Commands tested: {commands}")

        if "progress_stages" in results:
            stages = ", ".join(results["progress_stages"])
            context_parts.append(f"\n⏱️  Progress stages: {stages}")

        if "timing_analysis" in results:
            timing = results["timing_analysis"]
            if "total_duration" in timing:
                context_parts.append(f"\n⏰ Duration: {timing['total_duration']:.2f} seconds")

        return "\n".join(context_parts)

    @staticmethod
    def create_screenshot_manifest(screenshots_dir: Path) -> Dict[str, Any]:
        """Create a manifest of all screenshots for reporting.

        Args:
            screenshots_dir: Directory containing screenshots

        Returns:
            Screenshot manifest dictionary

        """
        if not screenshots_dir.exists():
            return {"exists": False, "error": f"Screenshots directory not found: {screenshots_dir}"}

        manifest = {"exists": True, "directory": str(screenshots_dir), "categories": {}, "total_screenshots": 0, "total_size_bytes": 0}

        # Scan for screenshots by category
        for category_dir in screenshots_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                screenshots = list(category_dir.glob("*.png"))

                category_info = {"count": len(screenshots), "files": [], "total_size": 0}

                for screenshot in screenshots:
                    file_info = {
                        "name": screenshot.name,
                        "path": str(screenshot),
                        "size": screenshot.stat().st_size,
                        "timestamp": screenshot.stem.split("_")[-1] if "_" in screenshot.stem else "unknown",
                    }
                    category_info["files"].append(file_info)
                    category_info["total_size"] += file_info["size"]

                manifest["categories"][category_name] = category_info
                manifest["total_screenshots"] += category_info["count"]
                manifest["total_size_bytes"] += category_info["total_size"]

        return manifest

    @staticmethod
    def validate_claude_context_with_screenshots(context_files: List[str]) -> Dict[str, Any]:
        """Validate that screenshot files exist and are suitable for Claude context.

        Args:
            context_files: List of file paths to validate

        Returns:
            Validation results

        """
        if not context_files:
            return {"valid": True, "warning": "No context files provided", "files": []}

        file_validations = []
        total_size = 0

        for file_path in context_files:
            path = Path(file_path)
            validation = {"path": file_path, "exists": path.exists(), "readable": False, "size": 0, "type": "unknown"}

            if path.exists():
                try:
                    validation["size"] = path.stat().st_size
                    validation["readable"] = True
                    validation["type"] = "image" if path.suffix.lower() in [".png", ".jpg", ".jpeg"] else "other"
                    total_size += validation["size"]
                except Exception as e:
                    validation["error"] = str(e)

            file_validations.append(validation)

        valid_files = [v for v in file_validations if v.get("exists", False) and v.get("readable", False)]

        return {
            "valid": len(valid_files) == len(context_files),
            "total_files": len(context_files),
            "valid_files": len(valid_files),
            "invalid_files": len(context_files) - len(valid_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "files": file_validations,
        }
