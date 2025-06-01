"""Utility functions for Oboyu benchmarks."""

import json
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

import psutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class Timer:
    """Simple timer context manager for measuring execution time."""
    
    def __init__(self, name: str = "Operation") -> None:
        """Initialize timer with a name."""
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0
    
    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop the timer and calculate elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


class SystemMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self, sample_interval: float = 0.1) -> None:
        """Initialize system monitor."""
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        self.samples: List[Dict[str, float]] = []
        self._monitoring = False
        self._start_time = 0
    
    def start(self) -> None:
        """Start monitoring system resources."""
        self._monitoring = True
        self._start_time = time.time()
        self.samples = []
    
    def stop(self) -> None:
        """Stop monitoring system resources."""
        self._monitoring = False
    
    def sample(self) -> Dict[str, float]:
        """Take a single sample of system resources."""
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
            memory_info = self.process.memory_info()
            
            result = {
                "timestamp": time.time() - self._start_time,
                "cpu_percent": cpu_percent,
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
            }
            
            # io_counters is not available on all platforms (e.g., macOS)
            try:
                io_counters = self.process.io_counters()
                result["disk_io_read_mb"] = io_counters.read_bytes / 1024 / 1024
                result["disk_io_write_mb"] = io_counters.write_bytes / 1024 / 1024
            except AttributeError:
                pass
            
            return result
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from collected samples."""
        if not self.samples:
            return {}
        
        cpu_values = [s["cpu_percent"] for s in self.samples if "cpu_percent" in s]
        memory_values = [s["memory_usage_mb"] for s in self.samples if "memory_usage_mb" in s]
        
        summary = {}
        if cpu_values:
            summary["cpu_percent_avg"] = sum(cpu_values) / len(cpu_values)
            summary["cpu_percent_max"] = max(cpu_values)
        
        if memory_values:
            summary["memory_usage_mb_avg"] = sum(memory_values) / len(memory_values)
            summary["memory_usage_mb_max"] = max(memory_values)
        
        return summary


@contextmanager
def temporary_directory() -> Iterator[Path]:
    """Create a temporary directory that is cleaned up after use."""
    temp_dir = tempfile.mkdtemp(prefix="oboyu_bench_")
    temp_path = Path(temp_dir)
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}PB"


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {}
    
    sorted_values = sorted(values)
    n = len(values)
    
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / n,
        "median": sorted_values[n // 2] if n % 2 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
        "p95": sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
        "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
        "std": (sum((x - sum(values) / n) ** 2 for x in values) / n) ** 0.5 if n > 1 else 0
    }


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, filepath: Path, pretty: bool = True) -> None:
    """Save data as JSON to a file."""
    ensure_directory(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get current timestamp in configured format."""
    from bench.config import OUTPUT_CONFIG
    return datetime.now().strftime(OUTPUT_CONFIG["timestamp_format"])


def create_progress_bar(description: str) -> Progress:
    """Create a Rich progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    )


def print_header(title: str) -> None:
    """Print a formatted header."""
    console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
    console.print(f"[bold white]{title:^60}[/bold white]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    console.print("[yellow]" + "-" * len(title) + "[/yellow]")


def print_metric(name: str, value: Any, unit: str = "") -> None:
    """Print a formatted metric."""
    if isinstance(value, float):
        if unit == "time":
            value_str = format_time(value)
        elif unit == "size":
            value_str = format_size(int(value))
        else:
            value_str = f"{value:.2f}{' ' + unit if unit else ''}"
    else:
        value_str = f"{value}{' ' + unit if unit else ''}"
    
    console.print(f"  {name:<30} [cyan]{value_str}[/cyan]")


def check_oboyu_installation() -> bool:
    """Check if Oboyu is properly installed."""
    try:
        import oboyu  # noqa: F401
        return True
    except ImportError:
        console.print("[red]Error: Oboyu is not installed or not in PYTHONPATH[/red]")
        console.print("Please install Oboyu first: pip install -e .")
        return False


def get_python_info() -> Dict[str, str]:
    """Get Python environment information."""
    return {
        "version": sys.version,
        "executable": sys.executable,
        "platform": sys.platform
    }


def run_warmup(func: Any, warmup_runs: int = 1) -> None:
    """Run warmup iterations of a function."""
    for _ in range(warmup_runs):
        func()
