"""Simple logger for benchmark operations."""

from typing import Optional

from rich.console import Console


class BenchmarkLogger:
    """Simple logger for benchmark operations."""

    def __init__(self, verbose: bool = False, console: Optional[Console] = None) -> None:
        """Initialize benchmark logger.

        Args:
            verbose: Enable verbose output
            console: Rich console instance

        """
        self.verbose = verbose
        self.console = console or Console()

    def section(self, message: str) -> None:
        """Print a section header.

        Args:
            message: Section header message

        """
        self.console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        self.console.print(f"[bold cyan]{message}[/bold cyan]")
        self.console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    def info(self, message: str) -> None:
        """Print an info message.

        Args:
            message: Info message

        """
        self.console.print(f"[blue]ℹ[/blue]  {message}")

    def success(self, message: str) -> None:
        """Print a success message.

        Args:
            message: Success message

        """
        self.console.print(f"[green]✓[/green]  {message}")

    def warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: Warning message

        """
        self.console.print(f"[yellow]⚠[/yellow]  {message}")

    def error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: Error message

        """
        self.console.print(f"[red]✗[/red]  {message}")

    def debug(self, message: str) -> None:
        """Print a debug message if verbose mode is enabled.

        Args:
            message: Debug message

        """
        if self.verbose:
            self.console.print(f"[dim]DEBUG:[/dim] {message}")

    def table(self, headers: list, rows: list) -> None:
        """Print a formatted table.

        Args:
            headers: Table headers
            rows: Table rows

        """
        from rich.table import Table
        
        table = Table()
        for header in headers:
            table.add_column(header)
        
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        
        self.console.print(table)
