"""Console utilities for CLI commands."""

from typing import Optional

import typer
from rich.console import Console

from oboyu.cli.hierarchical_logger import create_hierarchical_logger


class ConsoleManager:
    """Manages console output and logging setup.
    
    This service encapsulates all console and logging functionality,
    providing a clean interface for user interaction.
    """

    def __init__(self) -> None:
        """Initialize the console manager."""
        self.console = Console()
        self.logger = create_hierarchical_logger(self.console)

    def print_database_path(self, db_path: str) -> None:
        """Print the database path being used.

        Args:
            db_path: Database path to display

        """
        self.console.print(f"Using database: {db_path}")

    def confirm_database_operation(
        self,
        operation_name: str,
        force: bool = False,
        db_path: Optional[str] = None,
    ) -> bool:
        """Confirm a database operation with the user.

        Args:
            operation_name: Name of the operation (e.g., "clear", "delete")
            force: Whether to skip confirmation
            db_path: Database path for display

        Returns:
            True if operation should proceed, False otherwise

        """
        if force:
            return True

        if db_path:
            self.print_database_path(db_path)

        self.console.print(f"Warning: This will {operation_name} the index database.")
        confirm = typer.confirm("Are you sure you want to continue?")
        if not confirm:
            self.console.print("Operation cancelled.")
            return False

        return True


# Legacy alias for backward compatibility
ConsoleManager = ConsoleManager
