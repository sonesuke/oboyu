"""Common command patterns and decorators for CLI commands.

This module provides decorators and utilities that encapsulate
common command patterns to reduce code duplication.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

import typer

from oboyu.cli.base import BaseCommand

F = TypeVar('F', bound=Callable[..., Any])


def with_base_command(func: F) -> F:
    """Decorator that automatically creates a BaseCommand instance.

    This decorator extracts the typer.Context from the function arguments
    and creates a BaseCommand instance, passing it as the first argument
    to the wrapped function.

    Args:
        func: Function to decorate, must have ctx: typer.Context as first parameter

    Returns:
        Decorated function that receives BaseCommand as first argument

    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract context from the first argument (should be typer.Context)
        if not args or not isinstance(args[0], typer.Context):
            raise ValueError("Function must have typer.Context as first parameter")
        
        ctx = args[0]
        base_command = BaseCommand(ctx)
        
        # Call the original function with BaseCommand and remaining args
        return func(base_command, *args[1:], **kwargs)
    
    return wrapper  # type: ignore


def with_indexer_initialization(
    show_progress: bool = True,
    show_model_loading: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator for commands that need indexer initialization.

    Args:
        show_progress: Whether to show initialization progress
        show_model_loading: Whether to show model loading details
        config_overrides: Optional configuration overrides

    Returns:
        Decorator function

    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(base_command: BaseCommand, *args: Any, **kwargs: Any) -> Any:
            # Extract db_path from kwargs if present
            db_path = kwargs.get('db_path')
            
            # Create indexer config with any overrides
            overrides = config_overrides or {}
            config = base_command.create_indexer_config(db_path=db_path, **overrides)
            
            # Show database path
            base_command.print_database_path(str(config.db_path))
            
            # Create indexer with progress tracking
            with base_command.logger.live_display():
                indexer = base_command.create_indexer(
                    config,
                    show_progress=show_progress,
                    show_model_loading=show_model_loading,
                )
                
                try:
                    # Call the original function with indexer
                    return func(base_command, indexer, *args, **kwargs)
                finally:
                    # Ensure indexer is properly closed
                    indexer.close()
                    
        return wrapper  # type: ignore
    return decorator


def with_database_confirmation(operation_name: str) -> Callable[[F], F]:
    """Decorator for commands that modify the database.

    Args:
        operation_name: Description of the operation for confirmation

    Returns:
        Decorator function

    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(base_command: BaseCommand, *args: Any, **kwargs: Any) -> Any:
            # Extract force and db_path from kwargs if present
            force = kwargs.get('force', False)
            db_path = kwargs.get('db_path')
            
            # Get database path for confirmation
            config = base_command.create_indexer_config(db_path=db_path)
            resolved_db_path = str(config.db_path)
            
            # Confirm operation
            if not base_command.confirm_database_operation(
                operation_name, force, resolved_db_path
            ):
                return
                
            # Call the original function
            return func(base_command, *args, **kwargs)
            
        return wrapper  # type: ignore
    return decorator


class StandardOptions:
    """Standard option bundles for common command patterns."""
    
    @staticmethod
    def basic_command_options() -> Dict[str, Any]:
        """Basic options for most commands.
        
        Returns:
            Dictionary of basic option definitions

        """
        from oboyu.cli.common_options import DatabasePathOption, VerboseOption
        
        return {
            "db_path": DatabasePathOption,
            "verbose": VerboseOption,
        }
    
    @staticmethod
    def database_modification_options() -> Dict[str, Any]:
        """Options for commands that modify the database.
        
        Returns:
            Dictionary of database modification option definitions

        """
        from oboyu.cli.common_options import DatabasePathOption, ForceOption
        
        return {
            "db_path": DatabasePathOption,
            "force": ForceOption,
        }


def create_clear_command_handler() -> Callable[[typer.Context, Optional[str], bool], None]:
    """Create a standardized clear command handler.
    
    This function returns a command handler that can be used for
    clear commands across different command groups.
    
    Returns:
        Clear command handler function

    """
    @with_base_command
    def clear_handler(
        base_command: BaseCommand,
        db_path: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Clear all data from the index database."""
        base_command.handle_clear_operation(db_path=db_path, force=force)
        
    return clear_handler
