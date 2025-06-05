"""Health monitoring CLI app for Oboyu.

This module provides the command-line interface for health monitoring and diagnostics.
"""

from typing import Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(
    help="Health monitoring and diagnostics",
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

# Note: health is a Click group that cannot be directly added to Typer
# The health commands are available separately via the click group


@app.command()
def check(
    ctx: typer.Context,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format"),
    ] = "table",
) -> None:
    """Quick health check of the system."""
    # This delegates to the health status command
    from oboyu.cli.commands.health import status
    ctx.invoke(status, output_format=format)


@app.command()
def events(
    ctx: typer.Context,
    event_db: Annotated[
        Optional[str],
        typer.Option("--event-db", help="Path to event database"),
    ] = None,
    hours: Annotated[
        int,
        typer.Option("--hours", help="Hours of events to show"),
    ] = 24,
    event_type: Annotated[
        Optional[str],
        typer.Option("--event-type", help="Filter by event type"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format"),
    ] = "table",
) -> None:
    """Show recent events for debugging."""
    # This delegates to the health events command
    from oboyu.cli.commands.health import events as events_cmd
    ctx.invoke(events_cmd, event_db=event_db, hours=hours, event_type=event_type, output_format=format)


@app.command()
def timeline(
    ctx: typer.Context,
    operation_id: Annotated[
        str,
        typer.Argument(help="Operation ID to show timeline for"),
    ],
    event_db: Annotated[
        Optional[str],
        typer.Option("--event-db", help="Path to event database"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format"),
    ] = "table",
) -> None:
    """Show timeline of events for a specific operation."""
    # This delegates to the health timeline command
    from oboyu.cli.commands.health import timeline as timeline_cmd
    ctx.invoke(timeline_cmd, operation_id=operation_id, event_db=event_db, output_format=format)


@app.command()
def operations(
    ctx: typer.Context,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Number of recent operations to show"),
    ] = 20,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format"),
    ] = "table",
) -> None:
    """Show recent operations for debugging."""
    # This delegates to the health operations command
    from oboyu.cli.commands.health import operations as operations_cmd
    ctx.invoke(operations_cmd, limit=limit, output_format=format)
