"""Health monitoring CLI commands."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import click

from oboyu.cli.services.command_services import CommandServices
from oboyu.events import IndexEventBus
from oboyu.events.handlers import IndexMetricsCollector
from oboyu.events.store import EventStore
from oboyu.monitoring.health import IndexHealthMonitor


class HealthCommand:
    """Health monitoring command implementation."""
    
    def __init__(self, services: CommandServices) -> None:
        """Initialize health command with services."""
        self.services = services
        self.event_bus = IndexEventBus()
        self.health_monitor = IndexHealthMonitor()
        self.metrics_collector = IndexMetricsCollector()
        
        # Subscribe handlers to event bus
        self.event_bus.subscribe_global(self.health_monitor)
        self.event_bus.subscribe_global(self.metrics_collector)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        try:
            # Get health report
            health_report = self.health_monitor.get_health_report()
            
            # Get metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            return {
                "status": health_report.overall_status.value,
                "corruption_detected": health_report.corruption_detected,
                "last_successful_index": health_report.last_successful_index.isoformat() if health_report.last_successful_index else None,
                "success_rate": health_report.success_rate,
                "total_operations": health_report.total_operations,
                "issues": health_report.issues,
                "metrics": metrics_summary,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_recent_operations(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent operations for debugging."""
        try:
            health_report = self.health_monitor.get_health_report()
            
            recent_ops = []
            for op in health_report.recent_operations[-limit:]:
                recent_ops.append({
                    "operation_id": op.operation_id,
                    "operation_type": op.operation_type,
                    "timestamp": op.timestamp.isoformat(),
                    "success": op.success,
                    "duration_seconds": op.duration_seconds,
                    "error_type": op.error_type
                })
            
            return {
                "recent_operations": recent_ops,
                "total_count": len(health_report.recent_operations)
            }
        except Exception as e:
            return {
                "error": str(e),
                "recent_operations": []
            }


@click.group()
def health() -> None:
    """Health monitoring and diagnostics commands."""
    pass


@health.command()
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']),
              help='Output format')
@click.pass_context
def status(ctx: click.Context, output_format: str) -> None:
    """Show current system health status."""
    services = ctx.obj
    health_cmd = HealthCommand(services)
    
    try:
        health_status = health_cmd.get_health_status()
        
        if output_format == 'json':
            click.echo(json.dumps(health_status, indent=2))
        else:
            # Table format
            click.echo(f"System Health Status: {health_status['status'].upper()}")
            click.echo(f"Timestamp: {health_status['timestamp']}")
            click.echo()
            
            if health_status['corruption_detected']:
                click.echo("⚠️  Corruption detected!")
            
            if health_status['last_successful_index']:
                click.echo(f"Last successful index: {health_status['last_successful_index']}")
            
            click.echo(f"Success rate: {health_status['success_rate']:.2%}")
            click.echo(f"Total operations: {health_status['total_operations']}")
            
            if health_status['issues']:
                click.echo()
                click.echo("Issues:")
                for issue in health_status['issues']:
                    click.echo(f"  • {issue}")
            
            if 'metrics' in health_status:
                metrics = health_status['metrics']
                click.echo()
                click.echo("Performance Metrics:")
                click.echo(f"  Total operations: {metrics['total_operations']}")
                click.echo(f"  Average indexing duration: {metrics['average_indexing_duration']:.2f}s")
                click.echo(f"  Documents processed: {metrics['total_documents_processed']}")
                click.echo(f"  Chunks created: {metrics['total_chunks_created']}")
                
    except Exception as e:
        services.console_manager.print_error(f"Health check failed: {e}")
        ctx.exit(1)


@health.command()
@click.option('--limit', default=20, help='Number of recent operations to show')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']),
              help='Output format')
@click.pass_context
def operations(ctx: click.Context, limit: int, output_format: str) -> None:
    """Show recent operations for debugging."""
    services = ctx.obj
    health_cmd = HealthCommand(services)
    
    try:
        operations_data = health_cmd.get_recent_operations(limit)
        
        if output_format == 'json':
            click.echo(json.dumps(operations_data, indent=2))
        else:
            # Table format
            recent_ops = operations_data['recent_operations']
            
            if not recent_ops:
                click.echo("No recent operations found.")
                return
            
            click.echo(f"Recent Operations (showing last {len(recent_ops)}):")
            click.echo()
            
            # Table headers
            click.echo(f"{'Timestamp':<20} {'Operation':<20} {'Status':<10} {'Duration':<10} {'ID':<36}")
            click.echo("-" * 96)
            
            for op in recent_ops:
                timestamp = datetime.fromisoformat(op['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                status = "✅ Success" if op['success'] else f"❌ {op['error_type'] or 'Failed'}"
                duration = f"{op['duration_seconds']:.2f}s" if op['duration_seconds'] > 0 else "N/A"
                
                click.echo(f"{timestamp:<20} {op['operation_type']:<20} {status:<10} {duration:<10} {op['operation_id']}")
                
    except Exception as e:
        services.console_manager.print_error(f"Failed to get operations: {e}")
        ctx.exit(1)


@health.command()
@click.option('--event-db', help='Path to event database (optional)')
@click.option('--hours', default=24, help='Hours of events to show')
@click.option('--event-type', help='Filter by event type')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']),
              help='Output format')
@click.pass_context
def events(ctx: click.Context, event_db: Optional[str], hours: int, event_type: Optional[str], output_format: str) -> None:
    """Show recent events for debugging."""
    services = ctx.obj
    
    try:
        # Determine event database path
        if event_db:
            event_db_path = Path(event_db)
        else:
            # Use default path based on main database
            config = services.config_service.load_config()
            db_path = services.database_path_resolver.resolve_path(config)
            event_db_path = db_path.parent / "events.db"
        
        if not event_db_path.exists():
            click.echo("No event database found. Events may not be enabled.")
            ctx.exit(1)
        
        # Create event store and query events
        event_store = EventStore(event_db_path)
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        events_data = event_store.get_events_by_time_range(start_time, end_time, event_type)
        
        if output_format == 'json':
            click.echo(json.dumps(events_data, indent=2))
        else:
            # Table format
            if not events_data:
                click.echo(f"No events found in the last {hours} hours.")
                return
            
            click.echo(f"Events from last {hours} hours:")
            if event_type:
                click.echo(f"Filtered by type: {event_type}")
            click.echo()
            
            # Table headers
            click.echo(f"{'Timestamp':<20} {'Event Type':<25} {'Operation ID':<36}")
            click.echo("-" * 81)
            
            for event in events_data[-50:]:  # Show last 50 events
                timestamp = datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                click.echo(f"{timestamp:<20} {event['event_type']:<25} {event['operation_id']:<36}")
                
        click.echo(f"\nTotal events: {len(events_data)}")
        
    except Exception as e:
        services.console_manager.print_error(f"Failed to get events: {e}")
        ctx.exit(1)


@health.command()
@click.argument('operation_id')
@click.option('--event-db', help='Path to event database (optional)')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']),
              help='Output format')
@click.pass_context
def timeline(ctx: click.Context, operation_id: str, event_db: Optional[str], output_format: str) -> None:
    """Show timeline of events for a specific operation."""
    services = ctx.obj
    
    try:
        # Determine event database path
        if event_db:
            event_db_path = Path(event_db)
        else:
            # Use default path based on main database
            config = services.config_service.load_config()
            db_path = services.database_path_resolver.resolve_path(config)
            event_db_path = db_path.parent / "events.db"
        
        if not event_db_path.exists():
            click.echo("No event database found. Events may not be enabled.")
            ctx.exit(1)
        
        # Create event store and get operation timeline
        event_store = EventStore(event_db_path)
        timeline_events = event_store.get_operation_timeline(operation_id)
        
        if output_format == 'json':
            click.echo(json.dumps(timeline_events, indent=2))
        else:
            # Table format
            if not timeline_events:
                click.echo(f"No events found for operation: {operation_id}")
                return
            
            click.echo(f"Operation Timeline: {operation_id}")
            click.echo()
            
            for i, event in enumerate(timeline_events):
                timestamp = datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                click.echo(f"{i+1:2d}. {timestamp} - {event['event_type']}")
                
                # Show relevant event data
                if event['event_type'] == 'indexing_started':
                    click.echo(f"    Documents: {event.get('document_count', 0)}, Size: {event.get('total_size_bytes', 0)} bytes")
                elif event['event_type'] == 'indexing_completed':
                    click.echo(f"    Chunks: {event.get('chunks_created', 0)}, Duration: {event.get('duration_seconds', 0):.2f}s")
                elif event['event_type'] == 'indexing_failed':
                    click.echo(f"    Error: {event.get('error', 'Unknown')}")
                elif event['event_type'] == 'database_cleared':
                    click.echo(f"    Records deleted: {event.get('records_deleted', 0)}")
                
                click.echo()
        
    except Exception as e:
        services.console_manager.print_error(f"Failed to get operation timeline: {e}")
        ctx.exit(1)
