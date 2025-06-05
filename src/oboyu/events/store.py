"""Event storage for replay and debugging functionality."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .events import IndexEvent

logger = logging.getLogger(__name__)


class EventStore:
    """Stores events for replay and debugging purposes."""
    
    def __init__(self, db_path: Path) -> None:
        """Initialize the event store.
        
        Args:
            db_path: Path to the SQLite database file

        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the event storage database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient querying
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_event_type
                ON events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_operation_id
                ON events(operation_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp)
            """)
            
            conn.commit()
    
    def store_event(self, event: IndexEvent) -> None:
        """Store an event in the database.
        
        Args:
            event: The event to store

        """
        try:
            event_data = json.dumps(event.to_dict(), default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO events (event_type, operation_id, timestamp, event_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    event.event_type,
                    event.operation_id,
                    event.timestamp.isoformat(),
                    event_data
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store event {event.event_type}: {e}")
    
    def get_events_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[str] = None
    ) -> List[dict]:
        """Retrieve events within a time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            event_type: Optional filter by event type
            
        Returns:
            List of event dictionaries

        """
        query = """
            SELECT event_type, operation_id, timestamp, event_data
            FROM events
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                events = []
                
                for row in cursor.fetchall():
                    event_dict = json.loads(row[3])
                    events.append(event_dict)
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []
    
    def get_events_by_operation(self, operation_id: str) -> List[dict]:
        """Get all events for a specific operation.
        
        Args:
            operation_id: The operation identifier
            
        Returns:
            List of event dictionaries for the operation

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT event_type, operation_id, timestamp, event_data
                    FROM events
                    WHERE operation_id = ?
                    ORDER BY timestamp
                """, (operation_id,))
                
                events = []
                for row in cursor.fetchall():
                    event_dict = json.loads(row[3])
                    events.append(event_dict)
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to retrieve events for operation {operation_id}: {e}")
            return []
    
    def get_recent_events(self, limit: int = 100, event_type: Optional[str] = None) -> List[dict]:
        """Get the most recent events.
        
        Args:
            limit: Maximum number of events to return
            event_type: Optional filter by event type
            
        Returns:
            List of recent event dictionaries

        """
        query = """
            SELECT event_type, operation_id, timestamp, event_data
            FROM events
        """
        params = []
        
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(str(limit))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                events = []
                
                for row in cursor.fetchall():
                    event_dict = json.loads(row[3])
                    events.append(event_dict)
                
                return events
                
        except Exception as e:
            logger.error(f"Failed to retrieve recent events: {e}")
            return []
    
    def get_operation_timeline(self, operation_id: str) -> List[dict]:
        """Get a chronological timeline of events for an operation.
        
        Args:
            operation_id: The operation identifier
            
        Returns:
            Chronologically ordered list of events for the operation

        """
        return self.get_events_by_operation(operation_id)
    
    def cleanup_old_events(self, retention_days: int = 30) -> int:
        """Clean up events older than the retention period.
        
        Args:
            retention_days: Number of days to retain events
            
        Returns:
            Number of events deleted

        """
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - retention_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old events")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0
    
    def get_event_statistics(self) -> dict:
        """Get statistics about stored events.
        
        Returns:
            Dictionary with event statistics

        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total events
                cursor = conn.execute("SELECT COUNT(*) FROM events")
                total_events = cursor.fetchone()[0]
                
                # Events by type
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*)
                    FROM events
                    GROUP BY event_type
                    ORDER BY COUNT(*) DESC
                """)
                events_by_type = dict(cursor.fetchall())
                
                # Date range
                cursor = conn.execute("""
                    SELECT MIN(timestamp), MAX(timestamp)
                    FROM events
                """)
                date_range = cursor.fetchone()
                
                return {
                    "total_events": total_events,
                    "events_by_type": events_by_type,
                    "earliest_event": date_range[0] if date_range[0] else None,
                    "latest_event": date_range[1] if date_range[1] else None,
                }
                
        except Exception as e:
            logger.error(f"Failed to get event statistics: {e}")
            return {
                "total_events": 0,
                "events_by_type": {},
                "earliest_event": None,
                "latest_event": None,
            }
