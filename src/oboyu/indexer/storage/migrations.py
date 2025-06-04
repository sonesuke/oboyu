"""Database migration system for Oboyu indexer.

This module provides a robust database migration system that handles
schema evolution, version tracking, and safe migration execution.

Key features:
- Automatic migration detection and execution
- Rollback support for failed migrations
- Migration validation and checksum verification
- Safe migration execution with transaction support
- Schema version tracking and validation
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from duckdb import DuckDBPyConnection

from oboyu.indexer.storage.schema import SCHEMA_MIGRATIONS, DatabaseSchema, SchemaVersion

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Exception raised when migration operations fail."""

    pass


class MigrationManager:
    """Database migration management system.

    This class handles schema migrations, version tracking, and ensures
    safe database schema evolution over time.
    """

    def __init__(self, conn: DuckDBPyConnection, schema: DatabaseSchema) -> None:
        """Initialize migration manager.

        Args:
            conn: Database connection
            schema: Database schema manager

        """
        self.conn = conn
        self.schema = schema
        self._ensure_schema_version_table()

    def _ensure_schema_version_table(self) -> None:
        """Ensure schema version table exists."""
        try:
            schema_table = self.schema.get_schema_version_table()
            self.conn.execute(schema_table.sql)
        except Exception as e:
            logger.error(f"Failed to create schema version table: {e}")
            raise MigrationError(f"Schema version table creation failed: {e}") from e

    def get_current_version(self) -> Optional[str]:
        """Get the current database schema version.

        Returns:
            Current schema version or None if not set

        """
        try:
            result = self.conn.execute("""
                SELECT version
                FROM schema_version
                ORDER BY applied_at DESC
                LIMIT 1
            """).fetchone()

            return result[0] if result else None

        except Exception as e:
            logger.warning(f"Failed to get current schema version: {e}")
            return None

    def set_initial_version(self) -> None:
        """Set the initial schema version with conflict handling."""
        try:
            # Check if version already exists
            current_version = self.get_current_version()
            if current_version is not None:
                logger.debug(f"Schema version already set to: {current_version}")
                return
            
            version_data = self.schema.get_initial_schema_version_data()

            # Use INSERT OR IGNORE to handle concurrent access
            self.conn.execute(
                """
                INSERT OR IGNORE INTO schema_version (version, description, applied_at, migration_checksum)
                VALUES (?, ?, ?, ?)
            """,
                version_data,
            )

            logger.info(f"Set initial schema version: {version_data[0]}")

        except Exception as e:
            # Check if this is a concurrent access issue
            error_str = str(e).lower()
            if "duplicate" in error_str or "constraint violation" in error_str:
                logger.debug("Initial version already set by another process")
                return
                
            logger.error(f"Failed to set initial schema version: {e}")
            raise MigrationError(f"Initial version setup failed: {e}") from e

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get the history of applied migrations.

        Returns:
            List of migration records

        """
        try:
            results = self.conn.execute("""
                SELECT version, description, applied_at, migration_checksum
                FROM schema_version
                ORDER BY applied_at ASC
            """).fetchall()

            return [{"version": row[0], "description": row[1], "applied_at": row[2], "checksum": row[3]} for row in results]

        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []

    def get_pending_migrations(self) -> List[SchemaVersion]:
        """Get list of pending migrations that need to be applied.

        Returns:
            List of pending schema versions

        """
        current_version = self.get_current_version()

        if current_version is None:
            # No version set - return all migrations
            return list(SCHEMA_MIGRATIONS.values())

        # Find migrations newer than current version
        pending = []
        for version, migration in SCHEMA_MIGRATIONS.items():
            if self._is_version_newer(version, current_version):
                pending.append(migration)

        # Sort by version
        pending.sort(key=lambda x: x.version)
        return pending

    def _is_version_newer(self, version1: str, version2: str) -> bool:
        """Check if version1 is newer than version2.

        Args:
            version1: First version to compare
            version2: Second version to compare

        Returns:
            True if version1 is newer than version2

        """

        # Simple semantic version comparison
        def parse_version(version: str) -> Tuple[int, int, int]:
            parts = version.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))

        try:
            v1_parts = parse_version(version1)
            v2_parts = parse_version(version2)
            return v1_parts > v2_parts
        except (ValueError, IndexError):
            # Fallback to string comparison
            return version1 > version2

    def validate_migration(self, migration: SchemaVersion) -> bool:
        """Validate a migration before applying it.

        Args:
            migration: Migration to validate

        Returns:
            True if migration is valid

        """
        try:
            # Check if migration SQL is not empty
            if not migration.migration_sql:
                logger.error(f"Migration {migration.version} has no SQL statements")
                return False

            # Check if rollback SQL is provided
            if not migration.rollback_sql:
                logger.warning(f"Migration {migration.version} has no rollback SQL")

            # Validate SQL syntax (basic check)
            for sql in migration.migration_sql:
                if not sql.strip():
                    logger.error(f"Migration {migration.version} has empty SQL statement")
                    return False

            logger.debug(f"Migration {migration.version} validation passed")
            return True

        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False

    def apply_migration(self, migration: SchemaVersion) -> bool:
        """Apply a single migration.

        Args:
            migration: Migration to apply

        Returns:
            True if migration was applied successfully

        """
        if not self.validate_migration(migration):
            return False

        logger.info(f"Applying migration: {migration.version} - {migration.description}")

        try:
            # Start transaction
            self.conn.execute("BEGIN")

            # Apply migration SQL statements
            for sql_statement in migration.migration_sql:
                try:
                    self.conn.execute(sql_statement)
                    logger.debug(f"Executed: {sql_statement[:50]}...")
                except Exception as e:
                    logger.error(f"Failed to execute migration SQL: {e}")
                    raise

            # Calculate migration checksum
            migration_text = "\n".join(migration.migration_sql)
            checksum = hashlib.sha256(migration_text.encode()).hexdigest()

            # Record migration in schema_version table
            self.conn.execute(
                """
                INSERT INTO schema_version (version, description, applied_at, migration_checksum)
                VALUES (?, ?, ?, ?)
            """,
                (migration.version, migration.description, datetime.now().isoformat(), checksum),
            )

            # Commit transaction
            self.conn.execute("COMMIT")

            logger.info(f"Migration {migration.version} applied successfully")
            return True

        except Exception as e:
            logger.error(f"Migration {migration.version} failed: {e}")

            # Rollback transaction
            try:
                self.conn.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")

            return False

    def rollback_migration(self, migration: SchemaVersion) -> bool:
        """Rollback a migration.

        Args:
            migration: Migration to rollback

        Returns:
            True if rollback was successful

        """
        if not migration.rollback_sql:
            logger.error(f"No rollback SQL available for migration {migration.version}")
            return False

        logger.info(f"Rolling back migration: {migration.version}")

        try:
            # Start transaction
            self.conn.execute("BEGIN")

            # Apply rollback SQL statements
            for sql_statement in migration.rollback_sql:
                try:
                    self.conn.execute(sql_statement)
                    logger.debug(f"Rollback executed: {sql_statement[:50]}...")
                except Exception as e:
                    logger.error(f"Failed to execute rollback SQL: {e}")
                    raise

            # Remove migration record
            self.conn.execute(
                """
                DELETE FROM schema_version
                WHERE version = ?
            """,
                [migration.version],
            )

            # Commit transaction
            self.conn.execute("COMMIT")

            logger.info(f"Migration {migration.version} rolled back successfully")
            return True

        except Exception as e:
            logger.error(f"Rollback of migration {migration.version} failed: {e}")

            # Rollback transaction
            try:
                self.conn.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.error(f"Rollback transaction failed: {rollback_error}")

            return False

    def run_migrations(self, target_version: Optional[str] = None) -> bool:
        """Run all pending migrations up to target version.

        Args:
            target_version: Optional target version (latest if None)

        Returns:
            True if all migrations were applied successfully

        """
        # Get current version
        current_version = self.get_current_version()

        # If no current version, set initial version
        if current_version is None:
            self.set_initial_version()
            current_version = self.schema.CURRENT_VERSION

        # Get pending migrations
        pending_migrations = self.get_pending_migrations()

        if not pending_migrations:
            logger.debug("No pending migrations to apply")
            return True

        # Filter migrations by target version if specified
        if target_version:
            pending_migrations = [m for m in pending_migrations if not self._is_version_newer(m.version, target_version)]

        if not pending_migrations:
            logger.info(f"No migrations to apply up to version {target_version}")
            return True

        logger.info(f"Applying {len(pending_migrations)} migrations...")

        # Apply migrations one by one
        successful_migrations = []

        for migration in pending_migrations:
            if self.apply_migration(migration):
                successful_migrations.append(migration)
            else:
                logger.error(f"Migration {migration.version} failed, stopping migration process")
                return False

        logger.info(f"Successfully applied {len(successful_migrations)} migrations")
        return True

    def verify_migration_integrity(self) -> bool:
        """Verify the integrity of applied migrations.

        Returns:
            True if all migrations are valid

        """
        logger.info("Verifying migration integrity...")

        try:
            applied_migrations = self.get_migration_history()
            issues_found = False

            for applied in applied_migrations:
                version = applied["version"]
                stored_checksum = applied["checksum"]

                # Skip initial version (no checksum)
                if stored_checksum is None:
                    continue

                # Find corresponding migration definition
                if version not in SCHEMA_MIGRATIONS:
                    logger.warning(f"Applied migration {version} not found in migration definitions")
                    issues_found = True
                    continue

                # Verify checksum
                migration = SCHEMA_MIGRATIONS[version]
                migration_text = "\n".join(migration.migration_sql)
                expected_checksum = hashlib.sha256(migration_text.encode()).hexdigest()

                if stored_checksum != expected_checksum:
                    logger.error(f"Migration {version} checksum mismatch - migration may have been modified")
                    issues_found = True

            if issues_found:
                logger.warning("Migration integrity issues found")
                return False
            else:
                logger.info("Migration integrity verification passed")
                return True

        except Exception as e:
            logger.error(f"Migration integrity verification failed: {e}")
            return False

    def reset_migrations(self) -> bool:
        """Reset all migration history (dangerous operation).

        Returns:
            True if reset was successful

        """
        logger.warning("Resetting migration history - this is a dangerous operation!")

        try:
            self.conn.execute("DELETE FROM schema_version")
            logger.info("Migration history reset")
            return True

        except Exception as e:
            logger.error(f"Failed to reset migration history: {e}")
            return False

    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status information.

        Returns:
            Dictionary with migration status details

        """
        try:
            current_version = self.get_current_version()
            pending_migrations = self.get_pending_migrations()
            migration_history = self.get_migration_history()

            return {
                "current_version": current_version,
                "target_version": self.schema.CURRENT_VERSION,
                "is_up_to_date": len(pending_migrations) == 0,
                "pending_count": len(pending_migrations),
                "pending_migrations": [m.version for m in pending_migrations],
                "applied_count": len(migration_history),
                "applied_migrations": [h["version"] for h in migration_history],
                "integrity_valid": self.verify_migration_integrity(),
            }

        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {"error": str(e)}
