"""Database lock mechanism for cross-process synchronization."""

import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class DatabaseLock:
    """File-based lock for critical database operations.
    
    This provides a simple cross-process synchronization mechanism
    for operations that should not run concurrently.
    """
    
    def __init__(self, db_path: Path, lock_name: str = "operation") -> None:
        """Initialize the database lock.
        
        Args:
            db_path: Path to the database file
            lock_name: Name of the lock (used for lock file naming)

        """
        self.lock_file = db_path.parent / f".{db_path.stem}.{lock_name}.lock"
        self.lock_fd: Optional[int] = None
        
    @contextmanager
    def acquire(self, timeout: float = 10.0) -> Generator[None, None, None]:
        """Acquire the lock with timeout.
        
        Args:
            timeout: Maximum time to wait for the lock (in seconds)
            
        Yields:
            None when lock is acquired
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout

        """
        start_time = time.time()
        retry_delay = 0.1  # 100ms initial delay
        
        # Ensure lock directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open or create the lock file
        self.lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_WRONLY)
        
        try:
            while True:
                try:
                    # Try to acquire exclusive lock (non-blocking)
                    fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.debug(f"Acquired lock: {self.lock_file}")
                    break
                    
                except IOError:
                    # Check if we've exceeded timeout
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Could not acquire lock {self.lock_file} within {timeout}s")
                    
                    # Log only occasionally to avoid spam
                    if int(time.time() - start_time) % 5 == 0:
                        logger.debug(f"Waiting for lock {self.lock_file}...")
                    
                    time.sleep(retry_delay)
                    # Cap the retry delay at 1 second
                    retry_delay = min(retry_delay * 1.5, 1.0)
            
            # Lock acquired, yield control
            yield
            
        finally:
            # Release the lock
            if self.lock_fd is not None:
                try:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                    os.close(self.lock_fd)
                    logger.debug(f"Released lock: {self.lock_file}")
                except Exception as e:
                    logger.warning(f"Error releasing lock: {e}")
                
                self.lock_fd = None
                
                # Try to remove the lock file (best effort)
                try:
                    self.lock_file.unlink()
                except Exception:
                    logger.debug("Failed to remove lock file")
    
    def is_locked(self) -> bool:
        """Check if the lock is currently held by another process.
        
        Returns:
            True if locked, False otherwise

        """
        if not self.lock_file.exists():
            return False
            
        try:
            fd = os.open(str(self.lock_file), os.O_WRONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            except IOError:
                return True
            finally:
                os.close(fd)
        except Exception:
            return False
