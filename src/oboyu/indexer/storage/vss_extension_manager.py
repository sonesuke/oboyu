"""VSS Extension Manager for DuckDB vector operations."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class VSSExtensionManager:
    """Manages VSS extension installation, loading, and verification for DuckDB."""

    def ensure_vss_extension(self, conn: "DuckDBPyConnection") -> None:
        """Ensure VSS extension is properly loaded.
        
        Args:
            conn: Database connection
            
        Raises:
            RuntimeError: If VSS extension cannot be loaded

        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # First check if VSS extension is already functional
                try:
                    conn.execute("SELECT [1.0, 2.0, 3.0]::FLOAT[3]").fetchone()
                    logger.debug("VSS extension already functional")
                    # Still set HNSW persistence if needed
                    try:
                        conn.execute("SET hnsw_enable_experimental_persistence=true")
                    except Exception as e:
                        logger.debug(f"HNSW persistence setting skipped: {e}")
                    return
                except Exception:
                    # VSS not functional, need to load it
                    logger.debug("VSS extension not yet functional, attempting to load")

                # Install VSS extension (may fail if already installed, which is fine)
                try:
                    conn.execute("INSTALL vss")
                    logger.debug("VSS extension installed")
                except Exception as install_error:
                    # Installation might fail if already installed
                    install_msg = str(install_error).lower()
                    if "already installed" not in install_msg and "exists" not in install_msg:
                        logger.warning(f"VSS installation warning: {install_error}")

                # Load VSS extension
                try:
                    conn.execute("LOAD vss")
                    logger.debug("VSS extension loaded successfully")
                except Exception as load_error:
                    load_msg = str(load_error).lower()
                    
                    # Handle specific concurrent loading errors
                    if "already exists" in load_msg and "hnsw" in load_msg:
                        # Another process already loaded the extension, verify it works
                        try:
                            conn.execute("SELECT [1.0, 2.0, 3.0]::FLOAT[3]").fetchone()
                            logger.debug("VSS extension loaded by another process")
                        except Exception as verify_error:
                            # Extension state is inconsistent, retry
                            if attempt < max_retries - 1:
                                import time
                                time.sleep(0.1 * (attempt + 1))
                                logger.debug(f"VSS verification failed, retrying (attempt {attempt + 2}/{max_retries})")
                                continue
                            else:
                                raise RuntimeError(f"VSS extension in inconsistent state: {verify_error}")
                    else:
                        # Other load error, retry if not final attempt
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(0.1 * (attempt + 1))
                            logger.debug(f"VSS load failed, retrying (attempt {attempt + 2}/{max_retries}): {load_error}")
                            continue
                        else:
                            raise load_error
                
                # Enable HNSW persistence (best effort)
                try:
                    conn.execute("SET hnsw_enable_experimental_persistence=true")
                    logger.debug("HNSW persistence enabled")
                except Exception as e:
                    logger.debug(f"HNSW persistence setting skipped: {e}")
                
                # Final verification that VSS extension is working
                try:
                    conn.execute("SELECT [1.0, 2.0, 3.0]::FLOAT[3]").fetchone()
                    logger.debug("VSS extension verified successfully")
                    return
                except Exception as verify_error:
                    if attempt < max_retries - 1:
                        logger.debug(f"VSS verification failed, retrying (attempt {attempt + 2}/{max_retries}): {verify_error}")
                        continue
                    else:
                        raise RuntimeError(f"VSS extension verification failed: {verify_error}")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    logger.debug(f"VSS setup failed, retrying (attempt {attempt + 2}/{max_retries}): {e}")
                    continue
                else:
                    logger.error(f"Failed to load VSS extension after {max_retries} attempts: {e}")
                    raise RuntimeError(f"VSS extension setup failed: {e}") from e
