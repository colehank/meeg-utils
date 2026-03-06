# logger.py
"""Unified logging configuration for meeg-utils.

This module provides centralized logging configuration using loguru.
All logs are written to the .log directory in the project root.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Determine project root and log directory
# When imported as a package, this will be relative to the package location
_MODULE_DIR = Path(__file__).parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent  # Go up to project root
LOG_DIR = _PROJECT_ROOT / ".log"

# Global flag to track if logging has been initialized
_LOGGING_INITIALIZED = False


def setup_logging(
    stdout_level: str = "INFO",
    file_level: str = "DEBUG",
    enable_file_logging: bool = True,
    log_filename: str | None = None,
    log_dir: Path | None = None,
) -> None:
    """Setup unified logging configuration for the entire project.

    This function configures loguru to:
    1. Output to stdout with the specified level
    2. Write to a log file in the .log directory (if enabled)

    Parameters
    ----------
    stdout_level : str, optional
        Logging level for stdout output. Default is "INFO".
        Options: "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    file_level : str, optional
        Logging level for file output. Default is "DEBUG".
    enable_file_logging : bool, optional
        Whether to enable file logging. Default is True.
    log_filename : str | None, optional
        Custom log filename. If None, uses timestamp-based filename.
    log_dir : Path | None, optional
        Custom log directory. If None, uses default .log directory.

    Examples
    --------
    >>> from meeg_utils.logger import setup_logging
    >>> setup_logging(stdout_level="INFO", file_level="DEBUG")
    """
    global _LOGGING_INITIALIZED  # noqa: PLW0603

    # Remove all existing handlers
    logger.remove()

    # Add stdout handler with formatting
    logger.add(
        sys.stdout,
        level=stdout_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        enqueue=True,
    )

    # Add file handler if enabled
    if enable_file_logging:
        # Determine log directory
        target_log_dir = log_dir if log_dir is not None else LOG_DIR
        target_log_dir.mkdir(parents=True, exist_ok=True)

        # Determine log filename
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"meeg_utils_{timestamp}.log"
        elif not log_filename.endswith(".log"):
            log_filename = f"{log_filename}.log"

        log_path = target_log_dir / log_filename

        logger.add(
            log_path,
            level=file_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        logger.debug(f"File logging enabled: {log_path}")

    _LOGGING_INITIALIZED = True
    logger.info(
        f"Logging initialized (stdout: {stdout_level}, file: {file_level if enable_file_logging else 'disabled'})"
    )


def get_logger():
    """Get the configured logger instance.

    If logging hasn't been initialized, this will call setup_logging()
    with default parameters.

    Returns
    -------
    logger
        The loguru logger instance.
    """
    global _LOGGING_INITIALIZED  # noqa: PLW0602

    if not _LOGGING_INITIALIZED:
        setup_logging()

    return logger


# Initialize logging with default settings when module is imported
# This ensures consistent logging behavior across the entire project
setup_logging(
    stdout_level="INFO",
    file_level="DEBUG",
    enable_file_logging=True,
)
