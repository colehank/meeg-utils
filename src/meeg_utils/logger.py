"""Logging for meeg-utils.

meeg-utils logs through `loguru <https://loguru.readthedocs.io>`_ but, like any
library, configures nothing on import: its messages are disabled until you
opt in, and it never adds, removes or changes handlers you did not ask for.

Turn logging on with :func:`setup_logging`, or, if you configure loguru
yourself, with ``logger.enable("meeg_utils")``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

logger.disable("meeg_utils")

# Handler ids added by setup_logging, so that it only ever removes its own.
_HANDLER_IDS: list[int] = []

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


def setup_logging(
    level: str | None = "INFO",
    log_file: str | Path | None = None,
    file_level: str = "DEBUG",
    *,
    enqueue: bool = False,
) -> None:
    """Show meeg-utils log messages on the console and/or in a file.

    Calling it again replaces the handlers added by the previous call.
    Handlers you added yourself are left alone; loguru's built-in default
    stderr handler is removed so that messages are not printed twice.

    Parameters
    ----------
    level : str | None
        Minimum level shown on the console (stderr), e.g. ``"DEBUG"``,
        ``"INFO"`` or ``"WARNING"``. ``None`` disables console output.
    log_file : str | Path | None
        File to write log messages to; parent directories are created.
        ``None`` (default) disables file logging.
    file_level : str
        Minimum level written to ``log_file``.
    enqueue : bool
        Pass messages through a multiprocessing-safe queue. Use it when
        several processes write to the same file.

    Examples
    --------
    >>> import meeg_utils as meu
    >>> meu.setup_logging("INFO", log_file="logs/preproc.log")  # doctest: +SKIP
    """
    teardown_logging()
    try:
        logger.remove(0)  # loguru's default stderr handler, if still present
    except ValueError:
        pass
    logger.enable("meeg_utils")

    if level is not None:
        _HANDLER_IDS.append(
            logger.add(
                sys.stderr,
                level=level,
                format=_CONSOLE_FORMAT,
                filter="meeg_utils",
                enqueue=enqueue,
            )
        )

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _HANDLER_IDS.append(
            logger.add(
                log_file,
                level=file_level,
                format=_FILE_FORMAT,
                filter="meeg_utils",
                enqueue=enqueue,
                backtrace=True,
                diagnose=False,  # do not dump local variables (may hold data) into files
            )
        )


def teardown_logging() -> None:
    """Remove the handlers added by :func:`setup_logging` and silence meeg-utils again."""
    while _HANDLER_IDS:
        handler_id = _HANDLER_IDS.pop()
        try:
            logger.remove(handler_id)
        except ValueError:  # already removed by the user
            pass
    logger.disable("meeg_utils")


@contextmanager
def log_to_file(log_file: str | Path, level: str = "DEBUG") -> Iterator[Path]:
    """Write meeg-utils log messages to a file for the duration of a ``with`` block.

    Afterwards the file handler is removed and, unless :func:`setup_logging`
    is active, meeg-utils is silenced again.

    Parameters
    ----------
    log_file : str | Path
        File to write to; parent directories are created.
    level : str
        Minimum level written to the file.

    Yields
    ------
    Path
        The log file.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configured = bool(_HANDLER_IDS)
    logger.enable("meeg_utils")
    handler_id = logger.add(
        log_file, level=level, format=_FILE_FORMAT, filter="meeg_utils", enqueue=True
    )
    try:
        yield log_file
    finally:
        logger.remove(handler_id)
        if not configured:
            logger.disable("meeg_utils")
