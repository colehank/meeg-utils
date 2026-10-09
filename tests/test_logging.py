"""Tests for logging: silent by default, opt-in, never touching foreign handlers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

from meeg_utils.logger import log_to_file, setup_logging, teardown_logging


def _library_log(message: str) -> None:
    """Log as if from inside meeg_utils.

    loguru takes the module name from the calling frame's globals, so the
    call is executed with ``__name__`` set to a meeg_utils module.
    """
    code = compile("logger.info(message)", "<meeg_utils>", "exec")
    exec(code, {"__name__": "meeg_utils.fake", "logger": logger, "message": message})


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    teardown_logging()


def test_import_configures_nothing(tmp_path: Path) -> None:
    """Importing the package adds no handlers, prints nothing and writes no files."""
    code = (
        "from loguru import logger\n"
        "before = len(logger._core.handlers)\n"
        "import meeg_utils\n"
        "assert len(logger._core.handlers) == before, 'import added handlers'\n"
        "exec(compile(\"logger.info('should not appear')\", 'x', 'exec'),"
        " {'__name__': 'meeg_utils.fake', 'logger': logger})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "should not appear" not in result.stdout + result.stderr
    assert "Logging initialized" not in result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []


def test_messages_are_silenced_by_default() -> None:
    """Without opting in, meeg-utils messages reach no handler."""
    messages: list[str] = []
    handler = logger.add(messages.append, format="{message}")
    try:
        _library_log("hidden")
    finally:
        logger.remove(handler)
    assert messages == []


def test_setup_logging_writes_file_and_keeps_user_handlers(tmp_path: Path) -> None:
    """setup_logging enables messages, writes the file and leaves user handlers alone."""
    user_messages: list[str] = []
    user_handler = logger.add(user_messages.append, format="{message}")
    try:
        log_file = tmp_path / "logs" / "run.log"
        setup_logging(level=None, log_file=log_file, file_level="INFO")
        setup_logging(level=None, log_file=log_file, file_level="INFO")  # idempotent
        _library_log("visible")
        logger.complete()
        assert log_file.read_text().count("visible") == 1
        assert user_messages == ["visible\n"]  # user's handler still present
    finally:
        logger.remove(user_handler)


def test_teardown_silences_again(tmp_path: Path) -> None:
    """teardown_logging removes our handlers and disables messages."""
    log_file = tmp_path / "run.log"
    setup_logging(level=None, log_file=log_file)
    teardown_logging()
    messages: list[str] = []
    handler = logger.add(messages.append, format="{message}")
    try:
        _library_log("after teardown")
    finally:
        logger.remove(handler)
    assert messages == []


def test_log_to_file_is_scoped(tmp_path: Path) -> None:
    """log_to_file captures messages only inside the with block."""
    log_file = tmp_path / "scoped.log"
    with log_to_file(log_file, level="INFO"):
        _library_log("inside")
    _library_log("outside")
    text = log_file.read_text()
    assert "inside" in text
    assert "outside" not in text
