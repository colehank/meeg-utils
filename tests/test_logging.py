"""Test script to verify the unified logging system works correctly."""

from pathlib import Path

from meeg_utils import logger, setup_logging


def test_default_logging():
    """Test default logging configuration."""
    print("\n=== Testing Default Logging ===")
    logger.trace("This is a TRACE message (should only appear in file)")
    logger.debug("This is a DEBUG message (should only appear in file)")
    logger.info("This is an INFO message (appears in console and file)")
    logger.success("This is a SUCCESS message (appears in console and file)")
    logger.warning("This is a WARNING message (appears in console and file)")
    logger.error("This is an ERROR message (appears in console and file)")


def test_custom_logging():
    """Test custom logging configuration."""
    print("\n=== Testing Custom Logging ===")

    # Reconfigure logging to show DEBUG messages in console
    setup_logging(stdout_level="DEBUG", file_level="TRACE", log_filename="test_logging")

    logger.debug("DEBUG message should now appear in console")
    logger.info("INFO message with structured data", n_channels=64, duration=10.5)


def test_exception_logging():
    """Test exception logging."""
    print("\n=== Testing Exception Logging ===")

    try:
        # Intentionally cause an error
        _ = 1 / 0
    except ZeroDivisionError:
        logger.exception("Caught an exception - this should include traceback in file")


def test_log_file_creation():
    """Verify log files are created in .log directory."""
    print("\n=== Checking Log Files ===")

    # Get project root (3 levels up from this file's location)
    project_root = Path(__file__).parent.parent
    log_dir = project_root / ".log"

    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"✓ Log directory exists: {log_dir}")
        print(f"✓ Found {len(log_files)} log file(s):")
        for log_file in sorted(log_files)[-5:]:  # Show last 5 files
            size_mb = log_file.stat().st_size / 1024 / 1024
            print(f"  - {log_file.name} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Log directory not found: {log_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing meeg-utils Unified Logging System")
    print("=" * 70)

    test_default_logging()
    test_custom_logging()
    test_exception_logging()
    test_log_file_creation()

    print("\n" + "=" * 70)
    print("Logging tests completed!")
    print("=" * 70)
    print("\nCheck the .log directory for detailed log files.")
