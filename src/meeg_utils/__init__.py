"""meeg-utils: A Python-based MEEG processing toolkit.

This package provides utilities for processing MEG and EEG data,
including preprocessing, epoching, and feature extraction.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meeg-utils")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

from . import io, presets, steps
from .core import Pipeline, Step
from .logger import log_to_file, logger, setup_logging, teardown_logging
from .preprocessing import BatchPreprocessingPipeline, PreprocessingPipeline

__all__ = [
    "BatchPreprocessingPipeline",
    "Pipeline",
    "PreprocessingPipeline",
    "Step",
    "io",
    "log_to_file",
    "logger",
    "presets",
    "setup_logging",
    "steps",
    "teardown_logging",
]
