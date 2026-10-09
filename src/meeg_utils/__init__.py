"""meeg-utils: MEG and EEG quality control, preprocessing and analysis.

Use as ``import meeg_utils as meu``; processing steps live in
``meu.steps`` and ready-made pipelines in ``meu.presets``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meeg-utils")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

from . import epochs, io, presets, qc, report, steps
from .batch import process
from .core import Pipeline, Step
from .dataset import BatchResult, Dataset, process_dataset
from .logger import log_to_file, logger, setup_logging, teardown_logging

__all__ = [
    "BatchResult",
    "Dataset",
    "Pipeline",
    "Step",
    "epochs",
    "io",
    "log_to_file",
    "logger",
    "presets",
    "process",
    "process_dataset",
    "qc",
    "report",
    "setup_logging",
    "steps",
    "teardown_logging",
]
