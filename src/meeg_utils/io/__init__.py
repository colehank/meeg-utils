"""Reading, writing and inspecting MEG/EEG recordings."""

from __future__ import annotations

from .derivatives import save_derivative
from .read import read
from .system import MEG_SYSTEMS, SYSTEMS, detect_system, get_datatypes

__all__ = [
    "MEG_SYSTEMS",
    "SYSTEMS",
    "detect_system",
    "get_datatypes",
    "read",
    "save_derivative",
]
