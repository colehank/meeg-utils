"""Acquisition-quality checks: how well was the recording made?

Checks never modify the data. Run them all with :func:`inspect`::

    report = meu.qc.inspect(raw)
    report.flags        # findings that are not "ok", worst first
    report.plot()       # {check: {kind: Figure}}

or one at a time, with your own thresholds::

    meu.qc.Bridging(lm_cutoff=12).compute(raw).plot("topomap")
"""

from __future__ import annotations

from ._base import LEVELS, Check, Finding, QCReport, inspect
from .eeg import Bridging, Impedance
from .events import Events
from .head import Digitization, HeadMovement
from .signal import Amplitude, Blinks, HeartRate, Muscle, NarrowbandNoise, OutlierChannels

#: Checks run by :func:`inspect` by default, in report order.
DEFAULT_CHECKS: tuple[type[Check], ...] = (
    Amplitude,
    Bridging,
    Impedance,
    OutlierChannels,
    NarrowbandNoise,
    Muscle,
    Blinks,
    HeartRate,
    HeadMovement,
    Digitization,
    Events,
)

__all__ = [
    "DEFAULT_CHECKS",
    "LEVELS",
    "Amplitude",
    "Blinks",
    "Bridging",
    "Check",
    "Digitization",
    "Events",
    "Finding",
    "HeadMovement",
    "HeartRate",
    "Impedance",
    "Muscle",
    "NarrowbandNoise",
    "OutlierChannels",
    "QCReport",
    "inspect",
]
