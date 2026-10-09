"""Pytest configuration and shared fixtures for testing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: figures are created, never shown

import mne
import numpy as np
import pytest
from mne.io import BaseRaw


@pytest.fixture
def small_raw() -> BaseRaw:
    """Small EEG recording with a non-zero first_samp, meas_date and annotations.

    These are exactly the properties that steps must not corrupt.
    """
    sfreq = 100.0
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz", "EOG"], sfreq, ["eeg"] * 4 + ["eog"])
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (5, 2000)) + 3e-6
    raw = mne.io.RawArray(data, info, first_samp=1000, verbose=False)
    raw.set_meas_date(1_600_000_000)
    raw.set_annotations(
        mne.Annotations([12.0, 25.0], [0.5, 0.5], ["stim", "stim"], orig_time=raw.info["meas_date"])
    )
    return raw
