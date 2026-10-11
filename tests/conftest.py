"""Pytest configuration and shared fixtures for testing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: figures are created, never shown

from pathlib import Path

import mne
import numpy as np
import pytest
from mne.io import BaseRaw

from .simulation import NEUROMAG_SHA256, NEUROMAG_URL, make_eeg


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


@pytest.fixture
def eeg() -> BaseRaw:
    """Synthetic EEG (see make_eeg)."""
    return make_eeg()


@pytest.fixture
def eeg_bad() -> BaseRaw:
    """Synthetic EEG with a noisy and a flat channel."""
    return make_eeg(bad=True)


@pytest.fixture(scope="session")
def neuromag_fname(tmp_path_factory) -> Path:
    """Path to MNE's Neuromag test recording (downloaded once, hash-checked)."""
    pooch = pytest.importorskip("pooch")
    try:
        return Path(
            pooch.retrieve(
                NEUROMAG_URL,
                known_hash=f"sha256:{NEUROMAG_SHA256}",
                path=pooch.os_cache("meeg-utils-tests"),
                progressbar=False,
            )
        )
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"Neuromag test data unavailable: {exc}")


@pytest.fixture
def neuromag(neuromag_fname: Path) -> BaseRaw:
    """10 s of the Neuromag test recording (MEG + EEG)."""
    raw = mne.io.read_raw_fif(neuromag_fname, preload=False, verbose=False)
    return raw.crop(0, 10).load_data(verbose=False)
