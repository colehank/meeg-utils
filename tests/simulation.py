"""Test data: synthetic EEG with known sources, and MNE's Neuromag test recording."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from mne.io import BaseRaw

EEG_CHANNELS = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
    "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
    "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2",
]  # fmt: skip
SFREQ = 250.0
DURATION = 60.0
FRONTAL = ["Fp1", "Fp2", "AF3", "AF4"]
NOISY = "C4"
FLAT = "P8"

# MNE's small Neuromag recording (306 MEG + 60 EEG, 24 s), pinned to a release.
NEUROMAG_URL = (
    "https://raw.githubusercontent.com/mne-tools/mne-python/v1.13.2/mne/io/tests/data/test_raw.fif"
)
NEUROMAG_SHA256 = "b1dbe0150bd036cc2446f352265cdaf7deb7fa8eafb16eec6c64e403f4176e52"
#: cHPI coil frequencies of make_chpi (below the recording's 172 Hz low-pass).
CHPI_FREQS = (83.0, 103.0, 123.0, 143.0)


def make_eeg(
    *,
    line: bool = True,
    blinks: bool = True,
    bad: bool = False,
    seed: int = 0,
    line_amplitude: float = 20e-6,
) -> BaseRaw:
    """Simulate EEG with known sources.

    - Broadband background from 20 random dipole-like mixtures plus sensor noise.
    - Occipital 10 Hz alpha.
    - Blinks: large, slow deflections that decay from front to back.
    - 50 Hz line noise with a smooth spatial pattern.
    - Optionally a very noisy channel (C4) and a flat one (P8).
    - An EOG channel, non-zero first_samp, meas_date, annotations, line_freq = 50.
    """
    rng = np.random.default_rng(seed)
    n = int(SFREQ * DURATION)
    t = np.arange(n) / SFREQ
    montage = mne.channels.make_standard_montage("colin27_1020")
    info = mne.create_info([*EEG_CHANNELS, "EOG"], SFREQ, ["eeg"] * len(EEG_CHANNELS) + ["eog"])
    raw = mne.io.RawArray(
        np.zeros((len(EEG_CHANNELS) + 1, n)), info, first_samp=2500, verbose=False
    )
    raw.set_montage(montage, verbose=False)
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(len(EEG_CHANNELS))])
    y = (pos[:, 1] - pos[:, 1].min()) / np.ptp(pos[:, 1])  # 0 = back, 1 = front

    from scipy.signal import lfilter

    # Background: 1/f-like sources with smooth, overlapping scalp projections,
    # so neighbouring electrodes are correlated as in real EEG.
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    centres = rng.normal(0, 1, (20, 3))
    centres[:, 2] = np.abs(centres[:, 2])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    angle = np.arccos(np.clip(unit @ centres.T, -1, 1))
    mixing = np.exp(-(angle**2) / (2 * 0.6**2))
    sources = lfilter([1.0], [1.0, -0.97], rng.normal(0, 1, (20, n)), axis=1)
    data = mixing @ sources * 1.5e-6
    data += rng.normal(0, 0.5e-6, data.shape)  # sensor noise
    data += np.outer(np.exp(-4 * y), np.sin(2 * np.pi * 10 * t)) * 8e-6  # alpha, posterior

    blink = np.zeros(n)
    if blinks:
        kernel = np.hanning(int(0.3 * SFREQ))
        for onset in rng.choice(n - len(kernel), size=15, replace=False):  # 15 per minute
            blink[onset : onset + len(kernel)] += kernel
        data += np.outer(y**3, blink) * 100e-6

    if line:
        data += np.outer(0.5 + y, np.sin(2 * np.pi * 50 * t + 0.4)) * line_amplitude

    eog = blink * 300e-6 + rng.normal(0, 2e-6, n)
    raw._data[:] = np.vstack([data, eog])
    if bad:
        raw._data[EEG_CHANNELS.index(NOISY)] += rng.normal(0, 60e-6, n)
        raw._data[EEG_CHANNELS.index(FLAT)] = 0.0

    raw.info["line_freq"] = 50.0
    raw.set_meas_date(1_600_000_000)
    raw.set_annotations(
        mne.Annotations([15.0, 40.0], [1.0, 1.0], ["stim", "stim"], orig_time=raw.info["meas_date"])
    )
    return raw


#: Peak amplitude (V) of the simulated P300 at Pz for targets and standards.
P300 = {"target": 10e-6, "standard": 3e-6}


def make_erp(*, n_trials: int = 60, seed: int = 0, erp: bool = True) -> BaseRaw:
    """Simulate an oddball EEG experiment with a known ERP.

    Background as :func:`make_eeg` (no line noise, no blinks); every 0.9-1.1 s a
    ``stim/target`` (1 in 4) or ``stim/standard`` event (jittered by up to 0.2 s), followed by a
    posterior positivity peaking at 300 ms whose amplitude at Pz is ``P300``.
    With ``erp=False`` the same recording (same events) without the ERP.
    """
    raw = make_eeg(line=False, blinks=False, seed=seed)
    rng = np.random.default_rng(seed + 100)
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in range(len(EEG_CHANNELS))])
    pz = pos[EEG_CHANNELS.index("Pz")]
    topography = np.exp(-np.sum((pos - pz) ** 2, axis=1) / (2 * 0.05**2))
    kernel_t = np.arange(int(0.8 * SFREQ)) / SFREQ
    wave = np.exp(-((kernel_t - 0.3) ** 2) / (2 * 0.06**2))
    onsets, names = [], []
    for k in range(n_trials):
        onset = 2.0 + 0.9 * k + rng.uniform(0, 0.2)  # jitter: background is not phase-locked
        name = "stim/target" if rng.random() < 0.25 else "stim/standard"
        start = round(onset * SFREQ)
        if erp:
            raw._data[: len(EEG_CHANNELS), start : start + len(wave)] += (
                np.outer(topography, wave) * P300[name.split("/")[1]]
            )
        onsets.append(onset)
        names.append(name)
    raw.set_annotations(
        mne.Annotations(  # onsets relative to meas_date; the data start at first_samp
            np.array(onsets) + raw.first_time,
            [0.0] * len(onsets),
            names,
            orig_time=raw.info["meas_date"],
        )
    )
    return raw


def make_chpi(fname) -> tuple[BaseRaw, Any, np.ndarray]:
    """Neuromag data with simulated cHPI and a known head movement.

    The head moves 8 mm along the device-to-head x translation over 8 s.
    Returns the recording (MEG and stim channels), a function giving the
    true displacement (m) at given times, and the head positions.
    """
    raw = mne.io.read_raw_fif(fname, verbose=False).crop(0, 8).load_data(verbose=False)
    raw.pick(["meg", "stim"])
    with raw.info._unlock():
        for coil, freq in zip(raw.info["hpi_meas"][0]["hpi_coils"], CHPI_FREQS, strict=True):
            coil["coil_freq"] = freq
    t0 = raw.first_samp / raw.info["sfreq"]
    times = t0 + np.arange(0, 8.01, 0.5)
    dev_head = raw.info["dev_head_t"]["trans"]
    pos = np.zeros((len(times), 10))
    pos[:, 0] = times
    pos[:, 1:4] = mne.transforms.rot_to_quat(dev_head[:3, :3])
    pos[:, 4:7] = dev_head[:3, 3]
    pos[:, 4] += np.linspace(0, 0.008, len(times))
    pos[:, 7] = 1
    mne.simulation.add_chpi(raw, head_pos=pos, verbose=False)

    def truth(t: np.ndarray) -> np.ndarray:
        return np.interp(t, times, np.linspace(0, 0.008, len(times)))

    return raw, truth, pos
