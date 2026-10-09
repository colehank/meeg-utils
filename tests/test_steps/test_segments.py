"""BadSegments: finds artifacts injected at known times."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from meeg_utils import steps as S
from meeg_utils.testing import check_step

from ..simulation import make_eeg

ARTIFACTS = (10.3, 31.6, 47.2)  # onsets (s, from the start of the data)


def _with_artifacts(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Large, 0.3 s movement artifacts on a few frontal channels."""
    raw = raw.copy()
    sfreq = raw.info["sfreq"]
    picks = [raw.ch_names.index(ch) for ch in ("Fp1", "Fp2", "F7")]
    for onset in ARTIFACTS:
        start = int(onset * sfreq)
        n = int(0.3 * sfreq)
        raw._data[picks, start : start + n] += 400e-6 * np.hanning(n)
    return raw


def _segments(raw: mne.io.BaseRaw, description: str) -> list[tuple[float, float]]:
    offset = raw.first_time if raw.annotations.orig_time is not None else 0.0
    return [
        (a["onset"] - offset, a["onset"] - offset + a["duration"])
        for a in raw.annotations
        if a["description"] == description
    ]


@pytest.fixture
def artifacts():
    """Simulated EEG without blinks (the automatic threshold would also flag them)."""
    return _with_artifacts(make_eeg(blinks=False))


def test_contract(artifacts):
    check_step(S.BadSegments("amplitude"), artifacts)


def test_finds_amplitude_artifacts(artifacts):
    step = S.BadSegments("amplitude")
    out = step.fit_transform(artifacts)
    found = _segments(out, "BAD_amplitude")
    for onset in ARTIFACTS:
        assert any(start <= onset and onset + 0.3 <= end + 1.0 for start, end in found), onset
    assert step.qc_["amplitude"]["fraction"] < 0.15  # few false alarms
    assert 0 < step.reject_["eeg"] < 400e-6
    assert step.qc_["fraction_bad"] >= step.qc_["amplitude"]["fraction"]


def test_existing_annotations_kept(artifacts):
    orig_time = artifacts.annotations.orig_time
    offset = artifacts.first_time if orig_time is not None else 0.0
    onsets = [offset + 5.0, offset + 20.0]
    artifacts.set_annotations(
        mne.Annotations(onsets, [1.0, 2.0], ["stim", "BAD_manual"], orig_time=orig_time)
    )
    out = S.BadSegments("amplitude", reject={"eeg": 300e-6}).fit_transform(artifacts)
    descriptions = list(out.annotations.description)
    assert "stim" in descriptions and "BAD_manual" in descriptions
    assert descriptions.count("BAD_amplitude") == len(ARTIFACTS)


def test_annotations_follow_first_samp():
    """Onsets are right whether or not the recording has a measurement date."""
    for meas_date in (None, 1_600_000_000):
        raw = _with_artifacts(make_eeg())
        raw = mne.io.RawArray(raw.get_data(), raw.info, first_samp=1234, verbose=False)
        raw.set_meas_date(meas_date)
        out = S.BadSegments("amplitude", reject={"eeg": 300e-6}).fit_transform(raw)
        found = _segments(out, "BAD_amplitude")
        assert [int(start) for start, _ in found] == [int(o) for o in ARTIFACTS]


def test_flat(eeg):
    raw = eeg.copy()
    sfreq = raw.info["sfreq"]
    raw._data[:, int(20 * sfreq) : int(23 * sfreq)] = 0.0  # amplifier saturation / dropout
    step = S.BadSegments("flat", flat={"eeg": 1e-6})
    out = step.fit_transform(raw)
    assert _segments(out, "BAD_flat") == [(20.0, 23.0)]


def test_muscle():
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    info = mne.create_info([f"EEG{i}" for i in range(16)], sfreq, "eeg")
    data = rng.normal(0, 5e-6, (16, int(60 * sfreq)))
    burst = slice(int(30 * sfreq), int(32 * sfreq))
    t = np.arange(burst.stop - burst.start) / sfreq
    data[:, burst] += 30e-6 * np.sin(2 * np.pi * 125 * t) * rng.normal(1, 0.1, (16, 1))
    raw = mne.io.RawArray(data, info, verbose=False)
    step = S.BadSegments("muscle")
    out = step.fit_transform(raw)
    found = _segments(out, "BAD_muscle")
    assert found and all(start > 29.5 and end < 32.5 for start, end in found)
    assert step.qc_["muscle"]["duration_s"] > 1.5
    assert set(step.plot()) == {"segments"}


def test_muscle_needs_high_frequencies(eeg):
    with pytest.raises(ValueError, match="up to 140 Hz"):
        S.BadSegments("muscle").fit(eeg)  # 250 Hz sampling


def test_epochs_skip_bad_segments(artifacts):
    out = S.BadSegments("amplitude", reject={"eeg": 300e-6}).fit_transform(artifacts)
    epochs = mne.make_fixed_length_epochs(out, duration=1.0, preload=True, verbose=False)
    clean = mne.make_fixed_length_epochs(artifacts, duration=1.0, preload=True, verbose=False)
    assert len(clean) - len(epochs) >= len(ARTIFACTS)
    dropped = [log for log in epochs.drop_log if "BAD_amplitude" in log]
    assert len(dropped) >= len(ARTIFACTS)


class TestErrors:
    def test_unknown_method(self, eeg):
        with pytest.raises(ValueError, match="methods must be"):
            S.BadSegments("blinks").fit(eeg)

    def test_flat_needs_thresholds(self, eeg):
        with pytest.raises(ValueError, match="needs thresholds"):
            S.BadSegments("flat").fit(eeg)

    def test_auto_needs_data(self, eeg):
        with pytest.raises(ValueError, match="at least 20 clean windows"):
            S.BadSegments("amplitude").fit(eeg.copy().crop(0, 10))
