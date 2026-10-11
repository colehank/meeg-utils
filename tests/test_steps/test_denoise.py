"""ASR repairs injected bursts; SNS removes noise specific to single sensors."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from meeg_utils import steps as S
from meeg_utils.testing import check_step

from ..simulation import make_eeg

BURST = (30.0, 31.0)  # s, from the start of the data


@pytest.fixture(scope="module")
def clean():
    raw = make_eeg(blinks=False, line=False)
    return S.Filter(1.0, None).fit_transform(raw)


def _burst(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, slice]:
    """A 1 s, high-amplitude artifact shared by a group of channels (e.g. a cable movement)."""
    raw = raw.copy()
    sfreq = raw.info["sfreq"]
    window = slice(int(BURST[0] * sfreq), int(BURST[1] * sfreq))
    rng = np.random.default_rng(5)
    n = window.stop - window.start
    pattern = rng.normal(size=len(raw.ch_names))
    pattern[raw.ch_names.index("EOG")] = 0
    time_course = np.convolve(rng.normal(size=n), np.hanning(25), mode="same")
    raw._data[:, window] += np.outer(pattern, time_course / np.abs(time_course).max() * 300e-6)
    return raw, window


class TestASR:
    def test_contract(self, clean):
        check_step(S.ASR(), _burst(clean)[0])

    def test_repairs_burst_leaves_clean_data(self, clean):
        measured, window = _burst(clean)
        step = S.ASR()
        out = step.fit_transform(measured)
        eeg = mne.pick_types(clean.info, eeg=True)
        truth, data, before = clean.get_data(eeg), out.get_data(eeg), measured.get_data(eeg)

        def error(x: np.ndarray, sl: slice | np.ndarray) -> float:
            return float(np.linalg.norm(x[:, sl] - truth[:, sl]) / np.linalg.norm(truth[:, sl]))

        assert error(before, window) > 5
        assert error(data, window) < 0.3 * error(before, window)
        outside = np.ones(truth.shape[1], bool)
        outside[window.start - 250 : window.stop + 250] = False  # ASR windows overlap the edges
        assert error(data, outside) < 0.05
        assert 0 < step.qc_["fraction_reconstructed_samples"] < 0.2
        assert set(step.plot()) == {"reconstruction", "psd"}

    def test_eog_untouched(self, clean):
        measured, _ = _burst(clean)
        out = S.ASR().fit_transform(measured)
        assert np.array_equal(out.get_data("EOG"), measured.get_data("EOG"))

    def test_warns_without_highpass(self):
        with pytest.warns(UserWarning, match="drifts removed first"):
            S.ASR().fit(make_eeg(blinks=False))


class TestSNS:
    @pytest.fixture(scope="class")
    def sensor_noise(self):
        """32 channels mixing 6 sources, plus independent noise of the same power on each."""
        rng = np.random.default_rng(0)
        info = mne.create_info([f"EEG{i:02d}" for i in range(32)], 250.0, "eeg")
        signal = rng.normal(size=(32, 6)) @ rng.normal(size=(6, 15000)) * 5e-6
        noise = rng.normal(size=signal.shape) * signal.std()
        raw = mne.io.RawArray(signal + noise, info, first_samp=10, verbose=False)
        raw.set_meas_date(1_600_000_000)
        raw.set_annotations(
            mne.Annotations([1.0], [0.5], ["stim"], orig_time=raw.info["meas_date"])
        )
        return raw, signal

    def test_contract(self, sensor_noise):
        check_step(S.SNS(), sensor_noise[0])

    def test_removes_sensor_specific_noise(self, sensor_noise):
        raw, signal = sensor_noise
        step = S.SNS()
        out = step.fit_transform(raw)

        def error(x: np.ndarray) -> float:
            return float(np.linalg.norm(x - signal) / np.linalg.norm(signal))

        assert error(raw.get_data()) > 0.9
        assert error(out.get_data()) < 0.6  # noise halved with 10 neighbours
        assert step.qc_["eeg"]["variance_removed_pct"] > 30

    def test_epochs(self, sensor_noise):
        raw, _ = sensor_noise
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)
        out = S.SNS().fit_transform(epochs)
        assert out.get_data().shape == epochs.get_data().shape
        assert np.var(out.get_data()) < np.var(epochs.get_data())

    def test_bad_channels_untouched(self, sensor_noise):
        raw = sensor_noise[0].copy()
        raw.info["bads"] = ["EEG00"]
        out = S.SNS().fit_transform(raw)
        assert np.array_equal(out.get_data("EEG00"), raw.get_data("EEG00"))
