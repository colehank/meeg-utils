"""Acquisition-quality checks against data with known defects."""

from __future__ import annotations

import json

import mne
import numpy as np
import pytest
from mne.io import BaseRaw

from meeg_utils import qc
from meeg_utils.testing import check_check

from ..simulation import EEG_CHANNELS, FLAT, NOISY, SFREQ, make_chpi, make_eeg


def _levels(check: qc.Check) -> dict[str, str]:
    return {f.metric: f.level for f in check.findings_}


def _bridge(raw: BaseRaw, a: str, b: str, noise: float = 0.3e-6, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    names = raw.ch_names
    raw._data[names.index(b)] = raw._data[names.index(a)] + rng.normal(0, noise, raw.n_times)


@pytest.fixture(scope="module")
def chpi_raw(neuromag_fname):
    """Neuromag data with simulated cHPI and a known head movement (see make_chpi)."""
    return make_chpi(neuromag_fname)


class TestContracts:
    """Every check honours the contract on data it applies to."""

    @pytest.mark.parametrize(
        "check",
        [
            qc.Amplitude(),
            qc.OutlierChannels(),
            qc.NarrowbandNoise(),
            qc.Blinks(),
            qc.Digitization(),
            qc.Events(),
            qc.Impedance(impedances=dict.fromkeys(EEG_CHANNELS, 10.0)),
        ],
        ids=lambda c: type(c).__name__,
    )
    def test_eeg(self, check, eeg_bad):
        check_check(check, eeg_bad)

    def test_bridging(self):
        raw = make_eeg()
        _bridge(raw, "F3", "Fz")
        _bridge(raw, "Cz", "CP1", seed=2)
        check_check(qc.Bridging(), raw)

    def test_muscle(self):
        check_check(qc.Muscle(), _muscle_raw())

    def test_heart_rate(self):
        check_check(qc.HeartRate(), _ecg_raw(72.0))

    def test_head_movement(self, chpi_raw):
        check_check(qc.HeadMovement(), chpi_raw[0].copy())


class TestAmplitude:
    def test_finds_flat_clipped_and_missing(self, eeg_bad):
        names = eeg_bad.ch_names
        o1, oz = names.index("O1"), names.index("Oz")
        eeg_bad._data[o1] = np.clip(eeg_bad._data[o1], -20e-6, 20e-6)
        eeg_bad._data[oz, 1000:1250] = np.nan  # 1 s
        eeg_bad._data[names.index("Fz"), 5000:5100] = 0.0  # 0.4 s dropout
        check = qc.Amplitude().compute(eeg_bad)
        m = check.metrics_
        assert m["flat_channels"] == [FLAT]
        assert m["clipped_channels"] == ["O1"]
        assert m["nan_channels"] == ["Oz"]
        assert m["flat_segments_s"] == pytest.approx(0.4, abs=0.01)
        assert check.level == "warn"

    def test_clean_data(self, eeg):
        check = qc.Amplitude().compute(eeg)
        assert check.level == "ok"
        assert not any(check.metrics_["channels"].values())


class TestBridging:
    def test_finds_bridged_pairs(self):
        raw = make_eeg()
        _bridge(raw, "F3", "Fz")
        _bridge(raw, "Cz", "CP1", seed=2)
        check = qc.Bridging().compute(raw)
        assert check.metrics_["pairs"] == [["CP1", "Cz"], ["F3", "Fz"]]
        assert check.metrics_["largest_group"] == 2
        assert _levels(check) == {"n_pairs": "warn", "largest_group": "ok"}

        strict = qc.Bridging(bad_limit=1).compute(raw)
        assert strict.level == "fail"

    def test_no_false_bridges(self, eeg):
        assert qc.Bridging().compute(eeg).metrics_["n_pairs"] == 0

    def test_groups(self):
        assert qc.eeg._connected_groups([("A", "B"), ("B", "C"), ("D", "E")]) == [
            ["A", "B", "C"],
            ["D", "E"],
        ]

    def test_flat_eeg_is_not_applicable(self, eeg):
        eeg._data[:32] = 0.0
        assert "not flat" in qc.Bridging().not_applicable(eeg)


class TestImpedance:
    def test_explicit_values(self, eeg):
        values = dict.fromkeys(EEG_CHANNELS, 5.0) | {"C4": 60.0, "Pz": 30.0}
        check = qc.Impedance(impedances=values).compute(eeg)
        assert check.metrics_["high"] == ["Pz", "C4"]
        assert check.level == "warn"
        assert qc.Impedance(impedances=values, warn_kohm=100).compute(eeg).level == "ok"

    def test_brainvision_header(self, eeg):
        eeg.impedances = {
            "Fz": {"imp": 4000.0, "imp_unit": "Ohm"},
            "Cz": {"imp": 0.03, "imp_unit": "MOhm"},
            "Pz": {"imp": None, "imp_unit": "kOhm"},
        }
        check = qc.Impedance().compute(eeg)
        assert check.metrics_["impedances"] == {"Fz": 4.0, "Cz": 30.0}
        assert check.metrics_["high"] == ["Cz"]

    def test_needs_values(self, eeg):
        assert "no impedances" in qc.Impedance().not_applicable(eeg)


class TestOutlierChannels:
    def test_finds_noisy_channel(self, eeg_bad):
        check = qc.OutlierChannels().compute(eeg_bad)
        assert check.metrics_["outliers"]["eeg"] == [NOISY]

    def test_clean_data(self, eeg):
        assert qc.OutlierChannels().compute(eeg).metrics_["outliers"]["eeg"] == []

    def test_marked_bads_are_skipped(self, eeg_bad):
        eeg_bad.info["bads"] = [NOISY]
        check = qc.OutlierChannels().compute(eeg_bad)
        assert NOISY not in check.metrics_["scores"]["eeg"]


class TestNarrowbandNoise:
    def test_line_noise_is_reported_not_flagged(self, eeg):
        check = qc.NarrowbandNoise().compute(eeg)
        result = check.metrics_["channel_types"]["eeg"]
        assert result["line_db"]["50"] > 20
        assert result["other_peaks"] == []
        assert check.level == "ok"

    def test_other_peaks_and_line_outliers(self, eeg):
        t = eeg.times
        eeg._data[:32] += np.sin(2 * np.pi * 23.0 * t) * 3e-6  # equipment noise
        c4 = eeg.ch_names.index(NOISY)
        eeg._data[c4] += np.sin(2 * np.pi * 50 * t + 0.4) * 300e-6  # poor contact
        check = qc.NarrowbandNoise().compute(eeg)
        result = check.metrics_["channel_types"]["eeg"]
        assert result["other_peaks"] == [pytest.approx(23.0, abs=0.3)]
        assert result["line_outliers"] == [NOISY]
        assert _levels(check)["n_other_peaks_eeg"] == "warn"
        assert _levels(check)["n_line_outliers_eeg"] == "warn"

    def test_unknown_line_frequency(self, eeg):
        eeg.info["line_freq"] = None
        check = qc.NarrowbandNoise().compute(eeg)
        assert _levels(check)["fline"] == "warn"
        assert check.metrics_["channel_types"]["eeg"]["other_peaks"] == [pytest.approx(50, abs=0.3)]


class TestMuscle:
    def test_fraction(self):
        check = qc.Muscle().compute(_muscle_raw())
        assert check.metrics_["fraction"] == pytest.approx(0.2, abs=0.04)
        assert check.level == "warn"

    def test_needs_high_frequencies(self, eeg):
        assert "140 Hz" in qc.Muscle().not_applicable(eeg)


class TestBlinks:
    def test_rate(self, eeg):
        check = qc.Blinks().compute(eeg)
        assert check.metrics_["n_blinks"] == pytest.approx(15, abs=1)
        assert check.level == "ok"

    def test_flat_eog(self, eeg):
        eeg._data[eeg.ch_names.index("EOG")] = 0.0
        check = qc.Blinks().compute(eeg)
        assert check.metrics_["rate_per_min"] == 0
        assert check.metrics_["flat_channels"] == ["EOG"]
        assert check.level == "warn"

    def test_needs_eog(self, eeg):
        assert (
            qc.Blinks().not_applicable(eeg.copy().drop_channels(["EOG"])) == "needs an EOG channel"
        )


class TestHeartRate:
    @pytest.mark.parametrize("bpm", [55.0, 72.0, 95.0])
    def test_rate(self, bpm):
        check = qc.HeartRate().compute(_ecg_raw(bpm))
        assert check.metrics_["heart_rate_bpm"] == pytest.approx(bpm, rel=0.03)
        assert check.metrics_["channel"] == "ECG"
        assert check.level == "ok"

    def test_implausible_rate(self):
        check = qc.HeartRate(max_bpm=60).compute(_ecg_raw(72.0))
        assert _levels(check)["heart_rate_high"] == "warn"

    def test_needs_ecg_or_meg(self, eeg):
        assert "ECG" in qc.HeartRate().not_applicable(eeg)


class TestHeadMovement:
    def test_recovers_known_movement(self, chpi_raw):
        raw, truth, _ = chpi_raw
        check = qc.HeadMovement().compute(raw)
        times = check.pos_[:, 0]
        np.testing.assert_allclose(check.displacement_, truth(times), atol=0.3e-3)
        assert check.metrics_["max_displacement_mm"] == pytest.approx(
            truth(times).max() * 1000, abs=0.3
        )
        assert check.metrics_["max_rotation_deg"] < 0.5
        assert check.metrics_["coil_good_fraction"] == [1.0] * 4
        assert _levels(check)["max_displacement_mm"] == "warn"  # > 5 mm
        assert qc.HeadMovement(warn_mm=10).compute(raw).level == "ok"

    def test_precomputed_positions(self, chpi_raw):
        raw, _, pos = chpi_raw
        check = qc.HeadMovement(head_pos=pos).compute(raw)
        assert check.metrics_["max_displacement_mm"] == pytest.approx(8.0, abs=1e-6)
        assert "coil_good_fraction" not in check.metrics_

    def test_needs_chpi(self, neuromag):
        assert "cHPI" in qc.HeadMovement().not_applicable(neuromag)


class TestDigitization:
    def test_eeg_montage(self, eeg):
        check = qc.Digitization().compute(eeg)
        assert 80 < check.metrics_["head_radius_mm"] < 105
        assert check.metrics_["eeg_without_position"] == []
        assert check.level == "ok"

    def test_missing_positions(self, eeg):
        for name in ("Fz", "Cz"):
            eeg.info["chs"][eeg.ch_names.index(name)]["loc"][:3] = np.nan
        check = qc.Digitization().compute(eeg)
        assert check.metrics_["eeg_without_position"] == ["Fz", "Cz"]
        assert _levels(check)["n_eeg_without_position"] == "warn"

    def test_meg_head_to_sensor_distance(self, neuromag):
        check = qc.Digitization().compute(neuromag)
        assert 5 < check.metrics_["min_gap_mm"] < check.metrics_["median_gap_mm"] < 60
        assert check.level == "ok"

    def test_unmeasured_head_position(self, neuromag):
        with neuromag.info._unlock():
            neuromag.info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
        assert qc.Digitization().compute(neuromag).level == "fail"


class TestEvents:
    def test_counts_from_annotations(self, small_raw):
        check = qc.Events(expected={"stim": 2}).compute(small_raw)
        assert check.metrics_["counts"] == {"stim": 2}
        assert check.metrics_["source"] == "annotations"
        assert check.level == "ok"
        assert qc.Events(expected={"stim": 3}).compute(small_raw).level == "fail"
        assert qc.Events(expected={"target": 1}).compute(small_raw).level == "fail"

    def test_bad_annotations_are_not_events(self, small_raw):
        small_raw.annotations.append(14.0, 1.0, "BAD_movement")
        small_raw.annotations.append(25.0, 1.0, "BAD_movement")
        check = qc.Events().compute(small_raw)
        assert check.metrics_["counts"] == {"stim": 2}
        assert check.metrics_["n_in_bad"] == 1

    def test_stim_channel(self, neuromag):
        neuromag.set_annotations(None)
        check = qc.Events(min_interval=0.2).compute(neuromag)
        expected = mne.find_events(neuromag, shortest_event=1, verbose=False)
        assert check.metrics_["n_events"] == len(expected)
        assert check.metrics_["source"] == "stim"

    def test_simultaneous_events(self, small_raw):
        small_raw.annotations.append(12.0, 0.0, "response")
        assert qc.Events().compute(small_raw).metrics_["n_simultaneous"] == 1

    def test_nothing_to_check(self, small_raw):
        small_raw.set_annotations(None)
        assert "no event" in qc.Events().not_applicable(small_raw)


class TestInspect:
    def test_report(self, eeg_bad, tmp_path):
        report = qc.inspect(eeg_bad)
        assert report.level == "warn"
        assert set(report.skipped) == {
            "impedance", "muscle", "heart_rate", "head_movement",  # no impedances, 250 Hz, no ECG, EEG
            "squid_jumps", "chpi_snr", "empty_room", "bids_metadata",  # MEG or BIDS only
        }  # fmt: skip
        flagged = {(f.check, f.metric) for f in report.flags}
        assert ("amplitude", "n_flat_channels") in flagged
        assert ("outlier_channels", "n_outliers_eeg") in flagged
        json.dumps(report.to_dict(), allow_nan=True, default=str)
        assert {r["check"] for r in report.to_records()} == set(report.checks)
        assert "amplitude" in repr(report)

        figs = report.plot()
        assert set(figs) == set(report.checks)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_reads_paths(self, eeg, tmp_path):
        fname = tmp_path / "rec_raw.fif"
        eeg.save(fname, verbose=False)
        report = qc.inspect(fname, checks=[qc.Amplitude()])
        assert report.source == str(fname)
        assert list(report.checks) == ["amplitude"]

    def test_custom_checks_and_skips(self, eeg):
        report = qc.inspect(eeg, checks=[qc.Blinks(min_rate=30), qc.HeadMovement()])
        assert report.checks["blinks"].level == "warn"
        assert report.skipped == {"head_movement": "needs MEG channels"}

    def test_duplicate_names(self, eeg):
        with pytest.raises(ValueError, match="Two checks"):
            qc.inspect(eeg, checks=[qc.Amplitude(), qc.Amplitude()])


# ----------------------------------------------------------------------


def _muscle_raw() -> BaseRaw:
    """32-channel EEG at 1000 Hz with 110-140 Hz bursts in 20 % of the time."""
    rng = np.random.default_rng(0)
    sfreq, duration = 1000.0, 50.0
    n = int(sfreq * duration)
    info = mne.create_info(EEG_CHANNELS, sfreq, "eeg")
    data = rng.normal(0, 5e-6, (32, n))
    burst = mne.filter.filter_data(rng.normal(0, 1, n), sfreq, 110, 140, verbose=False)
    mask = np.zeros(n, bool)
    for start in range(0, n, 10_000):  # 2 s of every 10 s
        mask[start : start + 2000] = True
    data += np.outer(np.ones(32), np.where(mask, burst, 0)) * 20e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw.set_montage("colin27_1020", verbose=False)


def _ecg_raw(bpm: float) -> BaseRaw:
    """EEG plus an ECG channel with QRS-like peaks at ``bpm``."""
    raw = make_eeg(line=False)
    t = raw.times
    beats = np.arange(0.5, t[-1], 60 / bpm)
    ecg = sum(np.exp(-((t - b) ** 2) / (2 * 0.01**2)) for b in beats) * 1e-3
    ecg = ecg + np.random.default_rng(0).normal(0, 2e-5, len(t))
    info = mne.create_info(["ECG"], SFREQ, "ecg")
    ecg_raw = mne.io.RawArray(ecg[np.newaxis], info, first_samp=raw.first_samp, verbose=False)
    ecg_raw.set_meas_date(raw.info["meas_date"])
    return raw.add_channels([ecg_raw], force_update_info=True)
