"""Combining runs, and the had-meeg epochs stage."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from mne.transforms import Transform, translation

import meeg_utils as meu
from meeg_utils import steps as S

from ..simulation import make_erp


def _runs(n=2):
    step = S.Epoch("stim", -0.2, 0.8)
    first = make_erp(seed=0)
    step.fit(first)
    return [step.transform(first)] + [step.transform(make_erp(seed=i)) for i in range(1, n)]


class TestCombine:
    def test_concatenates_with_union_of_bads(self):
        a, b = _runs()
        a.info["bads"] = ["Cz"]
        b.info["bads"] = ["Pz"]
        combined = meu.epochs.combine([a, b])
        assert len(combined) == len(a) + len(b)
        assert combined.info["bads"] == ["Cz", "Pz"]
        assert a.info["bads"] == ["Cz"]  # inputs untouched
        np.testing.assert_array_equal(combined.get_data()[: len(a)], a.get_data())

    def test_interpolate_bads(self):
        a, b = _runs()
        b.info["bads"] = ["Pz"]
        combined = meu.epochs.combine([a, b], bads="interpolate")
        assert combined.info["bads"] == []
        assert not np.allclose(combined.get_data(["Pz"])[: len(a)], a.get_data(["Pz"]))

    def test_incompatible_runs(self):
        a, b = _runs()
        with pytest.raises(ValueError, match="different channels"):
            meu.epochs.combine([a, b.copy().drop_channels(["Fp1"])])
        with pytest.raises(ValueError, match="sampling rate or epoch times"):
            meu.epochs.combine([a, b.copy().crop(0, 0.5)])
        with pytest.raises(ValueError, match="bads must be"):
            meu.epochs.combine([a, b], bads="drop")

    def test_event_codes_are_unified_by_name(self):
        """Runs epoched separately can number the same condition differently."""
        a = S.Epoch("stim", -0.2, 0.8).fit_transform(make_erp(seed=0))
        other = make_erp(seed=1)
        other.annotations.append(5.0 + other.first_time, 0.0, "aaa")  # sorts first: shifts codes
        b = S.Epoch(None, -0.2, 0.8).fit_transform(other)
        assert b.event_id["stim/target"] != a.event_id["stim/target"]
        combined = meu.epochs.combine([a, b])
        assert combined.event_id["stim/target"] == a.event_id["stim/target"]
        assert len(combined["stim/target"]) == len(a["stim/target"]) + len(b["stim/target"])
        assert len(combined["aaa"]) == 1

    def test_meg_head_positions(self, neuromag):
        epochs = mne.make_fixed_length_epochs(
            neuromag.copy().pick("meg"), duration=1.0, preload=True, verbose=False
        )
        moved = epochs.copy()
        with moved.info._unlock():
            moved.info["dev_head_t"] = Transform(
                "meg", "head", translation(0.005, 0, 0) @ epochs.info["dev_head_t"]["trans"]
            )
        with pytest.raises(ValueError, match=r"5\.0 mm.*HeadAlign"):
            meu.epochs.combine([epochs, moved])
        combined = meu.epochs.combine([epochs, moved], max_head_shift_mm=10)
        assert len(combined) == 2 * len(epochs)


class TestHadMeegEpochs:
    @pytest.fixture
    def run(self):
        raw = make_erp()
        raw.annotations.rename(
            dict.fromkeys(set(raw.annotations.description), "video on"), verbose=False
        )
        return raw

    def test_eeg(self, run):
        pipe = meu.Pipeline.preset("had-meeg", datatype="eeg", stage="epochs")
        epochs = pipe.fit_transform(run)
        assert epochs.info["sfreq"] == 200.0
        assert epochs.info["lowpass"] == 40.0
        assert (epochs.tmin, epochs.tmax) == pytest.approx((-0.1, 2.0))
        assert list(epochs.event_id) == ["video on"]
        sel = epochs.times <= 0
        np.testing.assert_allclose(epochs.get_data("eeg")[..., sel].mean(axis=-1), 0, atol=1e-12)
        assert pipe.qc_["baseline"]["n_samples"] == 21  # too few to z-score each epoch

    def test_mastoids_are_dropped_before_the_reference(self, run):
        info = mne.create_info(["M1", "M2"], run.info["sfreq"], "eeg")
        mastoids = np.random.default_rng(0).normal(0, 50e-6, (2, run.n_times))
        run.add_channels([mne.io.RawArray(mastoids, info, first_samp=run.first_samp, verbose=False)],
                         force_update_info=True)  # fmt: skip
        epochs = meu.Pipeline.preset("had-meeg", datatype="eeg", stage="epochs").fit_transform(run)
        assert "M1" not in epochs.ch_names
        # the remaining channels are average-referenced (they sum to zero)
        np.testing.assert_allclose(epochs.get_data("eeg").mean(axis=1), 0, atol=1e-12)

    def test_meg_needs_head_destination(self):
        with pytest.raises(ValueError, match="average_dev_head_t"):
            meu.Pipeline.preset("had-meeg", datatype="meg", stage="epochs")
        pipe = meu.Pipeline.preset(
            "had-meeg", datatype="meg", stage="epochs", head_destination=np.eye(4)
        )
        assert [n for n, _ in pipe.steps] == ["lowpass", "resample", "align", "epoch", "baseline"]

    def test_unknown_stage(self):
        with pytest.raises(ValueError, match="stage must be"):
            meu.Pipeline.preset("had-meeg", datatype="eeg", stage="decoding")
