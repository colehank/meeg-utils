"""BridgedElectrodes, and the BIDS lookup of Maxwell cross-talk / calibration files."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from meeg_utils import steps as S
from meeg_utils.steps.channels import _maxwell_file
from meeg_utils.testing import check_step

from ..simulation import make_eeg


def _bridged_eeg():
    raw = make_eeg()
    rng = np.random.default_rng(1)
    names = raw.ch_names
    for a, b in (("F3", "Fz"), ("Cz", "CP1")):
        raw._data[names.index(b)] = raw._data[names.index(a)] + rng.normal(0, 0.3e-6, raw.n_times)
    return raw


class TestBridgedElectrodes:
    def test_contract(self):
        check_step(S.BridgedElectrodes(), _bridged_eeg())

    def test_repairs_bridges(self):
        raw = _bridged_eeg()
        truth = make_eeg()  # the same data before bridging
        step = S.BridgedElectrodes()
        out = step.fit_transform(raw)
        assert step.qc_["pairs"] == [["CP1", "Cz"], ["F3", "Fz"]]
        assert step.qc_["interpolated"] == ["CP1", "Cz", "F3", "Fz"]
        assert out.ch_names == raw.ch_names
        fz, f3 = out.get_data(["Fz", "F3"])
        # the two electrodes are no longer identical...
        assert np.std(fz - f3) > 10 * np.std(raw.get_data("Fz")[0] - raw.get_data("F3")[0])
        # ...and the interpolated Fz resembles the true Fz
        assert np.corrcoef(fz, truth.get_data("Fz")[0])[0, 1] > 0.8

    def test_large_groups(self):
        raw = _bridged_eeg()
        with pytest.raises(ValueError, match="larger than bad_limit=1"):
            S.BridgedElectrodes(bad_limit=1).fit(raw)
        step = S.BridgedElectrodes(bad_limit=1, large_groups="bad")
        out = step.fit_transform(raw)
        assert out.info["bads"] == ["F3", "Fz", "Cz", "CP1"]
        np.testing.assert_array_equal(out.get_data(), raw.get_data())

    def test_no_bridges(self, eeg):
        out = S.BridgedElectrodes().fit_transform(eeg)
        np.testing.assert_array_equal(out.get_data(), eeg.get_data())


class TestMaxwellFiles:
    @pytest.fixture
    def bids_meg(self, neuromag_fname, tmp_path) -> BIDSPath:
        raw = mne.io.read_raw_fif(neuromag_fname, verbose=False).crop(0, 2).load_data()
        bids_path = BIDSPath(subject="01", task="rest", datatype="meg", root=tmp_path / "bids")
        write_raw_bids(
            raw, bids_path, allow_preload=True, format="FIF", overwrite=True, verbose=False
        )
        return bids_path

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_found_in_bids(self, bids_meg):
        meg_dir = Path(bids_meg.root) / "sub-01" / "meg"
        crosstalk = meg_dir / "sub-01_acq-crosstalk_meg.fif"
        calibration = meg_dir / "sub-01_acq-calibration_meg.dat"
        crosstalk.write_bytes(b"")
        calibration.write_text("")
        raw = mne.io.read_raw_fif(bids_meg.fpath, verbose=False)
        assert _maxwell_file("auto", raw, "cross_talk", "neuromag") == crosstalk
        assert _maxwell_file("auto", raw, "calibration", "neuromag") == calibration

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_missing_in_bids(self, bids_meg):
        raw = mne.io.read_raw_fif(bids_meg.fpath, verbose=False)
        with pytest.raises(FileNotFoundError, match="No cross-talk file for sub-01"):
            _maxwell_file("auto", raw, "cross_talk", "neuromag")

    def test_explicit_values_and_other_systems(self, neuromag):
        assert _maxwell_file(None, neuromag, "cross_talk", "neuromag") is None
        assert _maxwell_file("ct.fif", neuromag, "cross_talk", "neuromag") == "ct.fif"
        assert _maxwell_file("auto", neuromag, "cross_talk", "ctf") is None
