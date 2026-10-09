"""ByChannelType: branches give the same result as processing each channel type alone."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from sklearn.base import clone

from meeg_utils import Pipeline, report
from meeg_utils import steps as S
from meeg_utils.testing import check_step

pytestmark = pytest.mark.filterwarnings("ignore:.*filter_length.*:RuntimeWarning")


def _branches() -> dict:
    return {
        "meg": [("filter", S.Filter(1.0, 40.0))],
        "eeg": [("filter", S.Filter(0.1, 30.0)), ("ref", S.Reference("average"))],
    }


def test_contract(neuromag):
    check_step(S.ByChannelType(_branches()), neuromag)


def test_contract_epochs(eeg):
    epochs = mne.make_fixed_length_epochs(eeg, duration=2.0, preload=True, verbose=False)
    check_step(S.ByChannelType({"eeg": S.Baseline((None, 0.5))}), epochs)


def test_same_as_each_type_alone(neuromag):
    step = S.ByChannelType(_branches())
    out = step.fit_transform(neuromag)

    meg = Pipeline(_branches()["meg"]).fit_transform(neuromag.copy().pick("meg"))
    eeg = Pipeline(_branches()["eeg"]).fit_transform(neuromag.copy().pick("eeg"))
    assert np.allclose(out.get_data(meg.ch_names), meg.get_data())
    assert np.allclose(out.get_data(eeg.ch_names), eeg.get_data())
    stim = mne.pick_types(neuromag.info, meg=False, stim=True)
    assert np.array_equal(out.get_data(stim), neuromag.get_data(stim))
    assert out.ch_names == neuromag.ch_names
    assert out.info["custom_ref_applied"]
    assert set(step.qc_) == {"meg", "eeg"} and set(step.qc_["eeg"]) == {"filter", "ref"}


def test_transform_reuses_fitted_branches(neuromag):
    step = S.ByChannelType(_branches()).fit(neuromag)
    assert np.allclose(
        step.transform(neuromag).get_data(),
        S.ByChannelType(_branches()).fit_transform(neuromag).get_data(),
    )


def test_bad_channels_are_carried_back(eeg_bad):
    step = S.ByChannelType({"eeg": S.BadChannels("prep", ransac=False)})
    out = step.fit_transform(eeg_bad)
    assert {"C4", "P8"} <= set(out.info["bads"])
    assert "EOG" not in out.info["bads"]


def test_maxwell_branch(neuromag):
    """Maxwell filtering only the MEG: its history and projector removal reach the result."""
    step = S.ByChannelType({"meg": S.Maxwell(cross_talk=None, calibration=None)})
    out = step.fit_transform(neuromag)
    assert out.info["proc_history"] and out.info["proc_history"][0]["max_info"]["sss_info"]
    assert not any(p["desc"].startswith("PCA") for p in out.info["projs"])
    eeg = mne.pick_types(neuromag.info, meg=False, eeg=True)
    assert np.array_equal(out.get_data(eeg), neuromag.get_data(eeg))
    rank = mne.compute_rank(out, rank="info", info=out.info, verbose=False)
    assert rank["meg"] == step.qc_["meg"]["maxwell"]["n_basis"]


class TestParams:
    def test_nested(self):
        pipe = Pipeline([("by_type", S.ByChannelType(_branches()))])
        pipe.set_params(by_type__eeg__ref__eeg=["Cz"], by_type__meg__filter__h_freq=30.0)
        branches = pipe["by_type"].branches
        assert branches["eeg"][1][1].eeg == ["Cz"]
        assert branches["meg"][0][1].h_freq == 30.0
        params = pipe.get_params()
        assert params["by_type__eeg__ref__eeg"] == ["Cz"]

    def test_single_step_branch(self):
        step = S.ByChannelType({"eeg": S.Reference("average")})
        step.set_params(eeg__eeg=["Cz"])
        assert step.get_params()["eeg__eeg"] == ["Cz"]

    def test_replace_branch_step(self):
        step = S.ByChannelType(_branches())
        step.set_params(eeg__ref=S.Reference(["Cz"]))
        assert step.branches["eeg"][1][1].eeg == ["Cz"]

    def test_clone_is_unfitted(self, eeg):
        step = S.ByChannelType({"eeg": [("ref", S.Reference("average"))]}).fit(eeg)
        cloned = clone(step)
        assert not hasattr(cloned, "pipelines_")
        assert not hasattr(cloned.branches["eeg"][0][1], "system_")
        assert cloned.branches["eeg"][0][1] is not step.branches["eeg"][0][1]

    def test_yaml(self, tmp_path):
        pipe = Pipeline([("by_type", S.ByChannelType(_branches()))])
        pipe.to_yaml(tmp_path / "pipe.yaml")
        restored = Pipeline.from_yaml(tmp_path / "pipe.yaml")
        assert restored.to_dict() == pipe.to_dict()
        branch = restored["by_type"].branches["eeg"]
        assert isinstance(branch[1][1], S.Reference)


class TestErrors:
    def test_overlapping_branches(self, neuromag):
        with pytest.raises(ValueError, match="in both"):
            S.ByChannelType({"meg": S.Filter(1.0, None), "mag": S.Filter(1.0, None)}).fit(neuromag)

    def test_missing_channels(self, eeg):
        with pytest.raises(ValueError, match="no meg channels"):
            S.ByChannelType({"meg": S.Filter(1.0, None)}).fit(eeg)
        step = S.ByChannelType(
            {"meg": S.Filter(1.0, None), "eeg": S.Filter(1.0, None)}, on_missing="ignore"
        )
        assert set(step.fit(eeg).pipelines_) == {"eeg"}

    def test_steps_changing_times(self, eeg):
        with pytest.raises(ValueError, match="outside ByChannelType"):
            S.ByChannelType({"eeg": S.Resample(100.0)}).fit(eeg)

    def test_unknown_group(self, eeg):
        with pytest.raises(ValueError, match="Unknown channel group"):
            S.ByChannelType({"brain": S.Filter(1.0, None)}).fit(eeg)

    def test_bad_branch(self, eeg):
        with pytest.raises(TypeError, match="list of"):
            S.ByChannelType({"eeg": ["not a step"]}).fit(eeg)


def test_plots_and_report(neuromag):
    pipe = Pipeline([("by_type", S.ByChannelType(_branches()))])
    pipe.fit_transform(neuromag)
    step = pipe["by_type"]
    assert set(step.plot()) == {
        "meg/filter/response",
        "meg/filter/psd",
        "eeg/filter/response",
        "eeg/filter/psd",
    }
    one = step.plot("eeg/filter/psd")
    assert set(one) == {"eeg/filter/psd"}
    html = report.build(pipe, title="by type")
    assert "eeg/filter/psd" in html.html[-1] or any("eeg/filter" in h for h in html.html)
