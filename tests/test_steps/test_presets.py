"""Presets: configuration and end-to-end runs."""

from __future__ import annotations

import numpy as np
import pytest
from mne.io import BaseRaw

import meeg_utils as meu
from meeg_utils import Pipeline
from meeg_utils import steps as S

from ..simulation import FLAT, NOISY


def test_available_lists_presets():
    presets = meu.presets.available()
    assert "had-meeg" in presets
    assert all(isinstance(summary, str) and summary for summary in presets.values())


def test_unknown_preset():
    with pytest.raises(ValueError, match="available"):
        Pipeline.preset("no-such-preset")


@pytest.mark.parametrize("datatype", [None, "fnirs"])
def test_had_meeg_needs_datatype(datatype):
    with pytest.raises((TypeError, ValueError)):
        Pipeline.preset("had-meeg", **({} if datatype is None else {"datatype": datatype}))


@pytest.mark.parametrize(
    ("datatype", "bads", "labeler"),
    [
        ("meg", S.BadChannels(method="maxwell"), "megnet"),
        ("eeg", S.BadChannels(method="prep"), "iclabel"),
    ],
)
def test_had_meeg_configuration(datatype, bads, labeler, tmp_path):
    pipe = Pipeline.preset("had-meeg", datatype=datatype)
    names = [name for name, _ in pipe.steps]
    assert names == ["filter", "resample", "bads", "interpolate", "line_noise", "reference", "ica"]
    assert (pipe["filter"].l_freq, pipe["filter"].h_freq) == (0.1, 100.0)
    assert pipe["resample"].sfreq == 250.0
    assert pipe["bads"].method == bads.method
    assert pipe["line_noise"].method == "zapline-plus"
    assert pipe["line_noise"].fline is None  # from the data
    assert pipe["ica"].labeler == labeler
    assert pipe["ica"].n_components == 20
    assert (pipe["ica"].fit_l_freq, pipe["ica"].fit_h_freq) == (1.0, 100.0)

    # presets are plain pipelines: YAML round trip, fresh objects each call
    pipe.to_yaml(tmp_path / "preset.yaml")
    assert Pipeline.from_yaml(tmp_path / "preset.yaml").to_dict() == pipe.to_dict()
    assert Pipeline.preset("had-meeg", datatype=datatype)["ica"] is not pipe["ica"]


def test_had_meeg_eeg_end_to_end(eeg_bad: BaseRaw):
    pipe = Pipeline.preset("had-meeg", datatype="eeg")
    out = pipe.fit_transform(eeg_bad)

    qc = pipe.qc_
    assert set(qc["bads"]["eeg"]["bads"]) == {NOISY, FLAT}
    assert set(qc["interpolate"]["interpolated"]) == {NOISY, FLAT}
    assert qc["line_noise"]["eeg"]["suppression_db"] > 10
    assert qc["ica"]["n_components"] <= 20
    assert out.info["sfreq"] == 250.0
    assert out.info["custom_ref_applied"] == 1  # average reference
    assert out.first_samp == eeg_bad.first_samp
    eeg = out.get_data("eeg")
    np.testing.assert_allclose(eeg.mean(axis=0), 0, atol=1e-12)  # still average-referenced
    assert np.all(eeg.std(axis=1) > 0)  # the flat channel was repaired

    # manual review (docs/presets/had-meeg.md): relabel, then transform again
    ica = pipe["ica"]
    keep = next(i for i in range(len(ica.labels_)) if i not in ica.exclude_)
    ica.relabel({keep: "eye blink"})
    assert keep in ica.exclude_
    assert pipe.qc_["ica"]["manual_labels"] == {str(keep): "eye blink"}
    relabeled = pipe.transform(eeg_bad)
    assert not np.allclose(relabeled.get_data("eeg"), eeg)


@pytest.mark.filterwarnings("ignore:filter_length.*longer than the signal")
def test_had_meeg_meg_until_ica(neuromag: BaseRaw):
    """MEGnet needs >= 60 s, so the short test recording stops before ICA."""
    neuromag.info["line_freq"] = 60.0
    pipe = Pipeline.preset("had-meeg", datatype="meg")[:-1]
    out = pipe.fit_transform(neuromag.pick(["meg", "stim"]))
    assert "MEG 2443" in pipe.qc_["bads"]["meg"]["bads"]
    assert "MEG 2443" in pipe.qc_["interpolate"]["interpolated"]
    assert out.info["sfreq"] == 250.0
    assert out.ch_names == neuromag.ch_names
