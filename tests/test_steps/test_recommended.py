"""Recommended presets: configuration (traced to their sources) and end-to-end runs."""

from __future__ import annotations

import numpy as np
import pytest

import meeg_utils as meu
from meeg_utils import Pipeline
from meeg_utils.presets import _recommended as R

from ..simulation import make_erp

NAMES = ("eeg-erp", "eeg-rest", "meg-erp", "meg-rest")


def test_listed():
    assert set(NAMES) <= set(meu.presets.available())


@pytest.mark.parametrize("name", NAMES)
def test_yaml_round_trip(name, tmp_path):
    options = {"system": "neuromag"} if name.startswith("meg") else {}
    for stage, extra in (
        ("preprocessing", {}),
        ("epochs", {"event_id": ["a"]} if name.endswith("erp") else {"epoch_duration": 2.0}),
    ):
        pipe = Pipeline.preset(name, stage=stage, **options, **extra)
        pipe.to_yaml(tmp_path / "p.yaml")
        assert Pipeline.from_yaml(tmp_path / "p.yaml").to_dict() == pipe.to_dict()


def test_eeg_preprocessing_parameters():
    pipe = Pipeline.preset("eeg-erp")
    assert [n for n, _ in pipe.steps] == [
        "bridged", "highpass", "line_noise", "bads", "interpolate", "reference", "ica",
    ]  # fmt: skip
    assert pipe["highpass"].l_freq == R.HIGHPASS == 0.1  # Tanner et al. 2015
    assert pipe["highpass"].h_freq is None  # ICLabel needs the 1-100 Hz band
    assert pipe["reference"].eeg == "average"  # ICLabel's training data
    ica = pipe["ica"]
    assert (ica.method, ica.labeler, ica.threshold) == ("infomax", "iclabel", 0.8)
    assert (ica.fit_l_freq, ica.fit_h_freq) == (1.0, 100.0)


@pytest.mark.parametrize(
    ("system", "noise"),
    [
        ("neuromag", ["bads", "sss"]),
        ("ctf", ["bads", "interpolate", "reference"]),
        ("kit", ["regression"]),
    ],
)
def test_meg_preprocessing_by_system(system, noise):
    pipe = Pipeline.preset("meg-erp", system=system)
    assert [n for n, _ in pipe.steps] == [*noise, "highpass", "resample", "line_noise", "ica"]
    ica = pipe["ica"]
    assert pipe["resample"].sfreq == 250.0  # MEGnet's requirements
    assert (ica.n_components, ica.method, ica.labeler) == (20, "infomax", "megnet")


def test_meg_needs_system():
    with pytest.raises(ValueError, match="system="):
        Pipeline.preset("meg-erp")
    with pytest.raises(ValueError, match="no recommended pipeline"):
        Pipeline.preset("meg-rest", system="opm")


def test_epochs_stages():
    erp = Pipeline.preset("eeg-erp", stage="epochs", event_id=["target"])
    assert [n for n, _ in erp.steps] == ["lowpass", "epoch", "autoreject"]
    assert erp["lowpass"].h_freq == 40.0
    assert (erp["epoch"].tmin, erp["epoch"].tmax, erp["epoch"].baseline) == (-0.2, 0.5, (None, 0.0))
    assert erp["autoreject"].n_interpolate == [4, 8, 16]

    rest = Pipeline.preset(
        "meg-rest", stage="epochs", epoch_duration=4.0, head_destination=np.eye(4)
    )
    assert [n for n, _ in rest.steps] == ["align", "epoch", "autoreject"]
    assert rest["epoch"].duration == 4.0

    with pytest.raises(ValueError, match="event_id"):
        Pipeline.preset("eeg-erp", stage="epochs")
    with pytest.raises(ValueError, match="epoch_duration"):
        Pipeline.preset("eeg-rest", stage="epochs")
    with pytest.raises(ValueError, match="stage"):
        Pipeline.preset("eeg-erp", stage="analysis")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_eeg_erp_end_to_end():
    """Both stages on a simulated oddball experiment: the P300 survives."""
    raw = make_erp()
    raw.info["bads"] = []
    clean = Pipeline.preset("eeg-erp").fit_transform(raw)
    assert clean.info["custom_ref_applied"]

    pipe = Pipeline.preset("eeg-erp", stage="epochs", event_id=["stim/target", "stim/standard"])
    epochs = pipe.fit_transform(clean)
    assert pipe.qc_["autoreject"]["fraction_dropped"] < 0.2
    target = epochs["stim/target"].average().get_data("Pz")[0]
    standard = epochs["stim/standard"].average().get_data("Pz")[0]
    times = epochs.times
    peak = times[np.argmax(target)]
    assert 0.25 < peak < 0.35
    window = (times > 0.25) & (times < 0.35)
    assert (target - standard)[
        window
    ].mean() > 3e-6  # simulated difference: 7 µV before re-referencing


@pytest.mark.filterwarnings("ignore:filter_length.*longer than the signal")
def test_meg_neuromag_until_ica(neuromag):
    """MEGnet needs >= 60 s, so the short test recording stops before ICA."""
    neuromag.info["line_freq"] = 60.0
    pipe = Pipeline.preset("meg-erp", system="neuromag")[:-1]
    pipe.set_params(  # not a BIDS dataset: no cross-talk / calibration files
        bads__cross_talk=None, bads__calibration=None, sss__cross_talk=None, sss__calibration=None
    )
    out = pipe.fit_transform(neuromag.pick(["meg", "eeg", "stim"]))
    assert "MEG 2443" in pipe.qc_["bads"]["meg"]["bads"]
    assert "MEG 2443" in pipe.qc_["sss"]["reconstructed"]  # rebuilt by SSS
    assert out.info["sfreq"] == 250.0
    assert out.ch_names == neuromag.ch_names
