"""Regression: environmental noise seen by reference sensors is removed (synthetic KIT)."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from mne.io.constants import FIFF

from meeg_utils import steps as S
from meeg_utils.io import detect_system
from meeg_utils.testing import check_step

N_MEG, N_REF, SFREQ = 64, 3, 500.0


@pytest.fixture(scope="module")
def kit():
    """KIT-like recording: brain signal + noise from 3 distant sources that the references see."""
    rng = np.random.default_rng(0)
    names = [f"MEG {i:03d}" for i in range(N_MEG)] + [f"REF {i}" for i in range(N_REF)]
    info = mne.create_info(names, SFREQ, ["mag"] * N_MEG + ["ref_meg"] * N_REF)
    for ch in info["chs"]:
        ch["coil_type"] = (
            FIFF.FIFFV_COIL_KIT_GRAD
            if ch["kind"] == FIFF.FIFFV_MEG_CH
            else FIFF.FIFFV_COIL_KIT_REF_MAG
        )
    n_times = int(20 * SFREQ)
    t = np.arange(n_times) / SFREQ
    brain = rng.normal(size=(N_MEG, 10)) @ rng.normal(size=(10, n_times)) * 1e-13
    sources = np.vstack([
        np.sin(2 * np.pi * 0.2 * t),               # slow drift (e.g. a passing vehicle)
        np.sin(2 * np.pi * 50 * t),                # power line
        np.cumsum(rng.normal(size=n_times)) / 50,  # random walk
    ]) * 5e-12  # fmt: skip
    noise_meg = rng.normal(size=(N_MEG, N_REF)) @ sources
    noise_ref = (
        rng.normal(size=(N_REF, N_REF)) @ sources + rng.normal(size=(N_REF, n_times)) * 1e-15
    )
    measured = np.vstack([brain + noise_meg, noise_ref])
    raw = mne.io.RawArray(measured, info, first_samp=200, verbose=False)
    raw.set_meas_date(1_600_000_000)
    raw.set_annotations(mne.Annotations([1.0], [0.5], ["stim"], orig_time=raw.info["meas_date"]))
    return raw, brain


def _error(raw, brain) -> float:
    data = raw.get_data(picks="meg")
    data = data - data.mean(axis=1, keepdims=True)
    truth = brain - brain.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(data - truth) / np.linalg.norm(truth))


def test_detected_as_kit(kit):
    assert detect_system(kit[0]) == "kit"


def test_contract(kit):
    check_step(S.Regression(), kit[0])


def test_removes_environmental_noise(kit):
    raw, brain = kit
    step = S.Regression()
    out = step.fit_transform(raw)
    assert _error(raw, brain) > 10
    assert _error(out, brain) < 0.05  # residual: reference-sensor noise
    assert step.qc_["n_artifact_channels"] == N_REF
    assert step.qc_["variance_removed_pct"] > 99
    assert step.qc_["power_change_db"]["mag"] < -20
    refs = mne.pick_types(raw.info, meg=False, ref_meg=True)
    assert np.array_equal(out.get_data(refs), raw.get_data(refs))  # predictors untouched


def test_coefficients_transfer(kit):
    """Coefficients fitted on one segment clean another."""
    raw, brain = kit
    step = S.Regression().fit(raw.copy().crop(0, 10))
    out = step.transform(raw.copy().crop(10.002, None))
    truth = brain[:, -out.n_times :]
    assert _error(out, truth) < 0.05


def test_bad_channels_are_left_alone(kit):
    raw, _ = kit
    raw = raw.copy()
    raw.info["bads"] = ["MEG 000"]
    out = S.Regression().fit_transform(raw)
    assert np.array_equal(out.get_data("MEG 000"), raw.get_data("MEG 000"))


def test_eog_regression(eeg):
    eeg = S.Reference("average").fit_transform(eeg)  # MNE requires the reference first
    step = S.Regression("eog")
    out = step.fit_transform(eeg)
    assert step.qc_["artifact_channels"] == ["EOG"]
    eog = eeg.get_data("EOG")[0]
    assert abs(np.corrcoef(eeg.get_data("Fp1")[0], eog)[0, 1]) > 0.5  # blinks
    assert abs(np.corrcoef(out.get_data("Fp1")[0], eog)[0, 1]) < 0.01


def test_needs_reference_channels(eeg):
    with pytest.raises(ValueError, match="no reference MEG channels"):
        S.Regression().fit(eeg)
