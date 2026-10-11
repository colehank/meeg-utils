"""HFC: validated on a simulated on-head OPM array with known interference."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from meeg_utils import steps as S
from meeg_utils.testing import check_step

from ..simulation import make_opm


def _sensors(info: mne.Info) -> tuple[np.ndarray, np.ndarray]:
    loc = np.array([ch["loc"] for ch in info["chs"]])
    return loc[:, :3], loc[:, 9:12]


def _homogeneous(info: mne.Info, truth: np.ndarray, factor: float = 30.0) -> np.ndarray:
    """A slowly varying uniform field (as from distant sources), ``factor`` x the signal."""
    t = np.arange(truth.shape[1]) / info["sfreq"]
    field = np.c_[np.sin(2 * np.pi * 0.5 * t), np.cos(2 * np.pi * 1.1 * t), 0.5 * np.ones_like(t)]
    out = _sensors(info)[1] @ field.T
    return out * factor * np.linalg.norm(truth) / np.linalg.norm(out)


def _gradient(info: mne.Info, truth: np.ndarray, factor: float = 30.0) -> np.ndarray:
    """A linear field B = G r (G symmetric and traceless: curl- and divergence-free)."""
    grad = np.array([[1.0, 0.4, -0.2], [0.4, -0.3, 0.5], [-0.2, 0.5, -0.7]])
    pos, normal = _sensors(info)
    t = np.arange(truth.shape[1]) / info["sfreq"]
    out = ((normal * (pos @ grad.T)).sum(1))[:, None] * np.sin(2 * np.pi * 0.3 * t)[None]
    return out * factor * np.linalg.norm(truth) / np.linalg.norm(out)


def _error(data: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(data - truth) / np.linalg.norm(truth))


@pytest.fixture(scope="module")
def triaxial():
    return make_opm(axes=3)


def _raw(info: mne.Info, data: np.ndarray) -> mne.io.BaseRaw:
    raw = mne.io.RawArray(data, info, first_samp=100, verbose=False)
    raw.set_meas_date(1_600_000_000)
    raw.set_annotations(mne.Annotations([1.0], [0.5], ["stim"], orig_time=raw.info["meas_date"]))
    return raw


def test_contract(triaxial):
    info, truth = triaxial
    check_step(S.HFC(), _raw(info, truth + _homogeneous(info, truth)))


def test_removes_homogeneous_field(triaxial):
    info, truth = triaxial
    measured = _raw(info, truth + _homogeneous(info, truth))
    step = S.HFC()
    out = step.fit_transform(measured)
    assert _error(measured.get_data(), truth) > 10
    assert _error(out.get_data(), truth) < 0.1  # brain signal kept on triaxial sensors
    # the interference is removed exactly: what remains is the projected brain signal
    assert np.allclose(out.get_data(), step.projector_ @ truth, atol=1e-6 * np.abs(truth).max())
    assert step.qc_["n_components"] == 3
    assert step.qc_["axes_per_location"] == 3
    assert step.qc_["shielding_db"] > 30
    assert all(p["active"] for p in out.info["projs"])


def test_gradient_needs_order_2(triaxial):
    info, truth = triaxial
    measured = _raw(info, truth + _gradient(info, truth))
    first = S.HFC(order=1).fit_transform(measured)
    second = S.HFC(order=2).fit_transform(measured)
    assert _error(first.get_data(), truth) > 5
    assert _error(second.get_data(), truth) < 0.25


def test_single_axis_array_warns():
    info, truth = make_opm(axes=1, n_locations=64)
    with pytest.warns(UserWarning, match="single sensor axis"):
        out = S.HFC().fit_transform(_raw(info, truth + _homogeneous(info, truth)))
    assert _error(out.get_data(), truth) > 0.5  # brain signal lost with the interference


def test_bad_channels_untouched(triaxial):
    info, truth = triaxial
    measured = _raw(info, truth + _homogeneous(info, truth))
    measured.info["bads"] = ["OPM000"]
    out = S.HFC().fit_transform(measured)
    assert np.array_equal(out.get_data("OPM000"), measured.get_data("OPM000"))
    assert _error(out.get_data(), truth) > 1  # the bad channel keeps its interference
    good = mne.pick_types(out.info, meg=True, exclude="bads")
    assert _error(out.get_data(good), truth[good]) < 0.1


def test_epochs(triaxial):
    info, truth = triaxial
    raw = _raw(info, truth + _homogeneous(info, truth))
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)
    out = S.HFC().fit_transform(epochs)
    clean = S.HFC().fit_transform(raw)
    expected = mne.make_fixed_length_epochs(clean, duration=1.0, preload=True, verbose=False)
    assert np.allclose(out.get_data(), expected.get_data())


def test_other_systems(neuromag):
    with pytest.raises(ValueError, match="applies to"):
        S.HFC().fit(neuromag)
