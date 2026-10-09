"""HeadAlign: validated against data simulated at known head positions.

Dipoles in a spherical head are projected (forward model) onto the sensors
twice: with the head at the reference position (ground truth) and after the
head moved. Mapping the moved recording to the reference position must
recover the ground truth.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest
from mne.transforms import Transform, rotation, translation

from meeg_utils import Pipeline
from meeg_utils import steps as S
from meeg_utils.io import average_dev_head_t
from meeg_utils.testing import check_step

ORIGIN = (0.0, 0.0, 0.04)


def _move(info: mne.Info, shift_mm: float, rot_deg: float) -> mne.Info:
    direction = np.array([1.0, -0.5, 0.7]) / np.linalg.norm([1.0, -0.5, 0.7])
    move = translation(*(direction * shift_mm / 1000)) @ rotation(
        np.deg2rad(rot_deg), 0, np.deg2rad(rot_deg / 2)
    )
    moved = info.copy()
    with moved._unlock():
        moved["dev_head_t"] = Transform("meg", "head", move @ info["dev_head_t"]["trans"])
    return moved


@pytest.fixture(scope="module")
def meg_info(neuromag_fname) -> mne.Info:
    """Every third sensor (mag + 2 grads) of the Neuromag recording, no projectors."""
    info = mne.io.read_info(neuromag_fname, verbose=False)
    meg = mne.pick_types(info, meg=True, exclude=[])
    info = mne.pick_info(info, meg[(np.arange(len(meg)) // 3) % 3 == 0])
    with info._unlock():
        info["bads"] = []
        info["projs"] = []
    return info


@pytest.fixture(scope="module")
def simulation(meg_info):
    """Factory: (measured raw after moving the head, ground-truth raw at the reference)."""
    rng = np.random.default_rng(0)
    n = 30
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction[:, 2] = np.abs(direction[:, 2])
    pos = np.array(ORIGIN) + direction * rng.uniform(0.03, 0.07, (n, 1))
    ori = rng.normal(size=(n, 3))
    ori -= (ori * direction).sum(1, keepdims=True) * direction  # tangential: visible to MEG
    ori /= np.linalg.norm(ori, axis=1, keepdims=True)
    dipoles = mne.Dipole(np.zeros(n), pos, np.full(n, 1e-8), ori, np.ones(n))
    sphere = mne.make_sphere_model(ORIGIN, head_radius=0.09, verbose=False)
    sources = rng.normal(size=(n, 1000)) * 1e-8

    def make(info: mne.Info) -> mne.io.BaseRaw:
        fwd, _ = mne.make_forward_dipole(dipoles, sphere, info, verbose=False)
        raw = mne.io.RawArray(fwd["sol"]["data"] @ sources, info, first_samp=300, verbose=False)
        raw.set_meas_date(1_600_000_000)
        raw.set_annotations(
            mne.Annotations([1.0], [0.5], ["stim"], orig_time=raw.info["meas_date"])
        )
        return raw

    truth = make(meg_info)
    cache: dict = {}

    def factory(shift_mm: float = 8.0, rot_deg: float = 4.0):
        if (shift_mm, rot_deg) not in cache:
            cache[shift_mm, rot_deg] = make(_move(meg_info, shift_mm, rot_deg))
        return cache[shift_mm, rot_deg].copy(), truth.copy()

    return factory


def _error(raw: mne.io.BaseRaw, truth: mne.io.BaseRaw, ch_type: str) -> float:
    picks = mne.pick_types(truth.info, meg=ch_type)
    diff = raw.get_data(picks) - truth.get_data(picks)
    return float(np.linalg.norm(diff) / np.linalg.norm(truth.get_data(picks)))


def test_contract(simulation):
    measured, truth = simulation()
    check_step(S.HeadAlign(truth.info["dev_head_t"]["trans"], origin=ORIGIN, mode="fast"), measured)


@pytest.mark.parametrize("ch_type", ["mag", "grad"])
def test_recovers_data_at_destination(simulation, ch_type):
    measured, truth = simulation()
    step = S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN)
    aligned = step.fit_transform(measured)

    unaligned = _error(measured, truth, ch_type)
    assert unaligned > 0.25  # the movement matters
    assert _error(aligned, truth, ch_type) < 0.05
    assert np.allclose(aligned.info["dev_head_t"]["trans"], truth.info["dev_head_t"]["trans"])


def test_reversed_mapping_is_worse_than_no_alignment(simulation, meg_info):
    """Mapping from the destination to the measured geometry (the arguments of
    ``_map_meg_or_eeg_channels`` swapped) moves the data the wrong way: it
    roughly doubles the error instead of removing it."""
    from mne.forward._field_interpolation import _map_meg_or_eeg_channels

    measured, truth = simulation()
    reversed_ = measured.copy()
    mapping = _map_meg_or_eeg_channels(truth.info, measured.info, mode="fast", origin=ORIGIN)
    reversed_._data = mapping @ measured.get_data()
    for ch_type in ("mag", "grad"):
        assert _error(reversed_, truth, ch_type) > 1.5 * _error(measured, truth, ch_type)


def test_qc_reports_movement(simulation):
    measured, truth = simulation()
    step = S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN, mode="fast").fit(measured)
    assert step.qc_["rotation_deg"] == pytest.approx(np.hypot(4.0, 2.0), abs=0.3)
    assert 4 < step.qc_["shift_mm"] < 15
    assert step.qc_["n_channels"] == len(measured.ch_names)

    still = S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN, mode="fast").fit(truth)
    assert still.qc_ == {**still.qc_, "shift_mm": 0.0, "rotation_deg": 0.0}


def test_large_movement_warns(simulation):
    measured, truth = simulation(25.0, 2.0)
    with pytest.warns(UserWarning, match="field mapping is less accurate"):
        S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN, mode="fast").fit(measured)


def test_bad_channels_are_left_unchanged(simulation):
    measured, truth = simulation()
    bad = measured.ch_names[5]
    measured.info["bads"] = [bad]
    aligned = S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN, mode="fast").fit_transform(
        measured
    )
    assert np.array_equal(aligned.get_data([bad]), measured.get_data([bad]))
    assert aligned.info["bads"] == [bad]


def test_destination_formats(simulation, tmp_path):
    measured, truth = simulation()
    fname = tmp_path / "ref_raw.fif"
    truth.save(fname, verbose=False)
    trans = truth.info["dev_head_t"]
    expected = S.HeadAlign(trans, origin=ORIGIN, mode="fast").fit_transform(measured).get_data()
    for destination in (str(fname), fname, trans["trans"], trans["trans"].tolist(), dict(trans)):
        out = S.HeadAlign(destination, origin=ORIGIN, mode="fast").fit_transform(measured)
        assert np.allclose(out.get_data(), expected)

    # the configuration round trip keeps the destination
    pipe = Pipeline([("align", S.HeadAlign(trans, origin=ORIGIN, mode="fast"))])
    pipe.to_yaml(tmp_path / "pipe.yaml")
    restored = Pipeline.from_yaml(tmp_path / "pipe.yaml")
    assert np.allclose(restored.fit_transform(measured).get_data(), expected)


@pytest.mark.parametrize(
    ("destination", "match"),
    [
        (None, "needs a destination"),
        (np.eye(3), "4x4"),
        (np.diag([2.0, 1.0, 1.0, 1.0]), "not a rigid"),
        (translation(2.0, 0, 0), "more than 1 m"),
        (Transform("head", "mri", np.eye(4)), "device to head"),
    ],
)
def test_invalid_destination(simulation, destination, match):
    measured, _ = simulation()
    with pytest.raises(ValueError, match=match):
        S.HeadAlign(destination, origin=ORIGIN).fit(measured)


def test_transform_refuses_other_head_position(simulation):
    measured, truth = simulation()
    step = S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN, mode="fast").fit(measured)
    with pytest.raises(ValueError, match="different head position"):
        step.transform(truth)


def test_eeg_is_not_supported(eeg):
    with pytest.raises(ValueError, match="eeg"):
        S.HeadAlign(np.eye(4)).fit(eeg)


def test_plot(simulation):
    import matplotlib.pyplot as plt

    measured, truth = simulation()
    step = S.HeadAlign(truth.info["dev_head_t"], origin=ORIGIN, mode="fast").fit(measured)
    figs = step.plot()
    assert "shift" in figs["positions"]._suptitle.get_text()
    plt.close("all")


class TestAverageDevHeadT:
    """average_dev_head_t weights runs by their good duration."""

    def _run(self, info, x_mm, duration, bad=0.0):
        moved = info.copy()
        with moved._unlock():
            moved["dev_head_t"] = Transform(
                "meg", "head", translation(x_mm / 1000, 0, 0) @ info["dev_head_t"]["trans"]
            )
        sfreq = info["sfreq"]
        raw = mne.io.RawArray(
            np.zeros((len(info.ch_names), int(duration * sfreq))), moved, verbose=False
        )
        if bad:
            raw.set_annotations(mne.Annotations([0.0], [bad], ["BAD_motion"]))
        return raw

    def test_equal_weights(self, meg_info):
        runs = [self._run(meg_info, -4, 2.0), self._run(meg_info, 6, 2.0)]
        mean = average_dev_head_t(runs)
        expected = translation(0.001, 0, 0) @ meg_info["dev_head_t"]["trans"]
        assert np.allclose(mean["trans"], expected, atol=1e-6)

    def test_bad_segments_reduce_weight(self, meg_info):
        runs = [self._run(meg_info, 0, 2.0, bad=1.0), self._run(meg_info, 3, 1.0)]
        mean = average_dev_head_t(runs)
        expected = translation(0.0015, 0, 0) @ meg_info["dev_head_t"]["trans"]
        assert np.allclose(mean["trans"], expected, atol=1e-6)

    def test_reads_paths(self, meg_info, tmp_path):
        paths = []
        for i, x in enumerate((-2, 2)):
            paths.append(tmp_path / f"run{i}_raw.fif")
            self._run(meg_info, x, 2.0).save(paths[-1], verbose=False)
        mean = average_dev_head_t(paths)
        assert np.allclose(mean["trans"], meg_info["dev_head_t"]["trans"], atol=1e-6)

    def test_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            average_dev_head_t([])
