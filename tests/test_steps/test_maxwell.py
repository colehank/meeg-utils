"""Maxwell: validated against data simulated with known sources and interference.

Dipoles in a spherical head are projected onto the 306 Neuromag sensors
(ground truth). External interference is a random field from the SSS
external basis (homogeneous fields and gradients, as from distant sources),
and head movement is simulated by projecting each segment from a different
head position.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest
from mne.transforms import Transform, rot_to_quat, rotation, translation

from meeg_utils import Pipeline
from meeg_utils import steps as S
from meeg_utils.testing import check_step

from ..simulation import CHPI_FREQS, make_chpi

ORIGIN = (0.0, 0.0, 0.04)
N_SEGMENTS = 4


def _move(trans: np.ndarray, shift_mm: float, rot_deg: float) -> np.ndarray:
    direction = np.array([1.0, -0.5, 0.7]) / np.linalg.norm([1.0, -0.5, 0.7])
    move = translation(*(direction * shift_mm / 1000)) @ rotation(
        np.deg2rad(rot_deg), 0, np.deg2rad(rot_deg / 2)
    )
    return move @ trans


def _with_head(info: mne.Info, trans: np.ndarray) -> mne.Info:
    info = info.copy()
    with info._unlock():
        info["dev_head_t"] = Transform("meg", "head", trans)
    return info


def _error(data: np.ndarray, truth: np.ndarray, info: mne.Info, ch_type: str) -> float:
    picks = mne.pick_types(info, meg=ch_type)
    return float(np.linalg.norm(data[picks] - truth[picks]) / np.linalg.norm(truth[picks]))


@pytest.fixture(scope="module")
def sim(neuromag_fname):
    """Ground truth, interference and moving-head data on the full Neuromag array."""
    info = mne.io.read_info(neuromag_fname, verbose=False)
    info = mne.pick_info(info, mne.pick_types(info, meg=True, exclude=[]))
    with info._unlock():
        info["bads"] = []
        info["projs"] = []
    rng = np.random.default_rng(0)
    n = 20
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction[:, 2] = np.abs(direction[:, 2])
    pos = np.array(ORIGIN) + direction * rng.uniform(0.03, 0.07, (n, 1))
    ori = rng.normal(size=(n, 3))
    ori -= (ori * direction).sum(1, keepdims=True) * direction  # tangential: visible to MEG
    ori /= np.linalg.norm(ori, axis=1, keepdims=True)
    dipoles = mne.Dipole(np.zeros(n), pos, np.full(n, 1e-8), ori, np.ones(n))
    sphere = mne.make_sphere_model(ORIGIN, head_radius=0.09, verbose=False)
    sfreq = info["sfreq"]
    n_times = int(6 * sfreq)
    sources = rng.normal(size=(n, n_times)) * 1e-8

    def project(inf: mne.Info, src: np.ndarray = sources) -> np.ndarray:
        fwd, _ = mne.make_forward_dipole(dipoles, sphere, inf, verbose=False)
        return fwd["sol"]["data"] @ src

    truth = project(info)

    basis = mne.preprocessing.compute_maxwell_basis(
        info, origin=ORIGIN, regularize=None, verbose=False
    )[0]
    n_in = 80  # int_order 8: 8 * 10 internal moments, then 15 external ones
    t = np.arange(n_times) / sfreq
    interference = basis[:, n_in:] @ (
        rng.normal(size=(15, 1)) * np.sin(2 * np.pi * 7 * t) + rng.normal(size=(15, 1)) * 0.5
    )
    mags = mne.pick_types(info, meg="mag")
    interference *= 10 * np.linalg.norm(truth[mags]) / np.linalg.norm(interference[mags])

    # head movement: N_SEGMENTS segments, each projected from its own head position
    dev_head = info["dev_head_t"]["trans"]
    heads = [_move(dev_head, s, r) for s, r in [(0, 0), (4, 2), (8, 4), (4, 3)]]
    length = n_times // N_SEGMENTS
    moving = np.zeros_like(truth)
    rows = []
    for i, trans in enumerate(heads):
        segment = slice(i * length, (i + 1) * length)
        moving[:, segment] = project(_with_head(info, trans), sources[:, segment])
        for sample in range(i * length, (i + 1) * length, int(0.25 * sfreq)):
            row = np.r_[sample / sfreq, rot_to_quat(trans[:3, :3]), trans[:3, 3], 1.0, 1e-3, 0.0]
            rows.append(row)
    destination = _move(dev_head, 6, 3)

    def raw(data: np.ndarray, inf: mne.Info = info) -> mne.io.BaseRaw:
        out = mne.io.RawArray(data, inf, verbose=False)
        out.set_meas_date(1_600_000_000)
        out.set_annotations(
            mne.Annotations([1.0], [0.5], ["stim"], orig_time=out.info["meas_date"])
        )
        return out

    return {
        "info": info,
        "raw": raw,
        "truth": truth,
        "interference": interference,
        "moving": moving,
        "head_pos": np.array(rows),
        "destination": destination,
        "truth_at_destination": project(_with_head(info, destination)),
    }


def _sss(**kwargs) -> S.Maxwell:
    return S.Maxwell(origin=ORIGIN, cross_talk=None, calibration=None, **kwargs)


def test_contract(sim):
    check_step(_sss(), sim["raw"](sim["truth"] + sim["interference"]))


def test_contract_movement(sim):
    check_step(_sss(head_pos=sim["head_pos"][::10]), sim["raw"](sim["moving"]))


def test_keeps_brain_signal(sim):
    out = _sss().fit_transform(sim["raw"](sim["truth"]))
    for ch_type in ("mag", "grad"):
        # reconstruction error of the regularized internal basis
        assert _error(out.get_data(), sim["truth"], sim["info"], ch_type) < 0.08


@pytest.mark.parametrize("ch_type", ["mag", "grad"])
def test_removes_external_interference(sim, ch_type):
    measured = sim["raw"](sim["truth"] + sim["interference"])
    step = _sss()
    out = step.fit_transform(measured)
    info = sim["info"]
    assert _error(measured.get_data(), sim["truth"], info, ch_type) > 0.35
    assert _error(out.get_data(), sim["truth"], info, ch_type) < 0.08
    assert step.qc_["power_change_db"]["mag"] < -15  # interference was 10x the signal
    assert 60 <= step.qc_["n_basis"] <= 80


def test_reconstructs_bad_channels(sim):
    measured = sim["raw"](sim["truth"].copy())
    bads = ["MEG 0113", "MEG 1442", "MEG 2343"]
    rng = np.random.default_rng(1)
    for ch in bads:
        idx = measured.ch_names.index(ch)
        measured._data[idx] = rng.normal(0, 50 * measured._data[idx].std(), measured.n_times)
    measured.info["bads"] = bads
    step = _sss()
    out = step.fit_transform(measured)
    assert out.info["bads"] == []
    assert step.qc_["reconstructed"] == bads
    picks = [out.ch_names.index(ch) for ch in bads]
    truth = sim["truth"][picks]
    assert np.linalg.norm(out.get_data(picks) - truth) / np.linalg.norm(truth) < 0.15


def test_movement_compensation(sim):
    measured = sim["raw"](sim["moving"])
    info = sim["info"]
    static = _sss().fit_transform(measured)
    step = _sss(head_pos=sim["head_pos"])
    compensated = step.fit_transform(measured)
    for ch_type in ("mag", "grad"):
        uncorrected = _error(static.get_data(), sim["truth"], info, ch_type)
        corrected = _error(compensated.get_data(), sim["truth"], info, ch_type)
        assert uncorrected > 0.15
        assert corrected < 0.1
    assert step.qc_["head_movement"]["max_displacement_mm"] == pytest.approx(9.4, abs=0.2)


def test_destination(sim):
    measured = sim["raw"](sim["truth"])
    step = _sss(destination=sim["destination"])
    out = step.fit_transform(measured)
    truth = sim["truth_at_destination"]
    info = sim["info"]
    for ch_type in ("mag", "grad"):
        assert _error(measured.get_data(), truth, info, ch_type) > 0.2
        assert _error(out.get_data(), truth, info, ch_type) < 0.08
    assert np.allclose(out.info["dev_head_t"]["trans"], sim["destination"])
    origin = np.r_[ORIGIN, 1.0]  # head origin in device coordinates, before and after
    before = np.linalg.inv(measured.info["dev_head_t"]["trans"]) @ origin
    after = np.linalg.inv(sim["destination"]) @ origin
    shift = 1000 * np.linalg.norm((before - after)[:3])
    assert step.qc_["destination_shift_mm"] == pytest.approx(shift, abs=0.01)


def test_tsss_removes_nearby_artifact(sim):
    """A source just outside the head, close to the sensors (like dental work), is
    poorly described by the SSS bases and survives SSS; tSSS removes it because its
    time course appears in both the internal space and the residual."""
    info, truth = sim["info"], sim["truth"]
    rng = np.random.default_rng(3)
    t = np.arange(truth.shape[1]) / info["sfreq"]
    pos, ori = np.array([[0.0, 0.1, 0.04]]), np.array([[1.0, 0.0, 0.0]])
    dipole = mne.Dipole(np.zeros(1), pos, np.ones(1), ori, np.ones(1))
    fwd, _ = mne.make_forward_dipole(
        dipole, mne.make_sphere_model(ORIGIN, head_radius=None, verbose=False), info, verbose=False
    )
    time_course = np.sign(np.sin(2 * np.pi * 1.3 * t)) + 0.3 * rng.normal(size=t.size)
    artifact = fwd["sol"]["data"] @ time_course[None]
    mags = mne.pick_types(info, meg="mag")
    artifact *= 5 * np.linalg.norm(truth[mags]) / np.linalg.norm(artifact[mags])
    noise = np.zeros_like(truth)  # sensor noise: without it the tSSS residual is empty
    for ch_type in ("mag", "grad"):
        picks = mne.pick_types(info, meg=ch_type)
        noise[picks] = rng.normal(size=(len(picks), t.size)) * 0.1 * truth[picks].std()
    measured = sim["raw"](truth + artifact + noise)

    sss = _sss().fit_transform(measured)
    step = _sss(st_duration=3.0)  # 1802 samples
    tsss = step.fit_transform(measured)
    assert step.qc_["tsss"]
    for ch_type in ("mag", "grad"):
        assert _error(sss.get_data(), truth, info, ch_type) > 3
        assert _error(tsss.get_data(), truth, info, ch_type) < 0.6


@pytest.mark.filterwarnings("ignore:.*filter_length.*:RuntimeWarning")
def test_head_positions_from_chpi(neuromag_fname):
    raw, _, _ = make_chpi(neuromag_fname)
    step = S.Maxwell(head_pos="chpi", cross_talk=None, calibration=None)
    out = step.fit_transform(raw)
    movement = step.qc_["head_movement"]
    assert movement["max_displacement_mm"] == pytest.approx(8.0, abs=1.5)
    assert step.qc_["chpi_removed"]
    # the coil signals are gone
    picks = mne.pick_types(raw.info, meg="mag")
    for freq in CHPI_FREQS:
        before = _band_power(raw, picks, freq)
        after = _band_power(out, picks, freq)
        assert after < before / 100
    assert set(step.plot()) == {"psd", "head_positions", "displacement"}


def _band_power(raw: mne.io.BaseRaw, picks: np.ndarray, freq: float) -> float:
    psd = raw.compute_psd(picks=picks, fmin=freq - 0.5, fmax=freq + 0.5, verbose=False)
    return float(psd.get_data().mean())


def test_neuromag_recording(neuromag):
    """On real data: EEG and stim channels are untouched, the MEG rank drops."""
    step = S.Maxwell(cross_talk=None, calibration=None)
    out = step.fit_transform(neuromag)
    eeg = mne.pick_types(neuromag.info, meg=False, eeg=True, stim=True)
    assert np.array_equal(out.get_data(eeg), neuromag.get_data(eeg))
    rank = mne.compute_rank(out, rank=None, info=out.info, verbose=False)
    assert rank["meg"] == step.qc_["n_basis"]
    assert step.qc_["projs_removed"] > 0  # the recording's MEG SSP projectors no longer apply


def test_pipeline_yaml(tmp_path, sim):
    pipe = Pipeline(
        [
            ("bads", S.BadChannels("maxwell", cross_talk=None, calibration=None, origin=ORIGIN)),
            ("sss", _sss(st_duration=3.0)),
        ]
    )
    pipe.to_yaml(tmp_path / "pipe.yaml")
    restored = Pipeline.from_yaml(tmp_path / "pipe.yaml")
    assert restored.to_dict() == pipe.to_dict()


class TestErrors:
    def test_other_systems(self, eeg):
        with pytest.raises(ValueError, match="applies to"):
            _sss().fit(eeg)

    def test_head_pos_needs_head_frame(self, sim):
        with pytest.raises(ValueError, match="coord_frame='head'"):
            _sss(head_pos=sim["head_pos"], coord_frame="meg").fit(sim["raw"](sim["truth"]))

    def test_unknown_head_pos(self, sim):
        with pytest.raises(ValueError, match="head_pos must be"):
            _sss(head_pos="cHPI-please").fit(sim["raw"](sim["truth"]))

    def test_no_chpi(self, sim):
        step = _sss(head_pos="chpi").fit(sim["raw"](sim["truth"]))
        with pytest.raises(ValueError, match="cHPI"):
            step.transform(sim["raw"](sim["truth"]))

    def test_odd_tsss_buffer(self, sim):
        with pytest.raises(ValueError, match=r"1201 samples .* st_duration=2\.0"):
            _sss(st_duration=2.0).fit_transform(sim["raw"](sim["truth"]))

    def test_files_outside_bids(self, sim):
        with pytest.raises(FileNotFoundError, match="not in a BIDS dataset"):
            S.Maxwell(origin=ORIGIN).fit(sim["raw"](sim["truth"]))

    def test_plots_before_transform(self, sim):
        step = _sss().fit(sim["raw"](sim["truth"]))
        assert step.plot() == {}
        with pytest.raises(ValueError, match="has not transformed"):
            step.plot("psd")
        with pytest.raises(ValueError, match="head_pos=None"):
            step.plot("head_positions")
