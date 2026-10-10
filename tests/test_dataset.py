"""Dataset selection and whole-dataset processing (preprocess, epoch, combine)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mne
import numpy as np
import pytest
from mne.transforms import Transform, translation
from mne_bids import BIDSPath, write_raw_bids

import meeg_utils as meu
from meeg_utils import steps as S
from meeg_utils.dataset import _with_destination

from .simulation import make_erp

pytestmark = pytest.mark.filterwarnings(
    "ignore:Converting data files to BrainVision format",
    "ignore:There are channels without locations",
    "ignore:Not setting position",
    'ignore:Encountered data in "double" format',
)
EVENTS = ["stim/target", "stim/standard"]


@pytest.fixture(scope="module")
def eeg_bids(tmp_path_factory):
    """2 subjects x 2 runs of a simulated oddball task; sub-02 run-02 has no events."""
    root = tmp_path_factory.mktemp("bids")
    for subject in ("01", "02"):
        for run in ("01", "02"):
            raw = make_erp(seed=int(subject) * 10 + int(run))
            if (subject, run) == ("02", "02"):
                raw.set_annotations(None)
            path = BIDSPath(subject=subject, task="oddball", run=run, datatype="eeg", root=root)
            write_raw_bids(raw, path, allow_preload=True, format="BrainVision", verbose=False)
    return root


def _pipelines():
    pre = meu.Pipeline([("filter", S.Filter(1.0, 30.0))])
    epochs = meu.Pipeline([("epoch", S.Epoch(EVENTS, -0.1, 0.5))])
    return pre, epochs


class TestDataset:
    def test_selection(self, eeg_bids):
        ds = meu.Dataset(eeg_bids)
        assert len(ds) == 4
        assert set(ds.groups()) == {"sub-01_task-oddball_eeg", "sub-02_task-oddball_eeg"}
        assert len(meu.Dataset(eeg_bids, subjects=["01"])) == 2
        assert len(meu.Dataset(eeg_bids, runs=["02"], datatype="eeg")) == 2
        assert len(meu.Dataset(eeg_bids, datatype="meg")) == 0
        assert len(meu.Dataset(eeg_bids, exclude=["sub-02_task-oddball_run-02"])) == 3
        assert "4 recordings, 2 subjects" in repr(ds)
        assert {row["group"] for row in ds.table()} == set(ds.groups())

    def test_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            meu.Dataset(tmp_path / "nope")


def test_process_dataset(eeg_bids, tmp_path):
    pre, epochs = _pipelines()
    out = tmp_path / "derivatives"
    with pytest.warns(UserWarning, match="tasks failed"):
        result = meu.process_dataset(meu.Dataset(eeg_bids), out, preprocessing=pre, epochs=epochs)

    assert len(result.outputs("preprocessing")) == 4
    assert len(result.outputs("epochs")) == 3
    failed = {(r["stage"], r["source"].split("/")[-1].split("_eeg")[0]) for r in result.failed}
    assert failed == {
        ("epochs", "sub-02_task-oddball_run-02"),  # no events
        ("combine", "sub-02_task-oddball"),  # a run is missing
    }
    combined = result.outputs("combine")
    assert [p.split("/")[-1] for p in combined] == ["sub-01_task-oddball_desc-epochs_epo.fif"]
    run_epochs = [mne.read_epochs(f, verbose=False) for f in sorted(result.outputs("epochs"))[:2]]
    all_epochs = mne.read_epochs(combined[0], verbose=False)
    assert len(all_epochs) == sum(len(e) for e in run_epochs)
    sidecar = json.loads(Path(combined[0].replace(".fif", ".json")).read_text())
    assert len(sidecar["Sources"]) == 2 and sidecar["MeegUtils"]["combined_runs"] == ["01", "02"]

    log = result.to_csv(tmp_path / "log.csv")
    assert log.read_text().count("\n") == 1 + len(result.records)
    assert "epochs: 3 ok, 0 skipped, 1 failed" in repr(result)

    # resume: everything done is skipped, the failures are retried (and fail again)
    with pytest.warns(UserWarning, match="tasks failed"):
        again = meu.process_dataset(meu.Dataset(eeg_bids), out, preprocessing=pre, epochs=epochs)
    statuses = {(r["stage"], r["status"]) for r in again.records}
    assert ("preprocessing", "ok") not in statuses and ("epochs", "ok") not in statuses
    assert len(again.failed) == 2


def test_raise_on_error(eeg_bids, tmp_path):
    pre, epochs = _pipelines()
    with pytest.raises(RuntimeError, match="2 of 5 tasks failed"):
        meu.process_dataset(
            meu.Dataset(eeg_bids, subjects=["02"]), tmp_path, preprocessing=pre, epochs=epochs,
            on_error="raise",
        )  # fmt: skip


def test_preprocessing_only(eeg_bids, tmp_path):
    pre, _ = _pipelines()
    result = meu.process_dataset(
        meu.Dataset(eeg_bids, subjects=["01"]), tmp_path, preprocessing=pre
    )
    assert {r["stage"] for r in result.records} == {"preprocessing"}
    assert not result.failed


@pytest.mark.filterwarnings("ignore:The head moves")
def test_meg_runs_aligned_to_average(neuromag_fname, tmp_path):
    root = tmp_path / "bids"
    raw = mne.io.read_raw_fif(neuromag_fname, verbose=False).crop(0, 8).load_data(verbose=False)
    raw.pick(["meg", "stim"])
    for run, shift in (("01", 0.0), ("02", 0.004)):
        run_raw = raw.copy()
        with run_raw.info._unlock():
            trans = translation(shift, 0, 0) @ raw.info["dev_head_t"]["trans"]
            run_raw.info["dev_head_t"] = Transform("meg", "head", trans)
        path = BIDSPath(subject="01", task="rest", run=run, datatype="meg", root=root)
        write_raw_bids(run_raw, path, allow_preload=True, format="FIF", verbose=False)
    # Neuromag calibration files and derivatives inside the dataset are not recordings
    meg_dir = root / "sub-01" / "meg"
    shutil.copy(neuromag_fname, meg_dir / "sub-01_acq-crosstalk_meg.fif")
    (meg_dir / "sub-01_acq-calibration_meg.dat").write_text("")
    deriv_meg = root / "derivatives" / "other" / "sub-01" / "meg"
    deriv_meg.mkdir(parents=True)
    shutil.copy(neuromag_fname, deriv_meg / "sub-01_task-rest_run-01_meg.fif")
    dataset = meu.Dataset(root)
    assert [p.run for p in dataset] == ["01", "02"]
    assert len(meu.qc.dataset.find_recordings(root)) == 2

    pre = meu.Pipeline([("filter", S.Filter(1.0, 40.0))])
    epochs = meu.Pipeline([("epoch", S.FixedLengthEpochs(1.0))])
    result = meu.process_dataset(dataset, tmp_path / "deriv", preprocessing=pre, epochs=epochs)
    assert not result.failed
    run_files = sorted(result.outputs("epochs"))
    heads = [mne.read_epochs(f, verbose=False).info["dev_head_t"]["trans"] for f in run_files]
    assert np.allclose(heads[0], heads[1])  # both mapped to the average position
    expected = raw.info["dev_head_t"]["trans"][:3, 3] + [0.002, 0, 0]
    assert np.allclose(heads[0][:3, 3], expected, atol=1e-4)
    assert len(result.outputs("combine")) == 1


def test_with_destination():
    dest = np.eye(4)
    plain = meu.Pipeline([("filter", S.Filter(None, 40.0)), ("epoch", S.Epoch(["a"]))])
    aligned = _with_destination(plain, dest)
    assert [n for n, _ in aligned.steps] == ["filter", "align", "epoch"]
    assert [n for n, _ in plain.steps] == ["filter", "epoch"]  # unchanged
    existing = meu.Pipeline([("head", S.HeadAlign(np.eye(4) * 2)), ("epoch", S.Epoch(["a"]))])
    assert _with_destination(existing, dest)["head"].destination is dest
