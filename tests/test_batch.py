"""meu.process: one pipeline, many recordings."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pytest
from mne.io import BaseRaw
from mne_bids import BIDSPath, write_raw_bids

import meeg_utils as meu
from meeg_utils import Pipeline
from meeg_utils import steps as S


@pytest.fixture
def pipe() -> Pipeline:
    return Pipeline([("filter", S.Filter(1.0, 30.0)), ("reference", S.Reference())])


@pytest.fixture
def fif_files(tmp_path: Path, small_raw: BaseRaw) -> list[Path]:
    paths = []
    for i in range(3):
        paths.append(tmp_path / "raw" / f"rec{i}_raw.fif")
        paths[-1].parent.mkdir(exist_ok=True)
        small_raw.save(paths[-1], verbose=False)
    return paths


def test_processes_each_recording(pipe, fif_files, tmp_path):
    records = meu.process(pipe, fif_files, tmp_path / "deriv")
    assert [r["status"] for r in records] == ["ok"] * 3
    for record, source in zip(records, fif_files, strict=True):
        out = Path(record["output"])
        assert out.name == f"{source.stem}_desc-preproc_eeg.fif"
        assert mne.io.read_raw_fif(out, verbose=False).info["highpass"] == 1.0
        sidecar = json.loads(out.with_suffix(".json").read_text())
        assert sidecar["MeegUtils"]["qc"]["filter"]["lowpass"] == 30.0
    assert not hasattr(pipe, "system_")  # the given pipeline stays unfitted


def test_failures_do_not_stop_the_others(pipe, fif_files, tmp_path):
    sources = [fif_files[0], tmp_path / "missing_raw.fif", fif_files[1]]
    with pytest.raises(RuntimeError, match=r"1 of 3 recordings failed:\n.*missing_raw"):
        meu.process(pipe, sources, tmp_path / "deriv")
    assert len(list((tmp_path / "deriv").glob("*.fif"))) == 2

    with pytest.warns(UserWarning, match="1 of 3 recordings failed"):
        records = meu.process(pipe, sources, tmp_path / "deriv2", on_error="warn")
    assert [r["status"] for r in records] == ["ok", "failed", "ok"]
    assert "FileNotFoundError" in records[1]["error"]


def test_existing_outputs(pipe, fif_files, tmp_path):
    root = tmp_path / "deriv"
    meu.process(pipe, fif_files[:1], root)
    with pytest.raises(RuntimeError, match="exists"):
        meu.process(pipe, fif_files[:1], root)

    records = meu.process(pipe, fif_files, root, skip_existing=True)
    assert [r["status"] for r in records] == ["skipped", "ok", "ok"]
    records = meu.process(pipe, fif_files[:1], root, overwrite=True)
    assert records[0]["status"] == "ok"


def test_parallel_matches_serial(pipe, fif_files, tmp_path):
    serial = meu.process(pipe, fif_files, tmp_path / "a")
    parallel = meu.process(pipe, fif_files, tmp_path / "b", n_jobs=2)
    for r1, r2 in zip(serial, parallel, strict=True):
        d1 = mne.io.read_raw_fif(r1["output"], verbose=False).get_data()
        d2 = mne.io.read_raw_fif(r2["output"], verbose=False).get_data()
        assert (d1 == d2).all()


@pytest.mark.filterwarnings("ignore:Converting data files to BrainVision format")
def test_bids_paths_use_bids_names(pipe, small_raw, tmp_path):
    root = tmp_path / "bids"
    sources = []
    for run in ("1", "2"):
        bids_path = BIDSPath(subject="01", task="rest", run=run, datatype="eeg", root=root)
        raw = small_raw.copy()
        raw.info["line_freq"] = 50.0
        write_raw_bids(raw, bids_path, allow_preload=True, format="BrainVision", verbose=False)
        sources.append(bids_path)
    deriv = root / "derivatives" / "meeg-utils"

    # a BIDSPath and a plain path inside the dataset name the output the same way
    records = meu.process(pipe, [sources[0], str(sources[1].fpath)], deriv)
    names = [Path(r["output"]).relative_to(deriv).as_posix() for r in records]
    assert names == [
        "sub-01/eeg/sub-01_task-rest_run-1_desc-preproc_eeg.fif",
        "sub-01/eeg/sub-01_task-rest_run-2_desc-preproc_eeg.fif",
    ]
    assert meu.io.existing_derivatives(sources[1], deriv) == [deriv / names[1]]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"on_error": "ignore"}, "on_error"),
        ({"skip_existing": True, "overwrite": True}, "mutually exclusive"),
    ],
)
def test_invalid_arguments(pipe, fif_files, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        meu.process(pipe, fif_files, tmp_path, **kwargs)


def test_no_sources(pipe, tmp_path):
    with pytest.raises(ValueError, match="No recordings"):
        meu.process(pipe, [], tmp_path)
