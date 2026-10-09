"""The meu command line."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mne
import pytest
from mne_bids import BIDSPath, write_raw_bids

import meeg_utils as meu
from meeg_utils import steps as S
from meeg_utils.cli import main

from .simulation import make_eeg


@pytest.fixture(autouse=True)
def _no_logging_leak():
    yield
    meu.teardown_logging()


@pytest.fixture
def recording(tmp_path) -> Path:
    fname = tmp_path / "raw" / "rec_raw.fif"
    fname.parent.mkdir()
    make_eeg(bad=True).save(fname, verbose=False)
    return fname


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert meu.__version__ in capsys.readouterr().out


def test_module_entry_point():
    out = subprocess.run(
        [sys.executable, "-m", "meeg_utils.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0 and "meeg-utils" in out.stdout


class TestQC:
    def test_one_recording(self, recording, tmp_path, capsys):
        out = tmp_path / "qc"
        assert main(["qc", str(recording), "--out", str(out), "-q"]) == 0
        report = json.loads((out / "rec_raw_qc.json").read_text())
        assert report["level"] == "warn"
        assert report["checks"]["amplitude"]["metrics"]["flat_channels"] == ["P8"]
        assert (out / "rec_raw_qc.html").exists()
        assert (out / "qc_summary.csv").exists()
        assert "amplitude" in capsys.readouterr().out

    def test_fail_on(self, recording, tmp_path):
        args = ["qc", str(recording), "--out", str(tmp_path / "qc"), "--no-report", "-q"]
        assert main([*args, "--fail-on", "warn"]) == 2
        assert main([*args, "--fail-on", "fail"]) == 0

    def test_selected_checks(self, recording, tmp_path):
        out = tmp_path / "qc"
        main(["qc", str(recording), "--out", str(out), "--checks", "amplitude,blinks",
              "--no-report", "-q"])  # fmt: skip
        report = json.loads((out / "rec_raw_qc.json").read_text())
        assert set(report["checks"]) == {"amplitude", "blinks"}
        with pytest.raises(SystemExit, match="Unknown checks"):
            main(["qc", str(recording), "--out", str(out), "--checks", "nonsense"])

    @pytest.mark.filterwarnings(
        "ignore:Converting data files to BrainVision format",
        "ignore:There are channels without locations:RuntimeWarning",
        "ignore:Not setting position of 1 eog channel:RuntimeWarning",
    )
    def test_bids_dataset(self, tmp_path):
        root = tmp_path / "bids"
        for run in ("1", "2"):
            raw = make_eeg(seed=int(run))
            bids_path = BIDSPath(subject="01", task="rest", run=run, datatype="eeg", root=root)
            write_raw_bids(raw, bids_path, allow_preload=True, format="BrainVision", verbose=False)
        out = tmp_path / "qc"
        assert main(["qc", str(root), "--out", str(out), "--no-report", "-q",
                     "--checks", "amplitude"]) == 0  # fmt: skip
        assert (out / "qc_overview.png").exists()
        assert len(list(out.glob("*_qc.json"))) == 2

    def test_missing_recording(self, tmp_path):
        with pytest.warns(UserWarning, match="QC failed"):
            code = main(["qc", str(tmp_path / "missing_raw.fif"), "--out", str(tmp_path), "-q"])
        assert code == 1


class TestRun:
    def test_config(self, recording, tmp_path, capsys):
        config = tmp_path / "pipeline.yaml"
        meu.Pipeline([("filter", S.Filter(1.0, 40.0))]).to_yaml(config)
        out = tmp_path / "deriv"
        args = ["run", str(config), "--sources", str(recording), "--out", str(out), "-q"]
        assert main(args) == 0
        output = out / "rec_raw_desc-preproc_eeg.fif"
        assert mne.io.read_raw_fif(output, verbose=False).info["lowpass"] == 40.0
        assert "1/1 recordings done" in capsys.readouterr().out
        assert main([*args, "--skip-existing"]) == 0
        with pytest.warns(UserWarning, match="1 of 1 recordings failed"):
            assert main(args) == 1  # the output exists

    def test_preset(self, recording, tmp_path):
        out = tmp_path / "deriv"
        args = ["run", "--preset", "had-meeg", "--datatype", "eeg",
                "--sources", str(recording), "--out", str(out), "-q"]  # fmt: skip
        assert main(args) == 0
        sidecar = json.loads((out / "rec_raw_desc-preproc_eeg.json").read_text())
        assert [s["name"] for s in sidecar["MeegUtils"]["pipeline"]["steps"]][-1] == "ica"

    def test_config_or_preset(self, recording, tmp_path):
        with pytest.raises(SystemExit, match="either a configuration file or --preset"):
            main(["run", "--sources", str(recording), "--out", str(tmp_path)])
