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


class TestPresetOptions:
    def test_epochs_stage(self, tmp_path):
        from .simulation import make_erp

        fname = tmp_path / "raw" / "erp_raw.fif"
        fname.parent.mkdir()
        make_erp().save(fname, verbose=False)
        out = tmp_path / "deriv"
        args = ["run", "--preset", "eeg-rest", "--option", "stage=epochs",
                "--option", "epoch_duration=2", "--sources", str(fname), "--out", str(out),
                "--desc", "epochs", "-q"]  # fmt: skip
        assert main(args) == 0
        saved = list(out.glob("*epo.fif"))
        assert len(saved) == 1
        assert mne.read_epochs(saved[0], verbose=False).times[-1] == pytest.approx(2.0, abs=0.01)

    def test_invalid_option(self, recording, tmp_path):
        with pytest.raises(SystemExit, match="invalid --option"):
            main(["run", "--preset", "eeg-erp", "--option", "colour=red",
                  "--sources", str(recording), "--out", str(tmp_path)])  # fmt: skip

    def test_option_needs_preset(self, recording, tmp_path):
        config = tmp_path / "pipeline.yaml"
        meu.Pipeline([("filter", S.Filter(1.0, 40.0))]).to_yaml(config)
        with pytest.raises(SystemExit, match="only applies to --preset"):
            main(["run", str(config), "--option", "stage=epochs",
                  "--sources", str(recording), "--out", str(tmp_path)])  # fmt: skip

    def test_meg_system_detected(self, neuromag_fname):
        import argparse

        from meeg_utils.cli import _preset_options

        args = argparse.Namespace(preset="meg-erp", option=[], datatype=None)
        assert _preset_options(args, [neuromag_fname])["system"] == "neuromag"
        args.option = ["system=ctf"]
        assert _preset_options(args, [neuromag_fname])["system"] == "ctf"


@pytest.mark.filterwarnings(
    "ignore:Converting data files to BrainVision format",
    "ignore:There are channels without locations:RuntimeWarning",
    "ignore:Not setting position of 1 eog channel:RuntimeWarning",
)
def test_run_dataset_with_epochs(tmp_path):
    from .simulation import make_erp

    root = tmp_path / "bids"
    for run in ("1", "2"):
        path = BIDSPath(subject="01", task="oddball", run=run, datatype="eeg", root=root)
        write_raw_bids(
            make_erp(seed=int(run)), path, allow_preload=True, format="BrainVision", verbose=False
        )
    pre = tmp_path / "pre.yaml"
    meu.Pipeline([("filter", S.Filter(1.0, 30.0))]).to_yaml(pre)
    ep = tmp_path / "epochs.yaml"
    meu.Pipeline([("epoch", S.Epoch(["stim/target", "stim/standard"], -0.1, 0.5))]).to_yaml(ep)
    out = tmp_path / "deriv"
    args = ["run", str(pre), "--sources", str(root), "--out", str(out), "--epochs", str(ep), "-q"]
    assert main(args) == 0
    assert (out / "sub-01" / "eeg" / "sub-01_task-oddball_desc-epochs_epo.fif").exists()
    assert (out / "meu_batch.csv").exists()
    assert main(args) == 0  # resumes: everything is skipped
    with pytest.raises(SystemExit, match="root of one BIDS dataset"):
        main(["run", str(pre), "--sources", str(root / "sub-01"), str(root), "--out", str(out),
              "--epochs", str(ep)])  # fmt: skip


class TestPresetCommand:
    def test_list(self, capsys):
        assert main(["preset"]) == 0
        out = capsys.readouterr().out
        assert "eeg-erp" in out and "had-meeg" in out and "options:" in out

    def test_write_and_run(self, tmp_path, recording):
        config = tmp_path / "pipe.yaml"
        assert main(["preset", "eeg-rest", "--option", "stage=epochs", "--option",
                     "epoch_duration=2", "-o", str(config)]) == 0  # fmt: skip
        pipe = meu.Pipeline.from_yaml(config)
        assert [n for n, _ in pipe.steps] == ["epoch", "autoreject"]

    def test_print(self, capsys):
        assert main(["preset", "meg-erp", "--option", "system=kit"]) == 0
        assert "Regression" in capsys.readouterr().out

    def test_meg_needs_system(self):
        with pytest.raises(SystemExit, match="system="):
            main(["preset", "meg-erp"])
        with pytest.raises(SystemExit, match="Unknown preset"):
            main(["preset", "nope"])


class TestReportCommand:
    @pytest.fixture(scope="class")
    def derivatives(self, tmp_path_factory):
        """Six recordings, one with many artifacts, processed with BadSegments."""
        tmp = tmp_path_factory.mktemp("report")
        sources = []
        for k in range(6):
            raw = make_eeg(seed=k, blinks=False)
            if k == 3:
                for start in range(5, 55, 5):
                    raw._data[:5, start * 250 : start * 250 + 100] += 5e-3
            fname = tmp / "raw" / f"rec{k}_raw.fif"
            fname.parent.mkdir(exist_ok=True)
            raw.save(fname, verbose=False)
            sources.append(fname)
        pipe = meu.Pipeline([("segments", S.BadSegments("amplitude", reject={"eeg": 1e-3}))])
        meu.process(pipe, sources, tmp / "deriv")
        return tmp / "deriv"

    def test_folder_summary(self, derivatives, capsys):
        assert main(["report", str(derivatives), "--desc", "preproc", "-q"]) == 0
        assert (
            "rec3_raw_desc-preproc_eeg.fif: segments.amplitude.fraction" in capsys.readouterr().out
        )
        assert (derivatives / "derivatives_summary.csv").exists()
        assert (derivatives / "derivatives_summary.html").exists()

    def test_one_derivative(self, derivatives, tmp_path):
        fname = derivatives / "rec0_raw_desc-preproc_eeg.fif"
        out = tmp_path / "rec0.html"
        assert main(["report", str(fname), "--out", str(out), "-q"]) == 0
        text = out.read_text()
        assert "segments: metrics" in text and "BadSegments" in text

    def test_errors(self, tmp_path):
        with pytest.raises(SystemExit, match="does not exist"):
            main(["report", str(tmp_path / "nope.fif")])
        with pytest.raises(ValueError, match="No meeg-utils derivatives"):
            main(["report", str(tmp_path)])


def test_set_step_parameters(recording, tmp_path):
    config = tmp_path / "pipeline.yaml"
    meu.Pipeline([("filter", S.Filter(1.0, 40.0))]).to_yaml(config)
    out = tmp_path / "deriv"
    args = ["run", str(config), "--sources", str(recording), "--out", str(out), "-q",
            "--set", "filter__h_freq=30"]  # fmt: skip
    assert main(args) == 0
    raw = mne.io.read_raw_fif(out / "rec_raw_desc-preproc_eeg.fif", verbose=False)
    assert raw.info["lowpass"] == 30.0
    with pytest.raises(SystemExit, match="no parameter"):
        main(
            [
                "run",
                str(config),
                "--sources",
                str(recording),
                "--out",
                str(out),
                "--set",
                "ica__x=1",
            ]
        )
    with pytest.raises(SystemExit, match="STEP__PARAM"):
        main(["run", str(config), "--sources", str(recording), "--out", str(out), "--set", "oops"])
