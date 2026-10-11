"""Dataset-level QC and HTML reports."""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import pytest
from mne_bids import BIDSPath, write_raw_bids

import meeg_utils as meu
from meeg_utils import qc
from meeg_utils import steps as S

from ..simulation import make_eeg

LINE_METRIC = "narrowband_noise.channel_types.eeg.line_db.50"
CHECKS = [qc.Amplitude(), qc.NarrowbandNoise(), qc.Blinks()]


@pytest.fixture
def dataset(tmp_path) -> list[Path]:
    """Six recordings; the last one has ten times more line noise."""
    paths = []
    for i in range(6):
        raw = make_eeg(seed=i, line_amplitude=200e-6 if i == 5 else 20e-6)
        paths.append(tmp_path / f"rec{i}_raw.fif")
        raw.save(paths[-1], verbose=False)
    return paths


class TestInspectDataset:
    def test_flags_outlying_recording(self, dataset, tmp_path):
        result = qc.inspect_dataset(dataset, CHECKS)
        assert len(result.reports) == 6
        line = [o for o in result.outliers if o["metric"] == LINE_METRIC]
        assert [o["source"] for o in line] == [str(dataset[5])]
        assert line[0]["z"] > qc.OUTLIER_Z

        rows = result.to_records()
        assert rows[5]["outlier_metrics"].count(LINE_METRIC) == 1
        fname = result.to_csv(tmp_path / "qc" / "summary.csv")
        with fname.open() as f:
            table = list(csv.DictReader(f))
        assert len(table) == 6
        assert LINE_METRIC in table[0]
        assert "6 recordings" in repr(result)

    def test_errors_do_not_stop_the_others(self, dataset, tmp_path):
        missing = tmp_path / "missing_raw.fif"
        with pytest.warns(UserWarning, match="QC failed for 1 of 3"):
            result = qc.inspect_dataset([dataset[0], missing, dataset[1]], CHECKS)
        assert len(result.reports) == 2
        assert "FileNotFoundError" in result.errors[str(missing)]
        assert result.to_records()[-1]["level"] == "error"

    def test_few_recordings_have_no_outliers(self, dataset):
        assert qc.inspect_dataset(dataset[3:], CHECKS).outliers == []

    def test_parallel(self, dataset):
        serial = qc.inspect_dataset(dataset[:3], CHECKS)
        parallel = qc.inspect_dataset(dataset[:3], CHECKS, n_jobs=2)
        assert serial.table == parallel.table

    def test_plot(self, dataset):
        import matplotlib.pyplot as plt

        figs = qc.inspect_dataset(dataset, CHECKS).plot()
        assert list(figs) == ["levels"]
        plt.close("all")

    @pytest.mark.filterwarnings("ignore:Converting data files to BrainVision format")
    def test_bids_root(self, small_raw, tmp_path):
        root = tmp_path / "bids"
        for run in ("1", "2"):
            raw = small_raw.copy()
            raw.info["line_freq"] = 50.0
            bids_path = BIDSPath(subject="01", task="rest", run=run, datatype="eeg", root=root)
            write_raw_bids(raw, bids_path, allow_preload=True, format="BrainVision", verbose=False)
        found = qc.find_recordings(root)
        assert [p.run for p in found] == ["1", "2"]
        result = qc.inspect_dataset(root, [qc.Amplitude()])
        assert len(result.reports) == 2


class TestReport:
    def test_qc_and_pipeline(self, eeg, tmp_path):
        pipe = meu.Pipeline([("filter", S.Filter(1.0, 40.0)), ("line_noise", S.LineNoise())])
        pipe.fit(eeg)
        report = meu.report.build(pipe, inst=eeg, qc=qc.inspect(eeg, CHECKS))
        assert isinstance(report, mne.Report)
        titles = list(report._content)
        assert any(c.name == "Findings" for c in titles)
        assert any(c.name == "line_noise: psd" for c in titles)
        assert any(c.name == "filter: response" for c in titles)
        assert any(c.name == "Provenance" for c in titles)
        fname = tmp_path / "report.html"
        report.save(fname, open_browser=False, verbose=False)
        text = fname.read_text(encoding="utf-8")
        assert "narrowband_noise" in text
        assert "suppression_db" in text

    def test_needs_content(self, eeg):
        with pytest.raises(ValueError, match="Nothing to report"):
            meu.report.build()
        with pytest.raises(ValueError, match="not fitted"):
            meu.report.build(meu.Pipeline([("filter", S.Filter(1.0, 40.0))]))
