"""Tests for io.read and io.save_derivative."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pytest
from mne.io import BaseRaw
from mne_bids import BIDSPath, mark_channels, write_raw_bids

import meeg_utils as meu
from meeg_utils.core import Pipeline

from ..test_core._steps import Demean, ToEpochs


@pytest.fixture
def bids_eeg(tmp_path: Path, small_raw: BaseRaw) -> BIDSPath:
    """A one-recording BIDS dataset (no session) with a bad channel marked in channels.tsv."""
    raw = small_raw.copy()
    raw.info["line_freq"] = 50.0
    raw.set_montage("standard_1020", on_missing="ignore")
    bids_path = BIDSPath(subject="01", task="rest", datatype="eeg", root=tmp_path / "bids")
    write_raw_bids(
        raw, bids_path, allow_preload=True, format="BrainVision", overwrite=True, verbose=False
    )
    mark_channels(bids_path, ch_names=["Pz"], status="bad", descriptions=["noisy"], verbose=False)
    return bids_path


pytestmark = pytest.mark.filterwarnings(
    "ignore:There are channels without locations:RuntimeWarning",
    "ignore:Not setting position of 1 eog channel:RuntimeWarning",
)


class TestRead:
    """io.read."""

    def test_reads_bids_path_with_metadata(self, bids_eeg: BIDSPath) -> None:
        """BIDS sidecars provide line frequency, bad channels and channel types."""
        raw = meu.io.read(bids_eeg)
        assert raw.preload
        assert raw.info["line_freq"] == 50.0
        assert raw.info["bads"] == ["Pz"]
        assert raw.get_channel_types(picks="EOG") == ["eog"]

    def test_plain_path_inside_bids_uses_bids_metadata(self, bids_eeg: BIDSPath) -> None:
        """A file path inside a BIDS dataset is read as BIDS."""
        raw = meu.io.read(str(bids_eeg.fpath))
        assert raw.info["bads"] == ["Pz"]
        assert raw.info["line_freq"] == 50.0

    def test_plain_file(self, small_raw: BaseRaw, tmp_path: Path) -> None:
        """Files outside BIDS are read with mne.io.read_raw."""
        fname = tmp_path / "rec_raw.fif"
        small_raw.save(fname)
        raw = meu.io.read(fname)
        assert raw.preload
        assert raw.first_samp == small_raw.first_samp

    def test_unknown_line_freq_is_reported(self, small_raw: BaseRaw, tmp_path: Path) -> None:
        """A missing line frequency is logged (and not guessed)."""
        fname = tmp_path / "rec_raw.fif"
        small_raw.save(fname)
        messages: list[str] = []
        meu.logger.enable("meeg_utils")
        handler = meu.logger.add(messages.append, level="WARNING", format="{message}")
        try:
            raw = meu.io.read(fname)
        finally:
            meu.logger.remove(handler)
            meu.logger.disable("meeg_utils")
        assert raw.info["line_freq"] is None
        assert any("power line frequency" in m for m in messages)

    @pytest.mark.parametrize("missing", ["nope.fif", BIDSPath(subject="99", root="/nonexistent")])
    def test_missing_file_raises(self, missing, tmp_path: Path) -> None:
        """Missing recordings raise FileNotFoundError."""
        if isinstance(missing, str):
            missing = tmp_path / missing
        with pytest.raises(FileNotFoundError):
            meu.io.read(missing)

    def test_pipeline_accepts_paths(self, bids_eeg: BIDSPath) -> None:
        """Pipelines read paths and BIDSPaths themselves."""
        out = Pipeline([("demean", Demean())]).fit_transform(bids_eeg)
        assert isinstance(out, BaseRaw)
        assert out.info["bads"] == ["Pz"]


class TestSaveDerivative:
    """io.save_derivative."""

    def test_bids_layout_and_sidecars(self, bids_eeg: BIDSPath, tmp_path: Path) -> None:
        """Output follows BIDS derivatives naming, without a ses- directory when there is no session."""
        root = tmp_path / "derivatives" / "meeg-utils"
        pipe = Pipeline([("demean", Demean())])
        clean = pipe.fit_transform(bids_eeg)

        fname = meu.io.save_derivative(clean, bids_eeg, root, pipeline=pipe)

        assert fname == root / "sub-01" / "eeg" / "sub-01_task-rest_desc-preproc_eeg.fif"
        description = json.loads((root / "dataset_description.json").read_text())
        assert description["DatasetType"] == "derivative"
        assert description["GeneratedBy"][0]["Name"] == "meeg-utils"

        sidecar = json.loads(fname.with_suffix(".json").read_text())
        assert sidecar["Sources"] == ["bids::sub-01/eeg/sub-01_task-rest_eeg.vhdr"]
        assert sidecar["MeegUtils"]["pipeline"]["steps"][0]["name"] == "demean"
        assert sidecar["MeegUtils"]["provenance"]["system"] == "eeg"
        assert sidecar["MeegUtils"]["qc"]["demean"]["max_abs_mean"] > 0

    def test_saved_data_round_trip(self, bids_eeg: BIDSPath, tmp_path: Path) -> None:
        """Saved data can be read back with timing and annotations intact."""
        raw = meu.io.read(bids_eeg)
        fname = meu.io.save_derivative(raw, bids_eeg, tmp_path / "deriv")
        back = meu.io.read(fname)  # inside a derivative dataset: read as a plain file
        assert back.first_samp == raw.first_samp
        assert back.ch_names == raw.ch_names
        assert list(back.annotations.onset) == pytest.approx(list(raw.annotations.onset))

    def test_epochs_suffix(self, bids_eeg: BIDSPath, tmp_path: Path) -> None:
        """Epochs are saved with the epo suffix."""
        epochs = ToEpochs(2.0).fit_transform(meu.io.read(bids_eeg))
        fname = meu.io.save_derivative(epochs, bids_eeg, tmp_path / "deriv", desc="clean")
        assert fname.name == "sub-01_task-rest_desc-clean_epo.fif"
        assert len(mne.read_epochs(fname, verbose=False)) == len(epochs)

    def test_refuses_to_overwrite(self, bids_eeg: BIDSPath, tmp_path: Path) -> None:
        """Existing outputs are only replaced with overwrite=True."""
        raw = meu.io.read(bids_eeg)
        meu.io.save_derivative(raw, bids_eeg, tmp_path / "deriv")
        with pytest.raises(FileExistsError, match="overwrite=True"):
            meu.io.save_derivative(raw, bids_eeg, tmp_path / "deriv")
        meu.io.save_derivative(raw, bids_eeg, tmp_path / "deriv", overwrite=True)

    def test_non_bids_source(self, small_raw: BaseRaw, tmp_path: Path) -> None:
        """Non-BIDS sources are named after the source file."""
        fname = meu.io.save_derivative(small_raw, "/data/rec01.fif", tmp_path / "deriv")
        assert fname == tmp_path / "deriv" / "rec01_desc-preproc_eeg.fif"
        assert json.loads(fname.with_suffix(".json").read_text())["Sources"] == ["/data/rec01.fif"]
