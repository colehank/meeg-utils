"""Tests for batch preprocessing functionality.

This file tests the BatchPreprocessingPipeline class for processing
multiple files in parallel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mne_bids import BIDSPath

if TYPE_CHECKING:
    from mne.io import BaseRaw

from meeg_utils.preprocessing import BatchPreprocessingPipeline


@pytest.fixture
def multiple_bids_paths(
    tmp_path: Path,
    sample_eeg_raw: BaseRaw,
) -> list[BIDSPath]:
    """Create multiple BIDS datasets for batch testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary directory.
    sample_eeg_raw : BaseRaw
        Sample EEG data.

    Returns
    -------
    list[BIDSPath]
        List of BIDSPath objects for testing.
    """
    from mne_bids import write_raw_bids

    bids_root = tmp_path / "bids_batch"
    paths = []

    for subject_id in ["01", "02", "03"]:
        bids_path = BIDSPath(
            subject=subject_id,
            session="01",
            task="test",
            datatype="eeg",
            root=bids_root,
        )

        write_raw_bids(
            sample_eeg_raw,
            bids_path,
            allow_preload=True,
            format="BrainVision",
            overwrite=True,
            verbose=False,
        )

        paths.append(bids_path)

    return paths


class TestBatchInputParsing:
    """Test batch pipeline input parsing."""

    def test_accepts_list_of_bids_paths(
        self,
        multiple_bids_paths: list[BIDSPath],
    ) -> None:
        """Test that batch pipeline accepts list of BIDSPath objects."""
        batch = BatchPreprocessingPipeline(input_paths=multiple_bids_paths)

        assert batch.input_paths is not None
        assert len(batch.input_paths) == 3

    def test_accepts_list_of_strings(
        self,
        multiple_bids_paths: list[BIDSPath],
    ) -> None:
        """Test that batch pipeline accepts list of string paths."""
        string_paths = [str(p.fpath) for p in multiple_bids_paths]

        batch = BatchPreprocessingPipeline(input_paths=string_paths)

        assert batch.input_paths is not None
        assert len(batch.input_paths) == 3

    def test_accepts_mixed_path_types(
        self,
        multiple_bids_paths: list[BIDSPath],
    ) -> None:
        """Test that batch pipeline accepts mixed path types."""
        mixed_paths = [
            multiple_bids_paths[0],  # BIDSPath
            str(multiple_bids_paths[1].fpath),  # string
            Path(multiple_bids_paths[2].fpath),  # Path
        ]

        batch = BatchPreprocessingPipeline(input_paths=mixed_paths)  # type: ignore[arg-type]

        assert batch.input_paths is not None
        assert len(batch.input_paths) == 3

    def test_raises_error_for_empty_list(self) -> None:
        """Test that batch pipeline raises error for empty input list."""
        with pytest.raises(ValueError):
            BatchPreprocessingPipeline(input_paths=[])


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_run_batch_default_params(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test running batch processing with default parameters."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        # Use minimal ICA components for speed
        batch.run(ica_params={"n_components": 2})

        # Check that output files were created for all subjects
        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3

    def test_run_batch_parallel_processing(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test batch processing with parallel jobs."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
            n_jobs=2,
        )

        # Use minimal ICA components for speed
        batch.run(ica_params={"n_components": 2})

        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3

    def test_run_batch_with_custom_params(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test batch processing with custom parameters."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        filter_params = {
            "highpass": 0.5,
            "lowpass": 50.0,
            "sfreq": 200.0,
        }

        # Use minimal ICA components for speed
        batch.run(filter_params=filter_params, ica_params={"n_components": 2})

        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3

    def test_run_batch_skip_existing_files(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch processing can skip existing output files."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        # Run once with minimal ICA components
        batch.run(ica_params={"n_components": 2})

        # Run again with skip_existing=True
        batch.run(skip_existing=True, ica_params={"n_components": 2})

        # Should still have 3 files (not duplicated)
        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3

    def test_run_batch_error_handling(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch processing handles errors gracefully."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        # Add an invalid path to the list
        invalid_path = BIDSPath(
            subject="99",
            session="99",
            task="nonexistent",
            datatype="eeg",
            root=temp_output_dir / "nonexistent",
        )

        paths_with_invalid = multiple_bids_paths + [invalid_path]

        batch = BatchPreprocessingPipeline(
            input_paths=paths_with_invalid,
            output_dir=temp_output_dir,
        )

        # Should not crash, but continue processing valid files
        batch.run(ica_params={"n_components": 2})

        # Should have processed the 3 valid files
        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) >= 3


class TestBatchLogging:
    """Test batch processing logging functionality."""

    def test_batch_creates_log_files(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch processing creates log files."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        batch.run(save_logs=True, ica_params={"n_components": 2})

        # Check for log files
        log_files = list(temp_output_dir.glob("**/*.log"))
        assert len(log_files) > 0

    def test_batch_logging_level(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch processing respects logging level."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        batch.run(logging_level="WARNING", ica_params={"n_components": 2})

        # Should complete without errors
        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3


class TestBatchOutputOrganization:
    """Test batch output directory organization."""

    def test_batch_preserves_bids_structure(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch output preserves BIDS directory structure."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        batch.run(ica_params={"n_components": 2})

        # Check that BIDS structure is preserved
        for subject_id in ["01", "02", "03"]:
            subject_dir = temp_output_dir / f"sub-{subject_id}"
            assert subject_dir.exists()

    def test_batch_creates_derivatives_directory(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch creates derivatives directory."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        batch.run(ica_params={"n_components": 2})

        # Check for derivatives structure
        assert temp_output_dir.exists()
        assert any(temp_output_dir.iterdir())


class TestBatchProgressTracking:
    """Test batch processing progress tracking."""

    def test_batch_reports_progress(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch processing reports progress."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        # Should complete and process all files
        batch.run(ica_params={"n_components": 2})

        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3

    def test_batch_handles_partial_completion(
        self,
        multiple_bids_paths: list[BIDSPath],
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that batch can resume from partial completion."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        batch = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths,
            output_dir=temp_output_dir,
        )

        # Process first file only (simulate interruption)
        batch_partial = BatchPreprocessingPipeline(
            input_paths=multiple_bids_paths[:1],
            output_dir=temp_output_dir,
        )
        batch_partial.run(ica_params={"n_components": 2})

        # Now run full batch with skip_existing
        batch.run(skip_existing=True, ica_params={"n_components": 2})

        # Should have all 3 files
        output_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(output_files) == 3
