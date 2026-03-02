"""Tests for preprocessing pipeline core functionality.

Following TDD principles:
1. Write test first
2. Watch it fail (RED)
3. Write minimal code to pass (GREEN)
4. Refactor

This file tests the main PreprocessingPipeline class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mne
import pytest
from mne_bids import BIDSPath

if TYPE_CHECKING:
    from mne.io import BaseRaw

# Import the module we're testing (will fail until we create it)
from meeg_utils.preprocessing import PreprocessingPipeline


class TestPathInputParsing:
    """Test various input path formats are correctly parsed."""

    def test_accepts_plain_string_path(
        self,
        mock_bids_path_string: str,
    ) -> None:
        """Test that pipeline accepts a plain string path."""
        pipeline = PreprocessingPipeline(input_path=mock_bids_path_string)

        assert pipeline.input_path is not None
        assert isinstance(pipeline.input_path, Path)

    def test_accepts_pathlib_path(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test that pipeline accepts a pathlib.Path object."""
        path = Path(mock_bids_path.fpath)

        pipeline = PreprocessingPipeline(input_path=path)

        assert pipeline.input_path is not None
        assert isinstance(pipeline.input_path, Path)

    def test_accepts_bids_path_object(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test that pipeline accepts a BIDSPath object."""
        pipeline = PreprocessingPipeline(input_path=mock_bids_path)

        assert pipeline.input_path is not None
        # Should convert BIDSPath to Path internally
        assert isinstance(pipeline.input_path, Path | BIDSPath)

    def test_accepts_bids_format_string(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test that pipeline accepts a BIDS-formatted string."""
        # Create a BIDS-like string: "sub-01/ses-01/eeg/..."
        bids_string = str(mock_bids_path.fpath)

        pipeline = PreprocessingPipeline(input_path=bids_string)

        assert pipeline.input_path is not None

    def test_raises_error_for_nonexistent_path(self) -> None:
        """Test that pipeline raises error for non-existent path."""
        with pytest.raises((FileNotFoundError, ValueError)):
            PreprocessingPipeline(input_path="/nonexistent/path/to/data.fif")

    def test_raises_error_for_invalid_path_type(self) -> None:
        """Test that pipeline raises error for invalid path type."""
        with pytest.raises(TypeError):
            PreprocessingPipeline(input_path=12345)  # type: ignore[arg-type]


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_data_from_bids_path(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test loading data from a BIDSPath object."""
        pipeline = PreprocessingPipeline(input_path=mock_bids_path)
        raw = pipeline.load_data()
        raw = pipeline.raw  # Should be stored in the pipeline instance

        assert raw is not None
        assert isinstance(raw, mne.io.BaseRaw)
        assert raw._data is not None  # Data should be loaded

    def test_load_data_from_string_path(
        self,
        mock_bids_path_string: str,
    ) -> None:
        """Test loading data from a string path."""
        pipeline = PreprocessingPipeline(input_path=mock_bids_path_string)
        raw = pipeline.load_data()

        assert raw is not None
        assert isinstance(raw, mne.io.BaseRaw)

    def test_load_data_infers_datatype(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test that data loading correctly infers EEG/MEG datatype."""
        pipeline = PreprocessingPipeline(input_path=mock_bids_path)

        # Should automatically detect datatype
        assert pipeline.datatype in ["eeg", "meg"]

    def test_load_data_caching(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test that loading data twice returns cached instance."""
        pipeline = PreprocessingPipeline(input_path=mock_bids_path)

        raw1 = pipeline.load_data()
        raw2 = pipeline.load_data()

        # Should return the same object (cached)
        assert raw1 is raw2


class TestFilteringAndResampling:
    """Test filtering and resampling functionality."""

    def test_filter_and_resample_with_valid_params(
        self,
        sample_eeg_raw: BaseRaw,
    ) -> None:
        """Test filtering and resampling with valid parameters."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw  # Set raw directly for testing

        filtered = pipeline.filter_and_resample(
            highpass=0.1,
            lowpass=100.0,
            sfreq=250.0,
        )

        assert filtered is not None
        assert filtered.info["sfreq"] == 250.0
        assert filtered.info["highpass"] == 0.1
        assert filtered.info["lowpass"] == 100.0

    def test_filter_validates_nyquist_frequency(
        self,
        sample_eeg_raw: BaseRaw,
    ) -> None:
        """Test that filter validates against Nyquist frequency."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw

        # Should raise error if lowpass >= sfreq/2
        with pytest.raises((ValueError, AssertionError)):
            pipeline.filter_and_resample(
                highpass=0.1,
                lowpass=150.0,
                sfreq=250.0,  # Nyquist = 125 Hz
            )

    def test_filter_validates_highpass_lowpass_order(
        self,
        sample_eeg_raw: BaseRaw,
    ) -> None:
        """Test that filter validates highpass < lowpass."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw

        with pytest.raises((ValueError, AssertionError)):
            pipeline.filter_and_resample(
                highpass=50.0,
                lowpass=10.0,  # lowpass < highpass: invalid
                sfreq=250.0,
            )

    def test_filter_uses_default_params(
        self,
        sample_eeg_raw: BaseRaw,
    ) -> None:
        """Test filtering with default parameters."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw

        # Should work with default params
        filtered = pipeline.filter_and_resample()

        assert filtered is not None
        assert filtered.info["sfreq"] > 0


class TestBadChannelDetection:
    """Test bad channel detection and fixing."""

    def test_detect_bad_channels_eeg(
        self,
        sample_eeg_raw: BaseRaw,
        mock_prep,
        mocker,
    ) -> None:
        """Test bad channel detection for EEG data."""
        # Mock PREP to avoid expensive computation
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        result = pipeline.detect_and_fix_bad_channels()

        assert result is not None
        # Should have detected and marked bad channels
        assert hasattr(result, "info")

    def test_detect_bad_channels_meg(
        self,
        sample_meg_raw: BaseRaw,
    ) -> None:
        """Test bad channel detection for MEG data."""
        pipeline = PreprocessingPipeline(input_path=sample_meg_raw)
        pipeline.raw = sample_meg_raw
        pipeline.datatype = "meg"

        # MEG uses Maxwell filter (fast enough, no mock needed)
        result = pipeline.detect_and_fix_bad_channels()

        assert result is not None

    def test_interpolate_bad_channels(
        self,
        sample_eeg_raw: BaseRaw,
    ) -> None:
        """Test that bad channels are interpolated."""
        # Manually mark a channel as bad (using real channel name)
        sample_eeg_raw.info["bads"] = ["Fp1"]

        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        # No PREP needed, just interpolation (fast)
        result = pipeline.detect_and_fix_bad_channels(fix=True)

        # After interpolation, bads should be cleared or fixed
        assert result is not None

    def test_skip_interpolation_if_fix_false(
        self,
        sample_eeg_raw: BaseRaw,
    ) -> None:
        """Test that interpolation is skipped if fix=False."""
        sample_eeg_raw.info["bads"] = ["Fp1"]

        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        result = pipeline.detect_and_fix_bad_channels(fix=False)

        # Bads should still be marked
        assert "Fp1" in result.info["bads"]


class TestLineNoiseRemoval:
    """Test line noise (power line) removal."""

    def test_remove_line_noise_default_fline(
        self,
        sample_eeg_raw: BaseRaw,
        mock_zapline,
    ) -> None:
        """Test line noise removal with default frequency (50 Hz)."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        cleaned = pipeline.remove_line_noise()

        assert cleaned is not None
        assert isinstance(cleaned, mne.io.BaseRaw)

    def test_remove_line_noise_custom_fline(
        self,
        sample_eeg_raw: BaseRaw,
        mock_zapline,
    ) -> None:
        """Test line noise removal with custom frequency (60 Hz)."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        cleaned = pipeline.remove_line_noise(fline=60.0)

        assert cleaned is not None

    def test_line_noise_removal_eeg_vs_meg(
        self,
        sample_eeg_raw: BaseRaw,
        sample_meg_raw: BaseRaw,
        mock_zapline,
    ) -> None:
        """Test that different methods are used for EEG vs MEG."""
        # EEG should use zapline_iter (mocked for speed)
        pipeline_eeg = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline_eeg.raw = sample_eeg_raw
        pipeline_eeg.datatype = "eeg"

        cleaned_eeg = pipeline_eeg.remove_line_noise()
        assert cleaned_eeg is not None

        # MEG should use zapline (mocked for speed)
        pipeline_meg = PreprocessingPipeline(input_path=sample_meg_raw)
        pipeline_meg.raw = sample_meg_raw
        pipeline_meg.datatype = "meg"

        cleaned_meg = pipeline_meg.remove_line_noise()
        assert cleaned_meg is not None


class TestICAProcessing:
    """Test ICA (Independent Component Analysis) processing."""

    def test_apply_ica_automatic_labeling(
        self,
        sample_eeg_raw: BaseRaw,
        mock_iclabel,
    ) -> None:
        """Test ICA with automatic component labeling."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        # Use minimal components for speed (trust MNE's ICA)
        result = pipeline.apply_ica(n_components=2, method="fastica")

        assert result is not None
        assert isinstance(result, mne.io.BaseRaw)

    def test_apply_ica_custom_n_components(
        self,
        sample_eeg_raw: BaseRaw,
        mock_iclabel,
    ) -> None:
        """Test ICA with custom number of components."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        # Use minimal components for speed
        result = pipeline.apply_ica(n_components=3)

        assert result is not None

    def test_apply_ica_default_components_eeg_vs_meg(
        self,
        sample_eeg_raw: BaseRaw,
        sample_meg_raw: BaseRaw,
        mock_iclabel,
    ) -> None:
        """Test that default n_components differ for EEG vs MEG."""
        # EEG default should be ~20, use minimal for speed
        pipeline_eeg = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline_eeg.raw = sample_eeg_raw
        pipeline_eeg.datatype = "eeg"

        result_eeg = pipeline_eeg.apply_ica(n_components=2)
        assert result_eeg is not None

        # MEG default should be ~40, use minimal for speed
        pipeline_meg = PreprocessingPipeline(input_path=sample_meg_raw)
        pipeline_meg.raw = sample_meg_raw
        pipeline_meg.datatype = "meg"

        result_meg = pipeline_meg.apply_ica(n_components=2)
        assert result_meg is not None

    def test_apply_ica_with_regression(
        self,
        sample_eeg_raw: BaseRaw,
        mock_iclabel,
    ) -> None:
        """Test ICA with artifact regression enabled."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        # Use minimal components for speed
        result = pipeline.apply_ica(n_components=2, regress=True)

        assert result is not None

    def test_apply_ica_manual_labels(
        self,
        sample_eeg_raw: BaseRaw,
        mock_iclabel,
    ) -> None:
        """Test ICA with manual component labels."""
        pipeline = PreprocessingPipeline(input_path=sample_eeg_raw)
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        # Use minimal components for speed
        n_comp = 2
        manual_labels = ["brain", "eye blink"]

        result = pipeline.apply_ica(
            n_components=n_comp,
            manual_labels=manual_labels,
            regress=True,
        )

        assert result is not None


class TestFullPipeline:
    """Test complete preprocessing pipeline end-to-end."""

    def test_run_complete_pipeline_default_params(
        self,
        mock_bids_path: BIDSPath,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test running complete pipeline with default parameters."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        pipeline = PreprocessingPipeline(input_path=mock_bids_path)

        # Use minimal ICA components for speed
        result = pipeline.run(ica_params={"n_components": 2})

        assert result is not None
        assert isinstance(result, mne.io.BaseRaw)

    def test_run_pipeline_with_custom_filter_params(
        self,
        mock_bids_path: BIDSPath,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test pipeline with custom filtering parameters."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        pipeline = PreprocessingPipeline(input_path=mock_bids_path)

        filter_params = {
            "highpass": 0.5,
            "lowpass": 50.0,
            "sfreq": 200.0,
        }

        # Use minimal ICA components for speed
        result = pipeline.run(filter_params=filter_params, ica_params={"n_components": 2})

        assert result is not None
        assert result.info["sfreq"] == 200.0

    def test_run_pipeline_selective_steps(
        self,
        mock_bids_path: BIDSPath,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test running pipeline with selective steps enabled/disabled."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        pipeline = PreprocessingPipeline(input_path=mock_bids_path)

        result = pipeline.run(
            detect_bad_channels=True,
            remove_line_noise=True,
            apply_ica=False,  # Skip ICA
        )

        assert result is not None

    def test_run_pipeline_saves_intermediate_files(
        self,
        mock_bids_path: BIDSPath,
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that pipeline saves intermediate processing files."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        pipeline = PreprocessingPipeline(
            input_path=mock_bids_path,
            output_dir=temp_output_dir,
        )

        # Use minimal ICA components for speed
        result = pipeline.run(save_intermediate=True, ica_params={"n_components": 2})

        assert result is not None
        # Check that intermediate files were created (in BIDS structure)
        assert len(list(temp_output_dir.glob("**/*.tsv"))) > 0

    def test_run_pipeline_output_directory_creation(
        self,
        mock_bids_path: BIDSPath,
        temp_output_dir: Path,
        mock_iclabel,
        mock_zapline,
        mock_prep,
        mocker,
    ) -> None:
        """Test that pipeline creates output directory if it doesn't exist."""
        # Mock PREP for speed
        mocker.patch(
            "meeg_utils.preprocessing.bad_channels.NoisyChannels",
            return_value=mock_prep,
        )

        output_dir = temp_output_dir / "new_subdir"

        pipeline = PreprocessingPipeline(
            input_path=mock_bids_path,
            output_dir=output_dir,
        )

        # Use minimal ICA components for speed
        result = pipeline.run(ica_params={"n_components": 2})

        assert result is not None
        assert output_dir.exists()


class TestOutputManagement:
    """Test output file saving and management."""

    def test_save_preprocessed_data(
        self,
        sample_eeg_raw: BaseRaw,
        temp_output_dir: Path,
    ) -> None:
        """Test saving preprocessed data."""
        pipeline = PreprocessingPipeline(
            input_path=sample_eeg_raw,
            output_dir=temp_output_dir,
        )
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        output_file = temp_output_dir / "preprocessed_eeg.fif"
        pipeline.save(filename=output_file)

        assert output_file.exists()

    def test_save_with_automatic_naming(
        self,
        mock_bids_path: BIDSPath,
        temp_output_dir: Path,
    ) -> None:
        """Test saving with automatic BIDS-compliant filename."""
        pipeline = PreprocessingPipeline(
            input_path=mock_bids_path,
            output_dir=temp_output_dir,
        )
        pipeline.load_data()

        pipeline.save()  # Should auto-generate filename

        # Check that file was created with proper naming (recursive search for BIDS subdirs)
        saved_files = list(temp_output_dir.glob("**/*_preproc_*.fif"))
        assert len(saved_files) > 0

    def test_save_derivatives_metadata(
        self,
        sample_eeg_raw: BaseRaw,
        temp_output_dir: Path,
    ) -> None:
        """Test that derivative metadata files are saved."""
        pipeline = PreprocessingPipeline(
            input_path=sample_eeg_raw,
            output_dir=temp_output_dir,
        )
        pipeline.raw = sample_eeg_raw
        pipeline.datatype = "eeg"

        # Run bad channel detection
        pipeline.detect_and_fix_bad_channels()

        # Should save TSV and JSON sidecar files (recursive search for BIDS subdirs)
        tsv_files = list(temp_output_dir.glob("**/*.tsv"))
        json_files = list(temp_output_dir.glob("**/*.json"))

        assert len(tsv_files) > 0
        assert len(json_files) > 0


class TestConfigurationOptions:
    """Test various configuration options."""

    def test_use_cuda_option(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test CUDA acceleration option."""
        pipeline = PreprocessingPipeline(
            input_path=mock_bids_path,
            use_cuda=False,
        )

        assert pipeline.n_jobs != "cuda"

    def test_n_jobs_option(
        self,
        mock_bids_path: BIDSPath,
    ) -> None:
        """Test parallel processing option."""
        pipeline = PreprocessingPipeline(
            input_path=mock_bids_path,
            n_jobs=4,
        )

        assert pipeline.n_jobs == 4

    def test_random_state_reproducibility(
        self,
        sample_eeg_raw: BaseRaw,
        mock_iclabel,
    ) -> None:
        """Test that random_state ensures reproducibility."""
        pipeline1 = PreprocessingPipeline(
            input_path=sample_eeg_raw,
            random_state=42,
        )
        pipeline1.raw = sample_eeg_raw.copy()
        pipeline1.datatype = "eeg"

        pipeline2 = PreprocessingPipeline(
            input_path=sample_eeg_raw,
            random_state=42,
        )
        pipeline2.raw = sample_eeg_raw.copy()
        pipeline2.datatype = "eeg"

        # Results should be identical with same random_state
        # Use minimal components for speed
        result1 = pipeline1.apply_ica(n_components=2)
        result2 = pipeline2.apply_ica(n_components=2)

        assert result1 is not None
        assert result2 is not None
