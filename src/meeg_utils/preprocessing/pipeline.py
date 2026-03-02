"""Preprocessing pipeline for single MEG/EEG datasets.

This module provides the main PreprocessingPipeline class for processing
individual MEG/EEG datasets with filtering, bad channel detection,
line noise removal, and ICA artifact removal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mne
from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne_bids import BIDSPath, read_raw_bids

DType = Literal["eeg", "meg"]


class PreprocessingPipeline:
    """Main preprocessing pipeline for MEG/EEG data.

    This class provides a comprehensive preprocessing pipeline including:
    - Flexible input path parsing (string, Path, BIDSPath)
    - Data loading with automatic datatype inference
    - Filtering and resampling
    - Bad channel detection and interpolation
    - Line noise removal (zapline)
    - ICA artifact removal with automatic/manual labeling

    Parameters
    ----------
    input_path : str | Path | BIDSPath | BaseRaw
        Path to the input data, or a Raw object directly.
        Can be a plain string, pathlib.Path, mne_bids.BIDSPath object,
        or an MNE Raw object (useful for testing).
    output_dir : str | Path | None, optional
        Directory for saving outputs. If None, uses BIDS derivatives.
    n_jobs : int, optional
        Number of parallel jobs. Default is 1.
    use_cuda : bool, optional
        Whether to use CUDA acceleration. Default is False.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Attributes
    ----------
    input_path : Path | BIDSPath | None
        Parsed input path (None if Raw was provided directly).
    output_dir : Path
        Output directory path.
    n_jobs : int | str
        Number of parallel jobs or "cuda".
    random_state : int
        Random seed.
    raw : BaseRaw | None
        Loaded raw data.
    datatype : DType | None
        Inferred datatype ("eeg" or "meg").
    """

    def __init__(
        self,
        input_path: str | Path | BIDSPath | BaseRaw,
        output_dir: str | Path | None = None,
        n_jobs: int = 1,
        use_cuda: bool = False,
        random_state: int = 42,
    ) -> None:
        """Initialize preprocessing pipeline."""
        # Initialize state first (before parsing input)
        self.raw: BaseRaw | None = None
        self.datatype: DType | None = None
        self._ica: ICA | None = None
        self._ic_labels: dict | None = None

        # Parse input path (may set self.raw and self.datatype)
        self.input_path = self._parse_input_path(input_path)

        # Set up output directory
        if output_dir is None:
            if self.input_path is not None and isinstance(self.input_path, BIDSPath):
                self.output_dir = Path(self.input_path.root) / "derivatives" / "preproc"
            elif self.input_path is not None:
                self.output_dir = Path(self.input_path).parent / "derivatives" / "preproc"
            else:
                # Raw object was provided, use current directory
                self.output_dir = Path.cwd() / "derivatives" / "preproc"
        else:
            self.output_dir = Path(output_dir)

        # Configuration
        self.random_state = random_state

        # CUDA setup
        if use_cuda:
            logger.trace("Using CUDA for computations where possible.")
            self.n_jobs: int | str = "cuda"
            try:
                mne.cuda.init_cuda()
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA: {e}. Falling back to CPU.")
                self.n_jobs = n_jobs
        else:
            self.n_jobs = n_jobs
            logger.trace(f"Using {n_jobs} CPU cores for computations where possible.")

        logger.info(f"Initialized PreprocessingPipeline for {self.input_path}")

    def _parse_input_path(
        self, input_path: str | Path | BIDSPath | BaseRaw
    ) -> Path | BIDSPath | None:
        """Parse and validate input path.

        Parameters
        ----------
        input_path : str | Path | BIDSPath | BaseRaw
            Input path in various formats, or a Raw object for testing.

        Returns
        -------
        Path | BIDSPath | None
            Validated path object, or None if Raw was provided.

        Raises
        ------
        TypeError
            If input_path is not a valid type.
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the path is invalid.
        """
        # Special case: allow passing Raw object directly (for testing)
        if isinstance(input_path, BaseRaw):
            self.raw = input_path
            self.datatype = self._infer_datatype(input_path)
            return None

        # Type check
        if not isinstance(input_path, str | Path | BIDSPath):
            raise TypeError(
                f"input_path must be str, Path, BIDSPath, or BaseRaw, got {type(input_path)}"
            )

        # Convert to appropriate type
        if isinstance(input_path, BIDSPath):
            path = input_path
            # Validate BIDSPath exists
            if path.fpath is None or not Path(path.fpath).exists():
                raise FileNotFoundError(f"BIDSPath does not exist: {path}")
            # Infer datatype from BIDSPath
            if path.datatype in ["eeg", "meg"]:
                self.datatype = path.datatype
        elif isinstance(input_path, str | Path):
            path = Path(input_path)
            # Validate path exists
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")
        else:
            raise ValueError(f"Invalid input_path type: {type(input_path)}")

        return path

    def _infer_datatype(self, raw: BaseRaw) -> DType:
        """Infer datatype (EEG or MEG) from raw data.

        Parameters
        ----------
        raw : BaseRaw
            Raw data object.

        Returns
        -------
        DType
            Inferred datatype: "eeg" or "meg".

        Raises
        ------
        ValueError
            If datatype cannot be inferred.
        """
        ch_types = set(raw.get_channel_types())

        if "eeg" in ch_types:
            return "eeg"
        elif any(t in ch_types for t in ("mag", "grad", "planar1", "planar2")):
            return "meg"
        else:
            raise ValueError(
                f"Cannot infer datatype from channel types: {ch_types}. "
                "Must contain 'eeg' or MEG-related types."
            )

    def load_data(self) -> BaseRaw:
        """Load data from input path.

        Returns
        -------
        BaseRaw
            Loaded raw data.

        Raises
        ------
        ValueError
            If data cannot be loaded.
        """
        # Return cached data if already loaded
        if self.raw is not None:
            logger.trace("Returning cached raw data.")
            return self.raw

        logger.info("Loading data...")

        # Load based on path type
        if isinstance(self.input_path, BIDSPath):
            self.raw = read_raw_bids(self.input_path, verbose=False)
            self.raw.load_data()
        else:
            # Try to read as standard MNE format
            try:
                self.raw = mne.io.read_raw(self.input_path, preload=True, verbose=False)
            except Exception as e:
                raise ValueError(f"Failed to load data from {self.input_path}: {e}")

        # Infer and store datatype
        self.datatype = self._infer_datatype(self.raw)
        logger.info(f"Loaded {self.datatype.upper()} data with {len(self.raw.ch_names)} channels.")

        return self.raw

    def filter_and_resample(
        self,
        highpass: float = 0.1,
        lowpass: float = 100.0,
        sfreq: float = 250.0,
    ) -> BaseRaw:
        """Apply bandpass filter and resample data.

        Parameters
        ----------
        highpass : float, optional
            High-pass filter frequency in Hz. Default is 0.1.
        lowpass : float, optional
            Low-pass filter frequency in Hz. Default is 100.0.
        sfreq : float, optional
            Target sampling frequency in Hz. Default is 250.0.

        Returns
        -------
        BaseRaw
            Filtered and resampled raw data.

        Raises
        ------
        AssertionError
            If filter parameters are invalid.
        ValueError
            If data is not loaded.
        """
        if self.raw is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Validate parameters
        nyquist = sfreq / 2
        assert (
            lowpass < nyquist
        ), f"Lowpass frequency ({lowpass} Hz) must be less than Nyquist frequency ({nyquist} Hz)."
        assert highpass < lowpass, (
            f"Highpass frequency ({highpass} Hz) must be less than "
            f"lowpass frequency ({lowpass} Hz)."
        )

        logger.info(
            f"Filtering: highpass={highpass} Hz, lowpass={lowpass} Hz, resampling to {sfreq} Hz"
        )

        # Apply filter
        self.raw.filter(l_freq=highpass, h_freq=lowpass, n_jobs=self.n_jobs, verbose=False)

        # Resample
        self.raw.resample(sfreq, n_jobs=self.n_jobs, verbose=False)

        logger.success("Filtering and resampling completed.")

        return self.raw

    def detect_and_fix_bad_channels(
        self,
        fix: bool = True,
        reset_bads: bool = True,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.04),
    ) -> BaseRaw:
        """Detect and optionally interpolate bad channels.

        Parameters
        ----------
        fix : bool, optional
            Whether to interpolate bad channels. Default is True.
        reset_bads : bool, optional
            Whether to reset bads list after interpolation. Default is True.
        origin : tuple, optional
            Origin for MEG interpolation. Default is (0.0, 0.0, 0.04).

        Returns
        -------
        BaseRaw
            Raw data with bad channels marked/fixed.

        Raises
        ------
        ValueError
            If data not loaded or datatype not inferred.
        """
        if self.raw is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if self.datatype is None:
            self.datatype = self._infer_datatype(self.raw)

        logger.info("Detecting bad channels...")

        # Import detection modules
        from .bad_channels import detect_bad_channels_eeg, detect_bad_channels_meg

        # Detect based on datatype
        if self.datatype == "eeg":
            bad_channels = detect_bad_channels_eeg(self.raw, random_state=self.random_state)
        elif self.datatype == "meg":
            bad_channels = detect_bad_channels_meg(self.raw, origin=origin)
        else:
            raise ValueError(f"Unsupported datatype: {self.datatype}")

        # Mark bad channels
        self.raw.info["bads"].extend(bad_channels)
        logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")

        # Interpolate if requested
        if fix and len(bad_channels) > 0:
            logger.info("Interpolating bad channels...")
            self.raw.interpolate_bads(
                reset_bads=reset_bads,
                method=dict(meg="MNE", eeg="spline"),
                origin=origin if self.datatype == "meg" else "auto",
                verbose=False,
            )
            logger.success("Bad channels interpolated.")

        # Save derivative
        self._save_bad_channels_tsv(bad_channels)

        return self.raw

    def remove_line_noise(
        self,
        fline: float = 50.0,
    ) -> BaseRaw:
        """Remove power line noise using zapline method.

        Parameters
        ----------
        fline : float, optional
            Power line frequency in Hz. Default is 50.0.

        Returns
        -------
        BaseRaw
            Raw data with line noise removed.

        Raises
        ------
        ValueError
            If data not loaded or datatype not inferred.
        """
        if self.raw is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if self.datatype is None:
            self.datatype = self._infer_datatype(self.raw)

        logger.info(f"Removing {fline} Hz line noise...")

        from .line_noise import remove_line_noise_eeg, remove_line_noise_meg

        # Apply based on datatype
        if self.datatype == "eeg":
            self.raw = remove_line_noise_eeg(self.raw, fline=fline)
        elif self.datatype == "meg":
            self.raw = remove_line_noise_meg(self.raw, fline=fline)
        else:
            raise ValueError(f"Unsupported datatype: {self.datatype}")

        logger.success("Line noise removal completed.")

        return self.raw

    def apply_ica(
        self,
        n_components: int | None = None,
        method: str = "infomax",
        regress: bool = False,
        manual_labels: list[str] | None = None,
    ) -> BaseRaw:
        """Apply ICA for artifact removal.

        Parameters
        ----------
        n_components : int | None, optional
            Number of ICA components. If None, uses default (20 for EEG, 40 for MEG).
        method : str, optional
            ICA method. Default is "infomax".
        regress : bool, optional
            Whether to regress out artifact components. Default is False.
        manual_labels : list[str] | None, optional
            Manual labels for ICA components. If None, uses automatic labeling.

        Returns
        -------
        BaseRaw
            Raw data with ICA applied (if regress=True).

        Raises
        ------
        ValueError
            If data not loaded or datatype not inferred.
        """
        if self.raw is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if self.datatype is None:
            self.datatype = self._infer_datatype(self.raw)

        logger.info("Applying ICA...")

        from .ica import apply_ica_pipeline

        # Determine default n_components
        if n_components is None:
            n_components = 20 if self.datatype == "eeg" else 40

        # Apply ICA
        result = apply_ica_pipeline(
            raw=self.raw,
            datatype=self.datatype,
            n_components=n_components,
            method=method,
            regress=regress,
            manual_labels=manual_labels,
            random_state=self.random_state,
        )

        if regress:
            self.raw = result
            logger.success("ICA artifact regression completed.")
        else:
            logger.success("ICA decomposition completed (no regression).")

        return self.raw

    def run(
        self,
        filter_params: dict | None = None,
        detect_bad_channels: bool = True,
        remove_line_noise: bool = True,
        apply_ica: bool = True,
        ica_params: dict | None = None,
        save_intermediate: bool = True,
    ) -> BaseRaw:
        """Run complete preprocessing pipeline.

        Parameters
        ----------
        filter_params : dict | None, optional
            Filtering parameters. Keys: highpass, lowpass, sfreq.
        detect_bad_channels : bool, optional
            Whether to detect and fix bad channels. Default is True.
        remove_line_noise : bool, optional
            Whether to remove line noise. Default is True.
        apply_ica : bool, optional
            Whether to apply ICA. Default is True.
        ica_params : dict | None, optional
            ICA parameters. Keys: n_components, method, regress.
        save_intermediate : bool, optional
            Whether to save intermediate files. Default is True.

        Returns
        -------
        BaseRaw
            Preprocessed raw data.
        """
        logger.info("Starting preprocessing pipeline...")

        # Load data
        self.load_data()

        # Filtering
        if filter_params is None:
            filter_params = {"highpass": 0.1, "lowpass": 100.0, "sfreq": 250.0}

        self.filter_and_resample(**filter_params)

        # Bad channel detection
        if detect_bad_channels:
            self.detect_and_fix_bad_channels()

        # Line noise removal
        if remove_line_noise:
            self.remove_line_noise()

        # Apply reference
        self._apply_reference()

        # ICA
        if apply_ica:
            if ica_params is None:
                ica_params = {}
            self.apply_ica(**ica_params)

        # Re-reference EEG after ICA if regressed
        if apply_ica and ica_params.get("regress", False) and self.datatype == "eeg":
            logger.info("Re-referencing EEG after ICA regression.")
            self.raw.set_eeg_reference("average", verbose=False)

        logger.success("Preprocessing pipeline completed!")

        return self.raw

    def _apply_reference(self) -> None:
        """Apply appropriate reference for the datatype."""
        if self.datatype == "eeg":
            logger.info("Applying average reference for EEG.")
            self.raw.set_eeg_reference("average", verbose=False)
        elif self.datatype == "meg":
            logger.info("Applying gradient compensation 3 for MEG.")
            self.raw.apply_gradient_compensation(3, verbose=False)

    def save(self, filename: str | Path | None = None) -> None:
        """Save preprocessed data.

        Parameters
        ----------
        filename : str | Path | None, optional
            Output filename. If None, generates BIDS-compliant name.
        """
        if self.raw is None:
            raise ValueError("No data to save. Run pipeline first.")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            if isinstance(self.input_path, BIDSPath):
                basename = self.input_path.basename
                subject = self.input_path.subject
                session = self.input_path.session
                subdir = self.output_dir / f"sub-{subject}" / f"ses-{session}" / self.datatype
                subdir.mkdir(parents=True, exist_ok=True)
                filename = subdir / f"{basename}_preproc_{self.datatype}.fif"
            else:
                filename = self.output_dir / f"{self.input_path.stem}_preproc_{self.datatype}.fif"
        else:
            filename = Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving preprocessed data to {filename}")
        self.raw.save(filename, overwrite=True, verbose=False)
        logger.success(f"Saved to {filename}")

    def _save_bad_channels_tsv(self, bad_channels: list[str]) -> None:
        """Save bad channels information to TSV file."""
        if not hasattr(self, "output_dir"):
            return

        import json

        import pandas as pd

        # Create filename
        if self.input_path is None:
            # Use default naming when Raw object was provided directly
            subdir = self.output_dir
            basename = "preprocessed"
        elif isinstance(self.input_path, BIDSPath):
            basename = self.input_path.basename
            subject = self.input_path.subject
            session = self.input_path.session
            subdir = self.output_dir / f"sub-{subject}" / f"ses-{session}" / self.datatype
        else:
            subdir = self.output_dir
            basename = self.input_path.stem

        subdir.mkdir(parents=True, exist_ok=True)
        fname = subdir / f"{basename}_desc-badchs_{self.datatype}.tsv"

        # Create dataframe
        chs = self.raw.ch_names
        status = ["good"] * len(chs)
        status_desc = ["fixed" if ch in bad_channels else "n/a" for ch in chs]

        df = pd.DataFrame(
            {
                "name": chs,
                "type": [self.datatype] * len(chs),
                "status": status,
                "status_description": status_desc,
            }
        )

        # Save TSV
        df.to_csv(fname, sep="\t", index=False, encoding="utf-8", na_rep="n/a")

        # Save JSON sidecar
        fname_json = fname.with_suffix(".json")
        meta = {
            "name": "Channels' name",
            "type": "Channel type, e.g., EEG, MEG",
            "status": "Channel status, good or bad",
            "status_description": "Description of the channel status, e.g., fixed if interpolated",
        }
        with open(fname_json, "w") as f:
            json.dump(meta, f, indent=4)

        logger.trace(f"Saved bad channels info to {fname}")
