"""Batch preprocessing for multiple MEG/EEG datasets.

This module provides batch processing capabilities for preprocessing
multiple datasets in parallel.
"""

from __future__ import annotations

from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger
from mne_bids import BIDSPath

from .pipeline import PreprocessingPipeline


class BatchPreprocessingPipeline:
    """Batch preprocessing pipeline for multiple MEG/EEG datasets.

    Parameters
    ----------
    input_paths : list[str | Path | BIDSPath]
        List of input paths to process.
    output_dir : str | Path | None, optional
        Output directory for all datasets. If None, uses BIDS derivatives.
    n_jobs : int, optional
        Number of parallel jobs. Default is 1.
    use_cuda : bool, optional
        Whether to use CUDA. Default is False.
    random_state : int, optional
        Random seed. Default is 42.

    Attributes
    ----------
    input_paths : list[Path | BIDSPath]
        Parsed input paths.
    output_dir : Path | None
        Output directory.
    n_jobs : int
        Number of parallel jobs.
    use_cuda : bool
        CUDA flag.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        input_paths: list[str | Path | BIDSPath],
        output_dir: str | Path | None = None,
        n_jobs: int = 1,
        use_cuda: bool = False,
        random_state: int = 42,
    ) -> None:
        """Initialize batch preprocessing pipeline."""
        # Validate input
        if not input_paths:
            raise ValueError("input_paths cannot be empty.")

        # Parse paths
        self.input_paths = [self._parse_path(p) for p in input_paths]

        # Configuration
        self.output_dir = Path(output_dir) if output_dir else None
        self.n_jobs = n_jobs
        self.use_cuda = use_cuda
        self.random_state = random_state

        logger.info(
            f"Initialized BatchPreprocessingPipeline with {len(self.input_paths)} datasets, "
            f"n_jobs={n_jobs}"
        )

    def _parse_path(self, path: str | Path | BIDSPath) -> Path | BIDSPath:
        """Parse a single path."""
        if isinstance(path, BIDSPath):
            return path
        elif isinstance(path, str | Path):
            return Path(path)
        else:
            raise TypeError(f"Invalid path type: {type(path)}")

    def _process_single(
        self,
        input_path: Path | BIDSPath,
        filter_params: dict | None,
        detect_bad_channels: bool,
        remove_line_noise: bool,
        apply_ica: bool,
        ica_params: dict | None,
        save_intermediate: bool,
        skip_existing: bool,
        logging_level: str,
    ) -> None:
        """Process a single dataset.

        This method is called in parallel for each dataset.
        """
        # Check if output already exists
        if skip_existing:
            if self._check_output_exists(input_path):
                logger.info(f"Skipping {input_path} (output already exists)")
                return

        # Set up logging for this worker
        # (In parallel mode, each worker should have its own log config)
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=logging_level,
            format="<level>{message}</level>",
        )

        try:
            # Create pipeline for this dataset
            pipeline = PreprocessingPipeline(
                input_path=input_path,
                output_dir=self.output_dir,
                n_jobs=1,  # Each worker uses 1 job
                use_cuda=self.use_cuda,
                random_state=self.random_state,
            )

            # Run pipeline
            pipeline.run(
                filter_params=filter_params,
                detect_bad_channels=detect_bad_channels,
                remove_line_noise=remove_line_noise,
                apply_ica=apply_ica,
                ica_params=ica_params,
                save_intermediate=save_intermediate,
            )

            # Save result
            pipeline.save()

            logger.success(f"Completed: {input_path}")

        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")

    def _check_output_exists(self, input_path: Path | BIDSPath) -> bool:
        """Check if output file already exists."""
        if self.output_dir is None:
            return False

        # Try to construct expected output path
        if isinstance(input_path, BIDSPath):
            subject = input_path.subject
            session = input_path.session
            datatype = input_path.datatype
            basename = input_path.basename
            output_path = (
                self.output_dir
                / f"sub-{subject}"
                / f"ses-{session}"
                / datatype
                / f"{basename}_preproc_{datatype}.fif"
            )
        else:
            # For non-BIDS paths, check in output_dir
            output_path = self.output_dir / f"{input_path.stem}_preproc_*.fif"

        return output_path.exists() if isinstance(output_path, Path) else False

    def run(
        self,
        filter_params: dict | None = None,
        detect_bad_channels: bool = True,
        remove_line_noise: bool = True,
        apply_ica: bool = True,
        ica_params: dict | None = None,
        save_intermediate: bool = True,
        skip_existing: bool = False,
        logging_level: str = "INFO",
        save_logs: bool = False,
    ) -> None:
        """Run batch preprocessing on all datasets.

        Parameters
        ----------
        filter_params : dict | None, optional
            Filtering parameters.
        detect_bad_channels : bool, optional
            Whether to detect bad channels. Default is True.
        remove_line_noise : bool, optional
            Whether to remove line noise. Default is True.
        apply_ica : bool, optional
            Whether to apply ICA. Default is True.
        ica_params : dict | None, optional
            ICA parameters.
        save_intermediate : bool, optional
            Whether to save intermediate files. Default is True.
        skip_existing : bool, optional
            Whether to skip datasets with existing output. Default is False.
        logging_level : str, optional
            Logging level. Default is "INFO".
        save_logs : bool, optional
            Whether to save log files. Default is False.
        """
        logger.info(f"Starting batch preprocessing of {len(self.input_paths)} datasets...")

        # Setup log file if requested
        log_file_handler = None
        if save_logs:
            log_file = self.output_dir / "batch_preprocessing.log"
            log_file_handler = logger.add(
                log_file,
                rotation=None,
                level=logging_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            )
            logger.info(f"Logging to file: {log_file}")

        if self.n_jobs == 1:
            # Sequential processing
            for input_path in self.input_paths:
                self._process_single(
                    input_path=input_path,
                    filter_params=filter_params,
                    detect_bad_channels=detect_bad_channels,
                    remove_line_noise=remove_line_noise,
                    apply_ica=apply_ica,
                    ica_params=ica_params,
                    save_intermediate=save_intermediate,
                    skip_existing=skip_existing,
                    logging_level=logging_level,
                )
        else:
            # Parallel processing
            logger.info(f"Processing in parallel with {self.n_jobs} jobs...")

            Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_single)(
                    input_path=input_path,
                    filter_params=filter_params,
                    detect_bad_channels=detect_bad_channels,
                    remove_line_noise=remove_line_noise,
                    apply_ica=apply_ica,
                    ica_params=ica_params,
                    save_intermediate=save_intermediate,
                    skip_existing=skip_existing,
                    logging_level=logging_level,
                )
                for input_path in self.input_paths
            )

        logger.success(f"Batch preprocessing completed for {len(self.input_paths)} datasets!")

        # Remove log file handler if it was added
        if save_logs and log_file_handler is not None:
            try:
                logger.remove(log_file_handler)
            except ValueError:
                # Handler may have already been removed
                pass
