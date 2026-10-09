"""Batch preprocessing for multiple MEG/EEG datasets.

This module provides batch processing capabilities for preprocessing
multiple datasets in parallel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from joblib import Parallel, delayed  # type: ignore[import-untyped]
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
    ) -> dict[str, Any]:
        """Process a single dataset.

        This method is called in parallel for each dataset. It never raises;
        failures are reported in the returned record.
        """
        record: dict[str, Any] = {"input": str(input_path), "status": "ok", "error": None}

        # Check if output already exists
        if skip_existing:
            if self._check_output_exists(input_path):
                logger.info(f"Skipping {input_path} (output already exists)")
                record["status"] = "skipped"
                return record

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
            logger.exception(f"Error processing {input_path}: {e}")
            record["status"] = "failed"
            record["error"] = f"{type(e).__name__}: {e}"

        return record

    def _check_output_exists(self, input_path: Path | BIDSPath) -> bool:
        """Check if output file already exists."""
        if self.output_dir is None:
            return False

        # Try to construct expected output path
        if isinstance(input_path, BIDSPath):
            subject = input_path.subject
            session = input_path.session
            datatype = input_path.datatype
            basename = input_path.basename.split(".")[0]  # same naming as save()
            output_path = (
                self.output_dir
                / f"sub-{subject}"
                / f"ses-{session}"
                / datatype
                / f"{basename}_preproc_{datatype}.fif"
            )
        else:
            # For non-BIDS paths, check in output_dir
            return any(self.output_dir.glob(f"{input_path.stem}_preproc_*.fif"))

        return bool(output_path.exists())

    def run(
        self,
        filter_params: dict | None = None,
        detect_bad_channels: bool = True,
        remove_line_noise: bool = True,
        apply_ica: bool = True,
        ica_params: dict | None = None,
        save_intermediate: bool = False,
        skip_existing: bool = False,
        save_logs: bool = True,
        logging_level: str = "DEBUG",
        on_error: Literal["raise", "warn"] = "raise",
    ) -> list[dict[str, Any]]:
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
            Whether to save intermediate files. Default is False.
        skip_existing : bool, optional
            Whether to skip datasets with existing output. Default is False.
        save_logs : bool, optional
            Whether to write a log file to ``<output_dir>/logs`` for this run.
            Default is True. Console output is controlled separately with
            :func:`meeg_utils.setup_logging`.
        logging_level : str, optional
            Minimum level written to the log file. Default is "DEBUG".
        on_error : {"raise", "warn"}, optional
            What to do when datasets fail. All datasets are processed either
            way; with ``"raise"`` (default) a ``RuntimeError`` listing the
            failures is raised at the end, with ``"warn"`` the failures are
            only logged and reported in the returned records.

        Returns
        -------
        list of dict
            One record per dataset with keys ``"input"``, ``"status"``
            (``"ok"``, ``"skipped"`` or ``"failed"``) and ``"error"``.

        Raises
        ------
        RuntimeError
            If any dataset failed and ``on_error="raise"``.
        """
        if on_error not in ("raise", "warn"):
            raise ValueError(f"on_error must be 'raise' or 'warn', got {on_error!r}.")

        kwargs: dict[str, Any] = dict(
            filter_params=filter_params,
            detect_bad_channels=detect_bad_channels,
            remove_line_noise=remove_line_noise,
            apply_ica=apply_ica,
            ica_params=ica_params,
            save_intermediate=save_intermediate,
            skip_existing=skip_existing,
        )
        if save_logs and self.output_dir:
            from datetime import datetime

            from ..logger import log_to_file

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.output_dir / "logs" / f"batch_preprocessing_{stamp}.log"
            with log_to_file(log_file, level=logging_level):
                records = self._run_all(**kwargs)
        else:
            records = self._run_all(**kwargs)

        failed = [r for r in records if r["status"] == "failed"]
        if failed:
            summary = "\n".join(f"  {r['input']}: {r['error']}" for r in failed)
            message = f"{len(failed)} of {len(records)} datasets failed:\n{summary}"
            if on_error == "raise":
                raise RuntimeError(message)
            logger.warning(message)
        return records

    def _run_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        logger.info(f"Starting batch preprocessing of {len(self.input_paths)} datasets...")

        if self.n_jobs == 1:
            records = [self._process_single(input_path=p, **kwargs) for p in self.input_paths]
        else:
            logger.info(f"Processing in parallel with {self.n_jobs} jobs...")
            records = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_single)(input_path=p, **kwargs) for p in self.input_paths
            )

        n_ok = sum(r["status"] == "ok" for r in records)
        logger.info(f"Batch preprocessing finished: {n_ok}/{len(records)} datasets processed.")
        return list(records)
