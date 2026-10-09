"""Processing many recordings with one pipeline."""

from __future__ import annotations

import time
import traceback
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from mne_bids import BIDSPath
from sklearn.base import clone

from .core import Pipeline
from .io import save_derivative
from .io.derivatives import existing_derivatives
from .io.read import as_bids_path

ON_ERROR = ("raise", "warn")


def process(
    pipeline: Pipeline,
    sources: Sequence[str | Path | BIDSPath],
    root: str | Path,
    *,
    desc: str = "preproc",
    n_jobs: int = 1,
    on_error: str = "raise",
    skip_existing: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Process each recording with its own copy of a pipeline and save the results.

    Every recording is processed independently (the pipeline is cloned and
    fitted per recording, see the multi-run policy in the design notes) and
    written with :func:`~meeg_utils.io.save_derivative`, so each output has a
    sidecar with the pipeline configuration, provenance and QC metrics.

    Parameters
    ----------
    pipeline : Pipeline
        The (unfitted) pipeline; it is not modified.
    sources : sequence of str | Path | BIDSPath
        The recordings.
    root : str | Path
        Root of the derivatives dataset.
    desc : str
        Value of the BIDS ``desc`` entity of the outputs.
    n_jobs : int
        Recordings processed in parallel (joblib); ``-1`` uses all cores.
    on_error : {"raise", "warn"}
        What to do once all recordings are processed and some failed: raise
        a :class:`RuntimeError` listing them, or emit a warning. Failures
        never stop the other recordings.
    skip_existing : bool
        Skip recordings whose output already exists.
    overwrite : bool
        Replace existing outputs (otherwise an existing output is a failure).

    Returns
    -------
    list of dict
        One record per recording: ``source``, ``status`` (``"ok"``,
        ``"skipped"`` or ``"failed"``), ``output``, ``error`` and
        ``duration_s``.

    Raises
    ------
    RuntimeError
        If ``on_error="raise"`` and at least one recording failed.
    """
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}, got {on_error!r}")
    if skip_existing and overwrite:
        raise ValueError("skip_existing and overwrite are mutually exclusive.")
    if not sources:
        raise ValueError("No recordings to process.")
    root = Path(root)

    if n_jobs == 1:
        records = [_process_one(pipeline, s, root, desc, skip_existing, overwrite) for s in sources]
    else:
        from joblib import Parallel, delayed

        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_one)(pipeline, s, root, desc, skip_existing, overwrite)
            for s in sources
        )

    failed = [r for r in records if r["status"] == "failed"]
    n_ok = sum(r["status"] == "ok" for r in records)
    logger.info(f"Processed {n_ok}/{len(records)} recordings ({len(failed)} failed).")
    if failed:
        lines = "\n".join(f"  {r['source']}: {r['error']}" for r in failed)
        message = f"{len(failed)} of {len(records)} recordings failed:\n{lines}"
        if on_error == "raise":
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    return records


def _process_one(
    pipeline: Pipeline,
    source: str | Path | BIDSPath,
    root: Path,
    desc: str,
    skip_existing: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if not isinstance(source, BIDSPath):
        source = as_bids_path(Path(source)) or Path(source)
    record: dict[str, Any] = {
        "source": str(source.fpath if isinstance(source, BIDSPath) else source),
        "status": "ok",
        "output": None,
        "error": None,
        "duration_s": 0.0,
    }
    start = time.perf_counter()
    try:
        existing = existing_derivatives(source, root, desc=desc)
        if skip_existing and existing:
            record.update(status="skipped", output=str(existing[0]))
            logger.info(f"Skipping {record['source']}: {existing[0]} exists")
            return record
        pipe = clone(pipeline)
        out = pipe.fit_transform(source)
        fname = save_derivative(out, source, root, pipeline=pipe, desc=desc, overwrite=overwrite)
        record["output"] = str(fname)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.error(f"Failed to process {record['source']}:\n{traceback.format_exc()}")
    record["duration_s"] = round(time.perf_counter() - start, 3)
    return record
