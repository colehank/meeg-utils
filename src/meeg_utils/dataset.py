"""Whole datasets: select recordings, preprocess each run, epoch and combine per session."""

from __future__ import annotations

import time
import traceback
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from mne_bids import BIDSPath
from sklearn.base import clone

from .core import Pipeline

ON_ERROR = ("raise", "warn")
#: BIDS entities that distinguish recordings of the same session and task.
_RUN_ENTITIES = ("run", "split")


class Dataset:
    """The MEG/EEG recordings of a BIDS dataset, selected by entity.

    Parameters
    ----------
    root : str | Path
        Root of the BIDS dataset.
    datatype : {"meg", "eeg"} | None
        Keep only this datatype.
    subjects, sessions, tasks, runs : list of str | None
        Keep only these labels (without the ``sub-`` etc. prefixes).
    exclude : list of str
        Recordings to leave out: any whose file name contains one of these
        strings (e.g. ``["sub-07", "task-rest_run-02"]``).

    Attributes
    ----------
    recordings : list of BIDSPath
        The selected recordings, sorted.

    Examples
    --------
    >>> ds = meu.Dataset("/data/bids", datatype="meg", tasks=["faces"])  # doctest: +SKIP
    >>> ds.groups()  # doctest: +SKIP
    {'sub-01_ses-01_task-faces_meg': [BIDSPath(...run-01...), BIDSPath(...run-02...)], ...}
    """

    def __init__(
        self,
        root: str | Path,
        *,
        datatype: str | None = None,
        subjects: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        runs: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
    ) -> None:
        from .qc.dataset import find_recordings

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory.")
        selection = {"subject": subjects, "session": sessions, "task": tasks, "run": runs}
        recordings = []
        for path in find_recordings(self.root):
            if datatype is not None and path.datatype != datatype:
                continue
            if any(
                values is not None and str(getattr(path, entity)) not in {str(v) for v in values}
                for entity, values in selection.items()
            ):
                continue
            if any(pattern in path.basename for pattern in exclude):
                continue
            recordings.append(path)
        self.recordings: list[BIDSPath] = recordings

    def groups(self) -> dict[str, list[BIDSPath]]:
        """Group the recordings that belong together: the runs of one session and task.

        Returns
        -------
        dict
            Group name (the file name without the run) mapped to its
            recordings, in run order.
        """
        groups: dict[str, list[BIDSPath]] = {}
        for path in self.recordings:
            groups.setdefault(group_name(path), []).append(path)
        return groups

    def table(self) -> list[dict[str, Any]]:
        """One row per recording with its BIDS entities and file.

        Returns
        -------
        list of dict
            ``subject``, ``session``, ``task``, ``acquisition``, ``run``,
            ``datatype``, ``group`` and ``path``.
        """
        return [
            {
                "subject": p.subject,
                "session": p.session,
                "task": p.task,
                "acquisition": p.acquisition,
                "run": p.run,
                "datatype": p.datatype,
                "group": group_name(p),
                "path": str(p.fpath),
            }
            for p in self.recordings
        ]

    def __len__(self) -> int:
        """Return the number of recordings."""
        return len(self.recordings)

    def __iter__(self) -> Iterator[BIDSPath]:
        """Iterate over the recordings."""
        return iter(self.recordings)

    def __repr__(self) -> str:
        """Summarize the selection."""
        subjects = {p.subject for p in self.recordings}
        return (
            f"<Dataset {self.root.name}: {len(self.recordings)} recordings, "
            f"{len(subjects)} subjects, {len(self.groups())} session/task groups>"
        )


def group_name(path: BIDSPath) -> str:
    """Name of the group of a recording: its file name without the run and split.

    Parameters
    ----------
    path : BIDSPath
        The recording.

    Returns
    -------
    str
        E.g. ``"sub-01_ses-01_task-faces_meg"``.
    """
    return str(path.copy().update(run=None, split=None, extension=None, check=False).basename)


class BatchResult:
    """Outcome of :func:`process_dataset`: one record per run and stage, and per group.

    Attributes
    ----------
    records : list of dict
        ``stage`` (``"preprocessing"``, ``"epochs"`` or ``"combine"``),
        ``source`` (a recording, or a group for ``"combine"``), ``status``
        (``"ok"``, ``"skipped"`` or ``"failed"``), ``output``, ``error`` and
        ``duration_s``.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    @property
    def failed(self) -> list[dict[str, Any]]:
        """The failed records."""
        return [r for r in self.records if r["status"] == "failed"]

    def outputs(self, stage: str) -> list[str]:
        """Files written (or found, when skipped) by a stage.

        Parameters
        ----------
        stage : {"preprocessing", "epochs", "combine"}
            The stage.

        Returns
        -------
        list of str
            The output files.
        """
        return [r["output"] for r in self.records if r["stage"] == stage and r["output"]]

    def to_csv(self, fname: str | Path) -> Path:
        """Write the records as a CSV table.

        Parameters
        ----------
        fname : str | Path
            Output file.

        Returns
        -------
        Path
            The file written.
        """
        import csv

        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        fields = ["stage", "source", "status", "output", "error", "duration_s"]
        with fname.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r.get(k) for k in fields} for r in self.records)
        return fname

    def __repr__(self) -> str:
        """Count the records per stage and status."""
        lines = []
        for stage in ("preprocessing", "epochs", "combine"):
            records = [r for r in self.records if r["stage"] == stage]
            if records:
                counts = {
                    s: sum(r["status"] == s for r in records) for s in ("ok", "skipped", "failed")
                }
                lines.append(
                    f"  {stage}: {counts['ok']} ok, {counts['skipped']} skipped, "
                    f"{counts['failed']} failed"
                )
        return "<BatchResult\n" + "\n".join(lines) + ">"


def process_dataset(
    dataset: Dataset,
    root: str | Path,
    *,
    preprocessing: Pipeline,
    epochs: Pipeline | None = None,
    combine: bool = True,
    align: str | None = "average",
    desc: str = "preproc",
    epochs_desc: str = "epochs",
    n_jobs: int = 1,
    skip_existing: bool = True,
    on_error: str = "warn",
) -> BatchResult:
    """Process a whole dataset: each run, then the epochs of each session.

    1. **Preprocessing**: every recording is processed with its own copy of
       ``preprocessing`` (:func:`meeg_utils.process`) and saved with
       ``desc``.
    2. **Epochs** (if ``epochs`` is given), per group of runs
       (:meth:`Dataset.groups`): for MEG with ``align="average"``, the runs
       are first mapped to their average head position
       (:func:`meeg_utils.io.average_dev_head_t`; a :class:`~meeg_utils.steps.HeadAlign`
       step in ``epochs`` gets it as destination, otherwise one is inserted
       before epoching). Each run is epoched with its own copy of ``epochs``
       and saved with ``epochs_desc``.
    3. **Combine** (if ``combine``): the runs of each group are concatenated
       (:func:`meeg_utils.epochs.combine`) and saved without the run entity.

    Work already done is skipped (``skip_existing``), so an interrupted
    batch resumes where it stopped. Failures do not stop other recordings;
    they are collected in the result.

    Parameters
    ----------
    dataset : Dataset
        The recordings.
    root : str | Path
        Root of the derivatives dataset.
    preprocessing : Pipeline
        Per-run preprocessing (unfitted; not modified).
    epochs : Pipeline | None
        Per-run epoching, applied to the preprocessed runs (Raw -> Epochs).
    combine : bool
        Concatenate the epochs of each group's runs.
    align : {"average"} | None
        MEG head alignment of each group's runs before epoching; ``None``
        leaves them (``combine`` then requires their head positions to differ
        by less than 2 mm).
    desc, epochs_desc : str
        BIDS ``desc`` of the preprocessed runs and of the epochs.
    n_jobs : int
        Recordings (stage 1) or groups (stages 2-3) processed in parallel.
    skip_existing : bool
        Skip outputs that already exist (default ``True``: resume).
    on_error : {"warn", "raise"}
        After all recordings are processed, warn about failures or raise a
        :class:`RuntimeError` listing them.

    Returns
    -------
    BatchResult
        The records of every run and group; ``result.to_csv(...)`` saves them.
    """
    from .batch import process

    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}, got {on_error!r}")
    if align not in (None, "average"):
        raise ValueError(f"align must be 'average' or None, got {align!r}")
    if not len(dataset):
        raise ValueError("The dataset selection has no recordings.")
    root = Path(root)

    records = [
        {**r, "stage": "preprocessing"}
        for r in process(
            preprocessing,
            dataset.recordings,
            root,
            desc=desc,
            n_jobs=n_jobs,
            on_error="warn",
            skip_existing=skip_existing,
        )
    ]
    if epochs is not None:
        done = {r["source"]: r["output"] for r in records if r["output"]}
        jobs = []
        for group, paths in dataset.groups().items():
            runs = [(p, done[str(p.fpath)]) for p in paths if str(p.fpath) in done]
            if runs:
                jobs.append((group, runs))
        args = (epochs, root, combine, align, epochs_desc, skip_existing)
        if n_jobs == 1:
            results = [_epoch_group(group, runs, *args) for group, runs in jobs]
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(_epoch_group)(group, runs, *args) for group, runs in jobs
            )
        for group_records in results:
            records.extend(group_records)

    result = BatchResult(records)
    logger.info(f"Dataset processed: {result!r}")
    if result.failed:
        lines = "\n".join(f"  [{r['stage']}] {r['source']}: {r['error']}" for r in result.failed)
        message = f"{len(result.failed)} of {len(records)} tasks failed:\n{lines}"
        if on_error == "raise":
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    return result


def _epoch_group(
    group: str,
    runs: list[tuple[BIDSPath, str]],
    epochs: Pipeline,
    root: Path,
    combine: bool,
    align: str | None,
    desc: str,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    """Epoch the preprocessed runs of one group, then combine them."""
    from .io import average_dev_head_t, save_derivative
    from .io.derivatives import existing_derivatives

    records: list[dict[str, Any]] = []
    pipeline = epochs
    start = time.perf_counter()
    try:
        if align == "average" and runs[0][0].datatype == "meg":
            pipeline = _with_destination(epochs, average_dev_head_t([out for _, out in runs]))
    except Exception as exc:
        logger.error(f"Head alignment of {group} failed:\n{traceback.format_exc()}")
        return [_record("epochs", group, "failed", None, exc, start)]

    epoch_files = []
    for source, preprocessed in runs:
        start = time.perf_counter()
        existing = [
            f for f in existing_derivatives(source, root, desc=desc) if f.name.endswith("_epo.fif")
        ]
        if skip_existing and existing:
            records.append(_record("epochs", source, "skipped", existing[0], None, start))
            epoch_files.append(existing[0])
            continue
        try:
            import mne

            pipe = clone(pipeline)
            raw = mne.io.read_raw_fif(preprocessed, preload=True, verbose=False)
            out = pipe.fit_transform(raw)
            fname = save_derivative(
                out, source, root, pipeline=pipe, desc=desc, overwrite=not skip_existing
            )
            records.append(_record("epochs", source, "ok", fname, None, start))
            epoch_files.append(fname)
        except Exception as exc:
            logger.error(f"Epoching {source.basename} failed:\n{traceback.format_exc()}")
            records.append(_record("epochs", source, "failed", None, exc, start))

    if combine and epoch_files:
        records.append(_combine_group(group, runs, epoch_files, root, desc, skip_existing))
    return records


def _combine_group(
    group: str,
    runs: list[tuple[BIDSPath, str]],
    epoch_files: list[Path],
    root: Path,
    desc: str,
    skip_existing: bool,
) -> dict[str, Any]:
    import mne

    from .epochs import combine
    from .io import save_derivative
    from .io.derivatives import existing_derivatives

    start = time.perf_counter()
    target = runs[0][0].copy().update(run=None, split=None, check=False)
    existing = [
        f for f in existing_derivatives(target, root, desc=desc) if f.name.endswith("_epo.fif")
    ]
    if skip_existing and existing:
        return _record("combine", group, "skipped", existing[0], None, start)
    try:
        if len(runs) != len(epoch_files):
            raise RuntimeError(
                f"{len(runs) - len(epoch_files)} of {len(runs)} runs have no epochs; "
                "fix them first, or exclude them from the dataset."
            )
        combined = combine([mne.read_epochs(f, verbose=False) for f in epoch_files])
        fname = save_derivative(
            combined,
            target,
            root,
            desc=desc,
            overwrite=not skip_existing,
            sources=[str(f) for f in epoch_files],
            metadata={"combined_runs": [str(s.run) for s, _ in runs]},
        )
        return _record("combine", group, "ok", fname, None, start)
    except Exception as exc:
        logger.error(f"Combining {group} failed:\n{traceback.format_exc()}")
        return _record("combine", group, "failed", None, exc, start)


def _with_destination(epochs: Pipeline, destination: Any) -> Pipeline:
    """A copy of ``epochs`` whose head alignment maps to ``destination``."""
    from .steps import HeadAlign

    pipe: Pipeline = clone(epochs)
    for _, step in pipe.steps:
        if isinstance(step, HeadAlign):
            step.set_params(destination=destination)
            return pipe
    first_epochs = next(
        (name for name, step in pipe.steps if step.returns is not None), pipe.steps[0][0]
    )
    return pipe.insert_before(first_epochs, "align", HeadAlign(destination))


def _record(
    stage: str, source: Any, status: str, output: Any, error: Exception | None, start: float
) -> dict[str, Any]:
    if isinstance(source, BIDSPath):
        source = str(source.fpath)
    return {
        "stage": stage,
        "source": str(source),
        "status": status,
        "output": None if output is None else str(output),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "duration_s": round(time.perf_counter() - start, 3),
    }
