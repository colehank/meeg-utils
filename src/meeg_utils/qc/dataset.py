"""Quality checks over a whole dataset, with outlying recordings flagged."""

from __future__ import annotations

import csv
import traceback
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from loguru import logger
from mne_bids import BIDSPath
from sklearn.base import clone

from ._base import LEVELS, Check, QCReport, inspect

#: Modified z-score above which a recording is an outlier for a metric
#: (Iglewicz & Hoaglin, 1993, "How to detect and handle outliers").
OUTLIER_Z = 3.5


class DatasetQC:
    """Quality-check results of many recordings.

    Parameters
    ----------
    reports : dict
        Recording mapped to its :class:`QCReport`.
    errors : dict
        Recording mapped to the error that stopped its checks.
    min_recordings : int
        Outliers are only computed for metrics measured on at least this
        many recordings.
    """

    plot_kinds: ClassVar[dict[str, bool]] = {"levels": False}

    def __init__(
        self, reports: dict[str, QCReport], errors: dict[str, str], *, min_recordings: int = 5
    ) -> None:
        self.reports = reports
        self.errors = errors
        self.min_recordings = min_recordings
        self.table = self._table()
        self.outliers = self._outliers()

    def _table(self) -> dict[str, dict[str, float]]:
        """Recording -> {"check.metric": value} for scalar numeric metrics."""
        table: dict[str, dict[str, float]] = {}
        for source, report in self.reports.items():
            row: dict[str, float] = {}
            for name, check in report.checks.items():
                row.update(_scalars(check.metrics_, f"{name}."))
            table[source] = row
        return table

    def _outliers(self) -> list[dict[str, Any]]:
        return find_outliers(self.table, min_recordings=self.min_recordings)

    @property
    def levels(self) -> dict[str, dict[str, str]]:
        """Recording mapped to {check: level}."""
        return {
            source: {name: check.level for name, check in report.checks.items()}
            for source, report in self.reports.items()
        }

    def to_records(self) -> list[dict[str, Any]]:
        """One row per recording: its overall level and every scalar metric."""
        rows = []
        for source, report in self.reports.items():
            outlying = sorted(o["metric"] for o in self.outliers if o["source"] == source)
            rows.append(
                {
                    "source": source,
                    "level": report.level,
                    "n_flags": len(report.flags),
                    "outlier_metrics": ";".join(outlying),
                    **self.table[source],
                }
            )
        rows.extend({"source": s, "level": "error", "error": e} for s, e in self.errors.items())
        return rows

    def to_csv(self, fname: str | Path) -> Path:
        """Write :meth:`to_records` as a CSV file.

        Parameters
        ----------
        fname : str | Path
            Output file.

        Returns
        -------
        Path
            The file written.
        """
        rows = self.to_records()
        columns: list[str] = []
        for row in rows:
            columns.extend(c for c in row if c not in columns)
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        with fname.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        return fname

    def plot(self, kind: str | None = None) -> dict[str, Any]:
        """Draw the dataset overview: the level of every check on every recording.

        Parameters
        ----------
        kind : str | None
            ``"levels"`` or ``None`` (all figures).

        Returns
        -------
        dict
            Kind mapped to its Figure.
        """
        if kind not in (None, "levels"):
            raise ValueError(f"DatasetQC has no {kind!r} plot; available: ['levels'].")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        levels = self.levels
        sources = list(levels)
        checks = sorted({c for row in levels.values() for c in row})
        grid = np.full((len(sources), len(checks)), np.nan)
        for i, source in enumerate(sources):
            for j, check in enumerate(checks):
                if check in levels[source]:
                    grid[i, j] = LEVELS.index(levels[source][check])
        outlying = {(o["source"], o["metric"].split(".")[0]) for o in self.outliers}
        with plt.ioff():
            fig, ax = plt.subplots(
                figsize=(max(6.0, 1.0 + 0.7 * len(checks)), max(3.0, 1.5 + 0.28 * len(sources))),
                layout="constrained",
            )
            cmap = ListedColormap(["#b0bec5", "#4caf50", "#ffb300", "#e53935"])
            ax.imshow(
                grid, cmap=cmap, vmin=0, vmax=len(LEVELS) - 1, aspect="auto",
                interpolation="nearest",
            )  # fmt: skip
            for i, source in enumerate(sources):
                for j, check in enumerate(checks):
                    if (source, check) in outlying:
                        ax.text(j, i, "●", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(checks)), checks, rotation=45, ha="right")
            ax.set_yticks(range(len(sources)), [Path(s).name for s in sources], fontsize=7)
            ax.set_title(
                "QC per recording (grey measured only, green ok, amber warn, red fail; ● outlier)"
            )
        return {"levels": fig}

    def __repr__(self) -> str:
        counts = dict.fromkeys(LEVELS, 0)
        for report in self.reports.values():
            counts[report.level] += 1
        summary = ", ".join(f"{n} {level}" for level, n in counts.items())
        return (
            f"<DatasetQC {len(self.reports)} recordings ({summary}), "
            f"{len(self.outliers)} outlying metrics, {len(self.errors)} errors>"
        )


def inspect_dataset(
    sources: Sequence[str | Path | BIDSPath] | str | Path,
    checks: Sequence[Check] | None = None,
    *,
    n_jobs: int = 1,
    min_recordings: int = 5,
) -> DatasetQC:
    """Run the quality checks on every recording of a dataset.

    Parameters
    ----------
    sources : sequence of str | Path | BIDSPath, or str | Path
        The recordings, or the root of a BIDS dataset (every MEG and EEG
        recording in it is checked; derivatives are ignored).
    checks : sequence of Check | None
        Checks to run on each recording (cloned per recording); ``None``
        runs the default applicable checks, as :func:`inspect`.
    n_jobs : int
        Recordings checked in parallel (joblib).
    min_recordings : int
        Minimum number of recordings for outlier detection on a metric.

    Returns
    -------
    DatasetQC
        Per-recording reports, a metrics table and the outlying recordings.
        A recording whose checks raise is listed in ``errors``; the others
        are still checked.
    """
    if isinstance(sources, str | Path):
        items: list[str | Path | BIDSPath] = (
            list(find_recordings(sources)) if Path(sources).is_dir() else [sources]
        )
    else:
        items = list(sources)
    if not items:
        raise ValueError("No recordings to check.")
    if n_jobs == 1:
        results = [_inspect_one(s, checks) for s in items]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(delayed(_inspect_one)(s, checks) for s in items)
    reports = {source: r for source, r, _ in results if r is not None}
    errors = {source: e for source, _, e in results if e is not None}
    if errors:
        warnings.warn(f"QC failed for {len(errors)} of {len(items)} recordings.", stacklevel=2)
    return DatasetQC(reports, errors, min_recordings=min_recordings)


def find_recordings(root: str | Path) -> list[BIDSPath]:
    """Return every MEG and EEG recording of a BIDS dataset.

    Parameters
    ----------
    root : str | Path
        Root of the BIDS dataset.

    Returns
    -------
    list of BIDSPath
        The recordings, sorted.
    """
    from mne_bids import find_matching_paths
    from mne_bids.config import reader

    paths = find_matching_paths(
        root, suffixes=["meg", "eeg"], extensions=list(reader), ignore_json=True
    )
    return sorted(
        (p for p in paths if p.split is None or p.split == "01"),
        key=lambda p: str(p.fpath),
    )


def find_outliers(
    table: dict[str, dict[str, float]], *, min_recordings: int = 5
) -> list[dict[str, Any]]:
    """Recordings whose value of a metric is an outlier (modified z > :data:`OUTLIER_Z`).

    Parameters
    ----------
    table : dict
        Recording mapped to ``{metric: value}``.
    min_recordings : int
        Metrics measured on fewer recordings are skipped.

    Returns
    -------
    list of dict
        ``source``, ``metric``, ``value`` and ``z`` of each outlier.

    Notes
    -----
    When more than half of the values are identical (e.g. most runs have no
    bad segment), the median absolute deviation is zero and the mean
    absolute deviation is used instead (``z = (x - median) / (1.253314 *
    MeanAD)``), the usual fallback of the modified z-score.
    """
    columns = sorted({c for row in table.values() for c in row})
    found = []
    for column in columns:
        sources = [s for s, row in table.items() if column in row]
        if len(sources) < min_recordings:
            continue
        x = np.array([table[s][column] for s in sources], dtype=float)
        deviation = np.abs(x - np.median(x))
        mad = np.median(deviation)
        if mad > 0:
            z = 0.6745 * (x - np.median(x)) / mad
        elif deviation.mean() > 0:  # most values identical (e.g. zero): mean absolute deviation
            z = (x - np.median(x)) / (1.253314 * deviation.mean())
        else:
            continue
        for source, value, zi in zip(sources, x, z, strict=True):
            if abs(zi) > OUTLIER_Z:
                found.append(
                    {
                        "source": source,
                        "metric": column,
                        "value": float(value),
                        "z": round(float(zi), 2),
                    }
                )
    return found


def _scalars(metrics: dict[str, Any], prefix: str, max_keys: int = 10) -> dict[str, float]:
    """Numeric scalars of nested metrics; large dicts (per-channel values) are skipped."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            if len(value) <= max_keys:
                out.update(_scalars(value, f"{prefix}{key}.", max_keys))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            out[f"{prefix}{key}"] = float(value)
    return out


def _inspect_one(
    source: str | Path | BIDSPath, checks: Sequence[Check] | None
) -> tuple[str, QCReport | None, str | None]:
    key = str(source.fpath if isinstance(source, BIDSPath) else source)
    try:
        report = inspect(source, None if checks is None else [clone(c) for c in checks])
    except Exception as exc:
        logger.error(f"QC failed for {key}:\n{traceback.format_exc()}")
        return key, None, f"{type(exc).__name__}: {exc}"
    return key, report, None
