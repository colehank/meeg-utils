"""HTML reports of quality checks and processing pipelines."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
from mne.io import BaseRaw

if TYPE_CHECKING:
    from .core import Pipeline
    from .qc import QCReport

_LEVEL_COLORS = {"ok": "#2e7d32", "warn": "#ef6c00", "fail": "#c62828"}


def build(
    pipeline: Pipeline | None = None,
    inst: BaseRaw | None = None,
    qc: QCReport | None = None,
    *,
    title: str = "meeg-utils report",
) -> mne.Report:
    """Build an :class:`mne.Report` with every metric and figure.

    The report starts with the acquisition-quality checks (a table of all
    findings, then each check's figures), followed by one section per
    pipeline step (its ``qc_`` metrics and figures) and the provenance.

    Parameters
    ----------
    pipeline : Pipeline | None
        A fitted pipeline.
    inst : Raw | None
        The pipeline's input; with it, the figures that need data are drawn
        too (the pipeline is replayed, so each step plots its own input).
    qc : QCReport | None
        Results of :func:`meeg_utils.qc.inspect`.
    title : str
        Report title.

    Returns
    -------
    mne.Report
        The report; save it with ``report.save("report.html")``.
    """
    import matplotlib.pyplot as plt

    if pipeline is None and qc is None:
        raise ValueError("Nothing to report: pass a pipeline and/or a QC report.")
    report = mne.Report(title=title, verbose=False)

    if qc is not None:
        section = "Quality checks"
        report.add_html(_findings_table(qc), title="Findings", section=section, tags=("qc",))
        for name, figures in qc.plot(inst=inst).items():
            for kind, fig in figures.items():
                report.add_figure(fig, title=f"{name}: {kind}", section=section, tags=("qc", name))
                plt.close("all")

    if pipeline is not None:
        if not hasattr(pipeline, "provenance_"):
            raise ValueError("The pipeline is not fitted; fit it before building a report.")
        figures = pipeline.plot(inst=inst)
        for name, step in pipeline.steps:
            report.add_html(
                _dict_table(step.qc_), title=f"{name}: metrics", section=name, tags=("step", name)
            )
            for kind, fig in figures.get(name, {}).items():
                report.add_figure(fig, title=f"{name}: {kind}", section=name, tags=("step", name))
            plt.close("all")
        report.add_html(
            f"<pre>{html.escape(json.dumps(pipeline.provenance_, indent=2, default=str))}</pre>",
            title="Provenance",
            section="Provenance",
            tags=("provenance",),
        )
    return report


def _findings_table(qc: QCReport) -> str:
    rows = []
    for name, check in qc.checks.items():
        for f in check.findings_:
            color = _LEVEL_COLORS[f.level]
            rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f'<td style="color:{color};font-weight:bold">{f.level}</td>'
                f"<td>{html.escape(f.message)}</td></tr>"
            )
    for name, reason in qc.skipped.items():
        rows.append(
            f'<tr><td>{html.escape(name)}</td><td style="color:#757575">skipped</td>'
            f"<td>{html.escape(reason)}</td></tr>"
        )
    level = qc.level
    return (
        f'<p>Overall: <b style="color:{_LEVEL_COLORS[level]}">{level}</b>'
        f"{' — ' + html.escape(qc.source) if qc.source else ''}</p>"
        '<table class="table table-sm"><thead><tr><th>Check</th><th>Level</th>'
        "<th>Finding</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _dict_table(metrics: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(_short(value))}</td></tr>"
        for key, value in _flatten(metrics).items()
    )
    if not rows:
        return "<p>No metrics.</p>"
    return f'<table class="table table-sm"><tbody>{rows}</tbody></table>'


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value and len(value) <= 30:
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def _short(value: Any, limit: int = 300) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + " …"


# ----------------------------------------------------------------------
# Reports of saved derivatives


def from_derivative(fname: str | Path, *, title: str | None = None) -> mne.Report:
    """Build a report of a derivative saved by :func:`meeg_utils.io.save_derivative`.

    The fitted pipeline is not saved, so its figures cannot be redrawn;
    the report shows what was saved: the pipeline's steps and parameters,
    each step's QC metrics, warnings, the provenance, and the data
    themselves (MNE's views of Raw, Epochs or Evoked data, with spectra).

    Parameters
    ----------
    fname : str | Path
        The derivative ``.fif`` file (its JSON sidecar is read alongside).
    title : str | None
        Report title (default: the file name).

    Returns
    -------
    mne.Report
        The report.
    """
    import matplotlib.pyplot as plt

    fname = Path(fname)
    sidecar = _read_sidecar(fname)
    report = mne.Report(title=title or fname.name, verbose=False)
    meta = sidecar.get("MeegUtils", {})
    sources = "".join(f"<li>{html.escape(str(s))}</li>" for s in sidecar.get("Sources", []))
    report.add_html(
        f"<p>{html.escape(sidecar.get('Description', ''))}</p><p>Sources:</p><ul>{sources}</ul>",
        title="Derivative",
        section="Derivative",
    )
    steps = meta.get("provenance", {}).get("steps") or [
        {"name": s["name"], "class": s["class"], "params": s["params"]}
        for s in meta.get("pipeline", {}).get("steps", [])
    ]
    if steps:
        report.add_html(_steps_table(steps), title="Pipeline", section="Pipeline")
    for name, metrics in meta.get("qc", {}).items():
        report.add_html(_dict_table(metrics), title=f"{name}: metrics", section=name)
    extra = {k: v for k, v in meta.items() if k not in ("pipeline", "provenance", "qc")}
    if extra:
        report.add_html(_dict_table(extra), title="Details", section="Derivative")

    suffix = fname.stem.rsplit("_", 1)[-1]
    with mne.use_log_level("warning"):
        if suffix == "epo":
            report.add_epochs(mne.read_epochs(fname), title="Epochs", psd=True)
        elif suffix == "ave":
            report.add_evokeds(mne.read_evokeds(fname))
        else:
            report.add_raw(mne.io.read_raw_fif(fname), title="Data", psd=True, butterfly=False)
    plt.close("all")
    if "provenance" in meta:
        report.add_html(
            f"<pre>{html.escape(json.dumps(meta['provenance'], indent=2, default=str))}</pre>",
            title="Provenance",
            section="Provenance",
        )
    return report


class DerivativesSummary:
    """QC metrics of every derivative in a folder, with outlying files flagged.

    Built by :func:`summarize`. The metrics are the scalar values of each
    step's ``qc_`` (e.g. ``bads.eeg.n_bads``, ``ica.n_excluded``,
    ``autoreject.fraction_dropped``) read from the sidecars; a file is an
    outlier for a metric when its modified z-score exceeds 3.5 (as
    :func:`meeg_utils.qc.inspect_dataset`).

    Attributes
    ----------
    table : dict
        File (relative to the root) mapped to ``{metric: value}``.
    warnings : dict
        File mapped to the warnings its pipeline raised.
    outliers : list of dict
        ``source``, ``metric``, ``value`` and ``z`` of each outlier.
    """

    def __init__(
        self,
        table: dict[str, dict[str, float]],
        warnings_: dict[str, list[str]],
        *,
        min_files: int = 5,
    ) -> None:
        from .qc.dataset import find_outliers

        self.table = table
        self.warnings = warnings_
        self.outliers = find_outliers(table, min_recordings=min_files)

    def to_records(self) -> list[dict[str, Any]]:
        """One row per file: its metrics, warning count and outlying metrics."""
        rows = []
        for source, metrics in self.table.items():
            outlying = sorted(o["metric"] for o in self.outliers if o["source"] == source)
            rows.append(
                {
                    "file": source,
                    "n_warnings": len(self.warnings.get(source, [])),
                    "outlier_metrics": ";".join(outlying),
                    **metrics,
                }
            )
        return rows

    def to_csv(self, fname: str | Path) -> Path:
        """Write :meth:`to_records` as CSV.

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
        rows = self.to_records()
        fields = ["file", "n_warnings", "outlier_metrics"]
        fields += sorted({k for r in rows for k in r} - set(fields))
        with fname.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return fname

    def report(self, *, title: str = "Preprocessing summary", max_metrics: int = 24) -> mne.Report:
        """Build an HTML report: outliers, warnings and the distribution of each metric.

        Parameters
        ----------
        title : str
            Report title.
        max_metrics : int
            At most this many metric distributions are drawn (those that
            vary across files, outlying ones first).

        Returns
        -------
        mne.Report
            The report.
        """
        import matplotlib.pyplot as plt

        report = mne.Report(title=title, verbose=False)
        rows = "".join(
            f"<tr><td>{html.escape(o['source'])}</td><td>{html.escape(o['metric'])}</td>"
            f"<td>{o['value']:g}</td><td>{o['z']:g}</td></tr>"
            for o in self.outliers
        )
        report.add_html(
            f"<p>{len(self.table)} files.</p>"
            + (
                '<table class="table table-sm"><thead><tr><th>File</th><th>Metric</th>'
                f"<th>Value</th><th>z</th></tr></thead><tbody>{rows}</tbody></table>"
                if rows
                else "<p>No outlying files.</p>"
            ),
            title="Outlying files",
            section="Summary",
        )
        warned = "".join(
            f"<tr><td>{html.escape(f)}</td><td>{html.escape('; '.join(w))}</td></tr>"
            for f, w in self.warnings.items()
            if w
        )
        if warned:
            report.add_html(
                f'<table class="table table-sm"><tbody>{warned}</tbody></table>',
                title="Warnings",
                section="Summary",
            )
        for metric in self._metrics_to_plot(max_metrics):
            report.add_figure(self._strip(metric), title=metric, section="Metrics")
            plt.close("all")
        return report

    def _metrics_to_plot(self, limit: int) -> list[str]:
        columns = sorted({c for row in self.table.values() for c in row})
        varying = [
            c for c in columns if len({row[c] for row in self.table.values() if c in row}) > 1
        ]
        outlying = {o["metric"] for o in self.outliers}
        varying.sort(key=lambda c: (c not in outlying, c))
        return varying[:limit]

    def _strip(self, metric: str) -> Any:
        import matplotlib.pyplot as plt
        import numpy as np

        files = [f for f, row in self.table.items() if metric in row]
        values = np.array([self.table[f][metric] for f in files])
        flagged = {o["source"] for o in self.outliers if o["metric"] == metric}
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(6, 1.8), layout="constrained")
        colors = ["C3" if f in flagged else "C0" for f in files]
        ax.scatter(values, rng.uniform(-0.2, 0.2, len(values)), c=colors, s=18)
        for f, v in zip(files, values, strict=True):
            if f in flagged:
                ax.annotate(Path(f).name, (v, 0.2), fontsize=7, rotation=20, color="C3")
        ax.set(yticks=[], ylim=(-0.5, 0.8), xlabel=metric)
        return fig


def summarize(
    root: str | Path, *, desc: str | None = None, min_files: int = 5
) -> DerivativesSummary:
    """Collect the QC metrics of every derivative under ``root``.

    Parameters
    ----------
    root : str | Path
        Derivatives folder (searched recursively for meeg-utils sidecars).
    desc : str | None
        Only files with this BIDS ``desc`` (e.g. ``"preproc"``).
    min_files : int
        Outliers are only computed for metrics found in at least this many files.

    Returns
    -------
    DerivativesSummary
        The table, warnings and outliers.
    """
    from .qc.dataset import _scalars

    root = Path(root)
    table: dict[str, dict[str, float]] = {}
    warned: dict[str, list[str]] = {}
    for sidecar_path in sorted(root.rglob("*.json")):
        if desc is not None and f"_desc-{desc}_" not in sidecar_path.name:
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta = sidecar.get("MeegUtils") if isinstance(sidecar, dict) else None
        if not meta or "qc" not in meta:
            continue
        key = str(sidecar_path.with_suffix(".fif").relative_to(root))
        row: dict[str, float] = {}
        for step, metrics in meta["qc"].items():
            if isinstance(metrics, dict):
                row.update(_scalars(metrics, f"{step}."))
        table[key] = row
        warned[key] = [
            w
            for step in meta.get("provenance", {}).get("steps", [])
            for w in step.get("warnings", [])
        ]
    if not table:
        raise ValueError(f"No meeg-utils derivatives with QC metrics under {root}.")
    return DerivativesSummary(table, warned, min_files=min_files)


def _read_sidecar(fname: Path) -> dict[str, Any]:
    sidecar = fname.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"{sidecar} not found: is {fname.name} a meeg-utils derivative?")
    return dict(json.loads(sidecar.read_text(encoding="utf-8")))


def _steps_table(steps: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(s.get('name')))}</td>"
        f"<td>{html.escape(str(s.get('class', '')).rsplit(':', 1)[-1])}</td>"
        f"<td><code>{html.escape(_short(s.get('params', {}), 500))}</code></td>"
        f"<td>{s.get('duration_s', '')}</td>"
        f"<td>{html.escape('; '.join(s.get('warnings', [])))}</td></tr>"
        for s in steps
    )
    return (
        '<table class="table table-sm"><thead><tr><th>Step</th><th>Class</th><th>Parameters</th>'
        f"<th>Time (s)</th><th>Warnings</th></tr></thead><tbody>{rows}</tbody></table>"
    )
