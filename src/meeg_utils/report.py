"""HTML reports of quality checks and processing pipelines."""

from __future__ import annotations

import html
import json
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
