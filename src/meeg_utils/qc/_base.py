"""Base classes of the acquisition-quality checks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

from mne.io import BaseRaw
from mne_bids import BIDSPath
from sklearn.base import BaseEstimator

from ..core.plotting import PlotMixin
from ..io import detect_system, get_datatypes, read

#: Levels of a finding, from best to worst.
LEVELS = ("ok", "warn", "fail")


@dataclass(frozen=True)
class Finding:
    """One evaluated quality criterion.

    Parameters
    ----------
    check : str
        Name of the check that produced it.
    metric : str
        The metric that was evaluated.
    value : Any
        Its value.
    level : {"ok", "warn", "fail"}
        Verdict.
    message : str
        Human-readable explanation, including the threshold that applied.
    """

    check: str
    metric: str
    value: Any
    level: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a plain dict."""
        return asdict(self)


class Check(PlotMixin, BaseEstimator):
    """Base class of acquisition-quality checks.

    A check answers one question about how well a recording was acquired
    and never modifies the data. Like a step, it takes all parameters in
    ``__init__`` (thresholds included, with their source documented); then
    :meth:`compute` measures the recording and stores

    - ``metrics_``: the measurements (plain, JSON-serializable values);
    - ``findings_``: one :class:`Finding` per criterion evaluated, with a
      verdict ``"ok"``, ``"warn"`` or ``"fail"``;

    and :meth:`plot` draws the figures listed in ``plot_kinds``.

    Subclasses set ``name``, ``modalities`` (and optionally ``systems``),
    implement :meth:`_compute`, and may override :meth:`not_applicable` for
    extra requirements (e.g. an EOG channel).
    """

    #: Key of the check in reports.
    name: ClassVar[str] = ""
    #: Data types the check needs at least one of ("meg", "eeg").
    modalities: ClassVar[frozenset[str]] = frozenset({"meg", "eeg"})
    #: Acquisition systems the check applies to; ``None`` means all.
    systems: ClassVar[frozenset[str] | None] = None
    _fitted_attr: ClassVar[str] = "metrics_"

    def not_applicable(self, raw: BaseRaw) -> str | None:
        """Return why the check cannot run on ``raw``, or ``None`` if it can.

        Parameters
        ----------
        raw : Raw
            The recording.

        Returns
        -------
        str | None
            The reason, or ``None``.
        """
        datatypes = get_datatypes(raw)
        if not datatypes & self.modalities:
            return f"needs {' or '.join(sorted(self.modalities)).upper()} channels"
        if self.systems is not None:
            system = detect_system(raw)
            if system not in self.systems:
                return f"applies to {sorted(self.systems)} data, not {system!r}"
        return None

    def compute(self, raw: BaseRaw) -> Self:
        """Measure the recording.

        Parameters
        ----------
        raw : Raw
            The preloaded recording; it is not modified.

        Returns
        -------
        self : Check
            The check, with ``metrics_`` and ``findings_`` set.

        Raises
        ------
        ValueError
            If the check does not apply to the recording.
        """
        if not isinstance(raw, BaseRaw):
            raise TypeError(
                f"{type(self).__name__} needs a Raw recording, got {type(raw).__name__}."
            )
        if not raw.preload:
            raise ValueError(f"{type(self).__name__} needs preloaded data; call raw.load_data().")
        reason = self.not_applicable(raw)
        if reason is not None:
            raise ValueError(f"{type(self).__name__} {reason}.")
        self.system_ = detect_system(raw)
        self.metrics_: dict[str, Any] = {}
        self.findings_: list[Finding] = []
        self._compute(raw)
        return self

    @property
    def level(self) -> str:
        """The worst verdict of the findings (``"ok"`` if there are none)."""
        levels = [f.level for f in getattr(self, "findings_", [])]
        return max(levels, key=LEVELS.index, default="ok")

    def _compute(self, raw: BaseRaw) -> None:
        raise NotImplementedError

    def _judge(
        self,
        metric: str,
        value: float,
        *,
        warn: float | None = None,
        fail: float | None = None,
        below: bool = False,
        what: str,
        unit: str = "",
        detail: str = "",
    ) -> str:
        """Record a finding comparing ``value`` with the thresholds.

        By default larger values are worse (``value > threshold``); with
        ``below=True`` smaller values are worse (``value < threshold``).
        """

        def exceeds(threshold: float | None) -> bool:
            if threshold is None:
                return False
            return value < threshold if below else value > threshold

        level = "fail" if exceeds(fail) else "warn" if exceeds(warn) else "ok"
        threshold = fail if level == "fail" else warn if warn is not None else fail
        relation = ("<" if below else ">") if level != "ok" else ("≥" if below else "≤")
        shown = f"{value:g}" if isinstance(value, int | float) else str(value)
        message = f"{what}: {shown}{unit}"
        if threshold is not None:
            message += f" ({relation} {threshold:g}{unit})"
        if detail and level != "ok":
            message += f"; {detail}"
        self.findings_.append(Finding(self.name, metric, value, level, message))
        return level

    def _note(self, metric: str, value: Any, message: str, level: str = "ok") -> None:
        """Record a finding without a numeric threshold."""
        self.findings_.append(Finding(self.name, metric, value, level, message))


class QCReport:
    """The results of several checks on one recording.

    Parameters
    ----------
    checks : dict
        Check name mapped to the computed :class:`Check`.
    skipped : dict
        Check name mapped to why it did not run.
    source : str | None
        The recording.
    """

    def __init__(
        self, checks: dict[str, Check], skipped: dict[str, str], source: str | None = None
    ) -> None:
        self.checks = checks
        self.skipped = skipped
        self.source = source

    @property
    def findings(self) -> list[Finding]:
        """All findings of all checks."""
        return [f for check in self.checks.values() for f in check.findings_]

    @property
    def flags(self) -> list[Finding]:
        """The findings that are not ``"ok"``, worst first."""
        flagged = [f for f in self.findings if f.level != "ok"]
        return sorted(flagged, key=lambda f: -LEVELS.index(f.level))

    @property
    def level(self) -> str:
        """The worst verdict over all checks."""
        return max((c.level for c in self.checks.values()), key=LEVELS.index, default="ok")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (metrics, findings, skipped checks)."""
        return {
            "source": self.source,
            "level": self.level,
            "checks": {
                name: {
                    "level": check.level,
                    "params": check.get_params(),
                    "metrics": check.metrics_,
                    "findings": [f.to_dict() for f in check.findings_],
                }
                for name, check in self.checks.items()
            },
            "skipped": dict(self.skipped),
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Return one row per finding, e.g. for ``pandas.DataFrame(report.to_records())``."""
        return [{"source": self.source, **f.to_dict()} for f in self.findings]

    def plot(self, *, inst: BaseRaw | None = None) -> dict[str, dict[str, Any]]:
        """Draw the figures of every check.

        Parameters
        ----------
        inst : Raw | None
            Data for the figures that need it.

        Returns
        -------
        dict
            Check name mapped to ``{kind: Figure}``.
        """
        return {name: check.plot(inst=inst) for name, check in self.checks.items()}

    def __repr__(self) -> str:
        lines = [f"<QCReport {self.source or ''} level={self.level!r}>"]
        for name, check in self.checks.items():
            lines.append(f"  {check.level:<4} {name}")
            lines.extend(f"         - {f.message}" for f in check.findings_ if f.level != "ok")
        lines.extend(f"  skip {name}: {reason}" for name, reason in self.skipped.items())
        return "\n".join(lines)


def inspect(
    raw: BaseRaw | str | Path | BIDSPath,
    checks: Sequence[Check] | None = None,
) -> QCReport:
    """Run acquisition-quality checks on a recording.

    Parameters
    ----------
    raw : Raw | str | Path | BIDSPath
        The recording (paths are read with :func:`meeg_utils.io.read`).
    checks : sequence of Check | None
        The checks to run. ``None`` runs every built-in check with its
        default parameters that applies to the recording (see
        :data:`meeg_utils.qc.DEFAULT_CHECKS`); the others are listed in
        ``report.skipped`` with the reason.

    Returns
    -------
    QCReport
        The results.
    """
    from . import DEFAULT_CHECKS

    source = None
    if not isinstance(raw, BaseRaw):
        source = str(raw.fpath if isinstance(raw, BIDSPath) else raw)
        raw = read(raw)
    elif raw.filenames and raw.filenames[0] is not None:
        source = str(raw.filenames[0])

    to_run: Iterable[Check] = [cls() for cls in DEFAULT_CHECKS] if checks is None else checks
    done: dict[str, Check] = {}
    skipped: dict[str, str] = {}
    for check in to_run:
        reason = check.not_applicable(raw)
        if reason is not None:
            skipped[check.name] = reason
            continue
        if check.name in done:
            raise ValueError(f"Two checks are named {check.name!r}.")
        done[check.name] = check.compute(raw)
    return QCReport(done, skipped, source)
