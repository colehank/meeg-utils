"""Bad-channel detection, interpolation and referencing."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst
from ..io import get_datatypes
from ._plotting import plot_sensor_groups
from ._utils import head_origin

#: PREP criteria shown in the "scores" figure: diagnostic, label, threshold (PyPREP defaults).
_PREP_SCORES = {
    "deviation": ("bad_by_deviation", "robust_channel_deviations", "robust z (amplitude)", 5.0),
    "hf_noise": ("bad_by_hf_noise", "hf_noise_zscores", "robust z (HF noise)", 5.0),
    "psd": ("bad_by_psd", "psd_zscore", "z (PSD)", 3.0),
    "correlation": ("bad_by_correlation", "bad_window_fractions", "fraction of bad windows", 0.01),
    "ransac": ("bad_by_ransac", "bad_window_fractions", "fraction of bad windows", 0.4),
}

BAD_CHANNEL_METHODS = ("auto", "prep", "maxwell")
#: Time windows above the limit that make a channel noisy (MNE's default ``min_count``).
_MAXWELL_MIN_COUNT = 5


@contextlib.contextmanager
def _pyprep_psd(enabled: bool) -> Iterator[None]:
    """Run PyPREP with or without its PSD criterion.

    ``NoisyChannels.find_all_bads`` always runs ``find_bad_by_PSD`` (unless
    ``matlab_strict``, which also changes the other criteria), and the robust
    reference calls it internally, so the criterion is switched off on the
    class for the duration of the detection.
    """
    from pyprep.find_noisy_channels import NoisyChannels

    if enabled:
        yield
        return
    original = NoisyChannels.find_bad_by_PSD
    NoisyChannels.find_bad_by_PSD = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        NoisyChannels.find_bad_by_PSD = original


_MAXWELL_SYSTEMS = frozenset({"neuromag", "ctf"})


class BadChannels(Step):
    """Detect bad channels and mark them in ``info["bads"]``.

    The data themselves are not changed; follow with :class:`Interpolate`
    to repair the channels. Channels already marked bad stay bad.

    Parameters
    ----------
    method : {"auto", "prep", "maxwell"}
        - ``"prep"``: PREP noisy-channel detection for EEG (Bigdely-Shamlo et
          al., 2015) via PyPREP, running the PREP criteria: NaN/flat,
          deviation, high-frequency noise, correlation, low SNR and,
          optionally, RANSAC.
        - ``"maxwell"``: :func:`mne.preprocessing.find_bad_channels_maxwell`
          for MEG (Neuromag; CTF via a grade-0 compensated copy).
        - ``"auto"`` (default): PREP for EEG channels and Maxwell for MEG
          channels of Neuromag/CTF systems; MEG from other systems raises,
          because no validated default exists for them.
    ransac : bool
        PREP: also run RANSAC, which needs electrode positions.
    psd : bool
        PREP: also run PyPREP's band-power criterion (``find_bad_by_PSD``,
        robust z > 3 in 1-15, 15-30 or 30-45 Hz), which PyPREP adds to
        PREP ("not present in the original MATLAB PREP"). Off by default:
        it flags frontal channels, whose low-frequency power includes the
        blinks (in MNE's sample EEG, EEG 001, 004 and 007 were flagged by
        it alone), and these would then be interpolated before ICA could
        remove the blinks.
    robust_reference : bool
        PREP: detect the bad channels after PREP's robust average reference
        (pyprep's ``Reference``: iteratively re-reference to the average of
        the channels not yet found bad), as the PREP paper does. Without it,
        detection runs on the recording's own reference, which flags
        channels next to the reference electrode (their signals are small and
        poorly correlated with the others; seen in HAD-MEEG, where C1 and Cz
        record about 0.4 µV against 2-3 µV elsewhere). The data themselves
        are not re-referenced.
    random_state : int | None
        PREP: seed for RANSAC.
    reject_by_annotation : {"omit", None}
        PREP: whether to ignore ``BAD_`` annotated segments.
    origin : str | tuple
        Maxwell: head origin, ``"auto"`` fits it to the digitization (or uses
        (0, 0, 0.04) with a warning when there are no head-shape points).
    cross_talk, calibration : "auto" | str | None
        Maxwell, Neuromag only: cross-talk and fine-calibration files.
        ``"auto"`` (default) finds them in the BIDS dataset of the recording
        (``sub-*/[ses-*/]meg/*_acq-crosstalk_meg.fif`` and
        ``*_acq-calibration_meg.dat``) and raises if they are missing;
        ``None`` detects without them (less accurate).
    limit : float
        Maxwell: detection threshold.
    h_freq : float | None
        Maxwell: low-pass applied before detection.

    Attributes
    ----------
    bads_ : list of str
        Newly detected bad channels.
    qc_ : dict
        ``previous_bads`` and, per modality, ``bads``, ``n_bads``,
        ``fraction`` and ``by_criterion``; with the robust reference,
        ``eeg_before_reference`` (channels PREP flags on the original
        reference, for comparison); for MEG, ``noisy_references``
        (reference sensors Maxwell flags: reported with a warning, not
        marked bad, as they cannot be interpolated).
    scores_ : dict
        Per-channel detection scores and thresholds, per criterion.

    Notes
    -----
    Figures (:meth:`plot`): ``"sensors"`` (sensor layout with previously
    and newly marked bad channels) and ``"scores"`` (per-channel scores of
    each criterion against its threshold).
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    plot_kinds: ClassVar[dict[str, bool]] = {"sensors": False, "scores": False}

    def __init__(
        self,
        method: str = "auto",
        *,
        ransac: bool = True,
        psd: bool = False,
        robust_reference: bool = True,
        random_state: int | None = 42,
        reject_by_annotation: str | None = None,
        origin: str | tuple[float, float, float] = "auto",
        cross_talk: str | None = "auto",
        calibration: str | None = "auto",
        limit: float = 7.0,
        h_freq: float | None = 40.0,
    ) -> None:
        self.method = method
        self.ransac = ransac
        self.psd = psd
        self.robust_reference = robust_reference
        self.random_state = random_state
        self.reject_by_annotation = reject_by_annotation
        self.origin = origin
        self.cross_talk = cross_talk
        self.calibration = calibration
        self.limit = limit
        self.h_freq = h_freq

    def _fit(self, inst: BaseRaw) -> None:
        if self.method not in BAD_CHANNEL_METHODS:
            raise ValueError(f"method must be one of {BAD_CHANNEL_METHODS}, got {self.method!r}.")
        datatypes = get_datatypes(inst)
        run_prep = "eeg" in datatypes and self.method in ("auto", "prep")
        run_maxwell = "meg" in datatypes and self.method in ("auto", "maxwell")
        if self.method == "prep" and "eeg" not in datatypes:
            raise ValueError("method='prep' needs EEG channels.")
        if self.method == "maxwell" and "meg" not in datatypes:
            raise ValueError("method='maxwell' needs MEG channels.")
        if run_maxwell and self.system_ not in _MAXWELL_SYSTEMS:
            raise NotImplementedError(
                f"No validated bad-channel detection for {self.system_!r} MEG; "
                "mark bad channels manually (info['bads']) and drop this step."
            )

        self.qc_["previous_bads"] = list(inst.info["bads"])
        self.bads_: list[str] = []
        self.scores_: dict[str, dict[str, Any]] = {}
        self.info_ = inst.copy().pick(["meg", "eeg"], exclude=[]).info
        if run_prep:
            self._record("eeg", inst, *self._prep(inst))
        if run_maxwell:
            self._record("meg", inst, *self._maxwell(inst))

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        new = [ch for ch in self.bads_ if ch in inst.ch_names and ch not in inst.info["bads"]]
        inst.info["bads"] = [*inst.info["bads"], *new]
        return inst

    # ------------------------------------------------------------------

    def _prep(self, inst: BaseRaw) -> tuple[list[str], dict[str, list[str]], int]:
        eeg = inst.copy().pick("eeg", exclude=[])
        if self.ransac:
            pos = np.array([ch["loc"][:3] for ch in eeg.info["chs"]])
            if not np.isfinite(pos).all() or np.allclose(pos, 0):
                raise ValueError(
                    "PREP's RANSAC needs electrode positions: set a montage "
                    "(raw.set_montage) or use BadChannels(ransac=False)."
                )
        manual = [ch for ch in inst.info["bads"] if ch in eeg.ch_names]
        with _pyprep_psd(self.psd):
            detected, extra = self._prep_detect(eeg, manual)
        for name, (group, key, label, threshold) in _PREP_SCORES.items():
            values = extra.get(group, {}).get(key)
            if values is not None and np.ndim(values) == 1 and len(values) == len(eeg.ch_names):
                self.scores_[f"PREP {name}"] = {
                    "ch_names": list(eeg.ch_names),
                    "values": np.asarray(values, dtype=float),
                    "threshold": threshold,
                    "label": label,
                }
        by_criterion = {
            key.removeprefix("bad_by_"): sorted(map(str, chs))
            for key, chs in detected.items()
            if key not in ("bad_all", "bad_by_manual")
        }
        found = sorted({ch for chs in by_criterion.values() for ch in chs} - set(manual))
        return found, by_criterion, len(eeg.ch_names)

    def _prep_detect(self, eeg: BaseRaw, manual: list[str]) -> tuple[dict, dict]:
        """Run PyPREP's detection; return the bad channels by criterion and the diagnostics."""
        from pyprep.find_noisy_channels import NoisyChannels

        if self.robust_reference:
            from pyprep.reference import Reference as _RobustReference

            eeg.info["bads"] = manual
            reference = _RobustReference(
                eeg,
                params={"ref_chs": eeg.ch_names, "reref_chs": eeg.ch_names},
                ransac=self.ransac,
                random_state=self.random_state,
                reject_by_annotation=self.reject_by_annotation,
            )
            reference.perform_reference(interpolate_bads=False)
            detected = reference.noisy_channels_before_interpolation
            extra = reference._extra_info["interpolated"] or {}
            self.qc_["eeg_before_reference"] = sorted(
                {str(c) for k, v in reference.noisy_channels_original.items()
                 if k not in ("bad_all", "bad_by_manual") for c in v}
            )  # fmt: skip
        else:
            eeg.info["bads"] = []
            finder = NoisyChannels(
                eeg,
                do_detrend=True,
                random_state=self.random_state,
                ransac=self.ransac,
                bad_by_manual=manual or None,
                reject_by_annotation=self.reject_by_annotation,
            )
            finder.find_all_bads(ransac=self.ransac)
            detected = finder.get_bads(as_dict=True)
            extra = finder._extra_info
        return detected, extra

    def _maxwell(self, inst: BaseRaw) -> tuple[list[str], dict[str, list[str]], int]:
        from mne.preprocessing import find_bad_channels_maxwell

        meg = inst.copy().pick(["meg", "ref_meg"], exclude=[])
        if self.system_ == "ctf" and meg.compensation_grade != 0:
            meg.apply_gradient_compensation(0, verbose=False)  # required by Maxwell filtering
        cross_talk = _maxwell_file(self.cross_talk, inst, "cross_talk", self.system_)
        calibration = _maxwell_file(self.calibration, inst, "calibration", self.system_)
        self.qc_["maxwell_files"] = {
            "cross_talk": None if cross_talk is None else str(cross_talk),
            "calibration": None if calibration is None else str(calibration),
        }
        noisy, flat, scores = find_bad_channels_maxwell(
            meg,
            limit=self.limit,
            origin=head_origin(self.origin, inst.info),
            cross_talk=cross_talk,
            calibration=calibration,
            h_freq=self.h_freq,
            min_count=_MAXWELL_MIN_COUNT,
            return_scores=True,
            verbose=False,
        )
        # MNE marks a channel noisy when its score exceeds the limit in at least
        # min_count time windows (capped at the number of windows): show that count
        # (channels already bad have no scores)
        with np.errstate(invalid="ignore"):
            above = np.sum(scores["scores_noisy"] > scores["limits_noisy"], axis=1)
        min_count = min(_MAXWELL_MIN_COUNT, scores["scores_noisy"].shape[1])
        for ch_type in np.unique(scores["ch_types"]):
            sel = np.asarray(scores["ch_types"]) == ch_type
            self.scores_[f"Maxwell {ch_type}"] = {
                "ch_names": list(np.asarray(scores["ch_names"])[sel]),
                "values": above[sel].astype(float),
                "threshold": float(min_count),
                "label": "windows above limit",
            }
        refs = set(meg.copy().pick("ref_meg", exclude=[]).ch_names) if "ref_meg" in meg else set()
        bad_refs = sorted((set(noisy) | set(flat)) & refs)
        if bad_refs:  # cannot be interpolated; CTF compensation spreads their noise
            self.qc_["noisy_references"] = bad_refs
            warnings.warn(
                f"Reference sensors {bad_refs} look noisy or flat. They are not marked bad (they "
                "cannot be interpolated), but gradient compensation spreads their noise to every "
                "channel: inspect them, and compare compensation grades.",
                stacklevel=5,
            )
        found = sorted((set(noisy) | set(flat)) - refs)
        found = [ch for ch in found if ch not in inst.info["bads"]]
        n_meg = len(meg.copy().pick("meg", exclude=[]).ch_names)
        by = {"noisy": sorted(set(noisy) - refs), "flat": sorted(set(flat) - refs)}
        return found, by, n_meg

    def _plot_sensors(self, inst: BaseRaw | None) -> Any:
        return plot_sensor_groups(
            self.info_,
            {"previously bad": self.qc_["previous_bads"], "detected": self.bads_},
            title=f"Bad channels ({self.method})",
        )

    def _plot_scores(self, inst: BaseRaw | None) -> Any:
        import matplotlib.pyplot as plt

        if not self.scores_:
            raise ValueError("No detection scores were recorded.")
        fig, axes = plt.subplots(
            len(self.scores_),
            1,
            figsize=(10, 2.6 * len(self.scores_)),
            squeeze=False,
            layout="constrained",
        )
        for ax, (name, score) in zip(axes[:, 0], self.scores_.items(), strict=True):
            values = np.asarray(score["values"], dtype=float)
            colors = ["C3" if ch in self.bads_ else "0.6" for ch in score["ch_names"]]
            ax.bar(np.arange(len(values)), np.nan_to_num(values), color=colors, width=0.8)
            ax.axhline(score["threshold"], color="C3", ls="--", lw=1, label="threshold")
            ax.set(title=name, ylabel=score["label"], xlim=(-1, len(values)))
            if len(values) <= 64:
                ax.set_xticks(np.arange(len(values)), score["ch_names"], rotation=90, fontsize=6)
                for tick, ch in zip(ax.get_xticklabels(), score["ch_names"], strict=True):
                    if ch in self.bads_:  # e.g. flat channels, whose scores are 0 or NaN
                        tick.set(color="C3", fontweight="bold")
            else:  # too many channels for tick labels: name the detected ones
                ax.set_xticks([])
                for i, ch in enumerate(score["ch_names"]):
                    if ch in self.bads_:
                        ax.annotate(
                            ch, (i, np.nan_to_num(values[i])), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=7, color="C3",
                        )  # fmt: skip
        axes[0, 0].legend(frameon=False, fontsize="small")
        fig.suptitle("Bad-channel detection scores (detected channels in red)")
        return fig

    def _record(
        self, modality: str, inst: BaseRaw, found: list[str], by: dict[str, list[str]], n: int
    ) -> None:
        self.bads_.extend(ch for ch in found if ch not in self.bads_)
        self.qc_[modality] = {
            "bads": found,
            "n_bads": len(found),
            "fraction": len(found) / n if n else 0.0,
            "by_criterion": by,
        }


class Interpolate(Step):
    """Interpolate the channels marked bad.

    Thin wrapper of :meth:`mne.io.Raw.interpolate_bads`: spherical splines
    for EEG (needs electrode positions) and minimum-norm field mapping for
    MEG by default.

    Parameters
    ----------
    method : dict | None
        Interpolation method per channel type; ``None`` uses
        ``{"eeg": "spline", "meg": "MNE"}``.
    origin : str | tuple
        Head origin; ``"auto"`` fits it to the digitization (or uses
        (0, 0, 0.04) with a warning when there are no head-shape points).
    reset_bads : bool
        Whether to clear the interpolated channels from ``info["bads"]``.
    exclude : list of str
        Bad channels to leave as they are.

    Attributes
    ----------
    qc_ : dict
        ``interpolated`` (channel names) and ``fraction`` of channels.

    Notes
    -----
    Figures (:meth:`plot`): ``"sensors"``, the sensor layout with the
    interpolated channels highlighted.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    plot_kinds: ClassVar[dict[str, bool]] = {"sensors": False}

    def __init__(
        self,
        method: dict | None = None,
        *,
        origin: str | tuple[float, float, float] = "auto",
        reset_bads: bool = True,
        exclude: tuple[str, ...] | list[str] = (),
    ) -> None:
        self.method = method
        self.origin = origin
        self.reset_bads = reset_bads
        self.exclude = exclude

    def _transform(self, inst: Inst) -> Inst:
        bads = [ch for ch in inst.info["bads"] if ch not in self.exclude]
        self.info_ = inst.copy().pick(["meg", "eeg"], exclude=[]).info
        n_data = len(self.info_.ch_names)
        self.qc_.update(interpolated=bads, fraction=len(bads) / n_data if n_data else 0.0)
        if bads:
            inst.interpolate_bads(
                reset_bads=self.reset_bads,
                mode="accurate",
                origin=head_origin(self.origin, inst.info),
                method=self.method or {"eeg": "spline", "meg": "MNE"},
                exclude=list(self.exclude),
                verbose=False,
            )
        return inst

    def _plot_sensors(self, inst: Inst | None) -> Any:
        if not hasattr(self, "info_"):
            raise ValueError(
                "The 'sensors' plot needs the step to have transformed data: use fit_transform, "
                "or pipeline.plot(inst=data)."
            )
        return plot_sensor_groups(
            self.info_, {"interpolated": self.qc_["interpolated"]}, title="Interpolated channels"
        )


class Reference(Step):
    """Set the EEG reference and, for CTF MEG, the gradient compensation grade.

    Parameters
    ----------
    eeg : "average" | list of str | None
        EEG reference: ``"average"`` (default, excludes bad channels), a list
        of reference channels, or ``None`` to leave it unchanged.
    projection : bool
        Add the average reference as a projector instead of applying it.
    ctf_grade : int | None
        CTF only: synthetic gradiometer compensation grade to apply
        (default 3). ``None`` leaves it unchanged. Ignored for other systems,
        which have no compensation.

    Attributes
    ----------
    qc_ : dict
        ``eeg_reference`` and, for CTF data, ``ctf_grade_before`` and
        ``ctf_grade_after``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)

    def __init__(
        self,
        eeg: str | list[str] | None = "average",
        *,
        projection: bool = False,
        ctf_grade: int | None = 3,
    ) -> None:
        self.eeg = eeg
        self.projection = projection
        self.ctf_grade = ctf_grade

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        info: dict[str, Any] = {}
        if self.eeg is not None and "eeg" in get_datatypes(inst):
            ref = self.eeg if isinstance(self.eeg, str) else list(self.eeg)
            inst.set_eeg_reference(ref_channels=ref, projection=self.projection, verbose=False)
            info["eeg_reference"] = self.eeg if isinstance(self.eeg, str) else list(self.eeg)
        if self.system_ == "ctf" and self.ctf_grade is not None:
            info["ctf_grade_before"] = int(inst.compensation_grade)
            if inst.compensation_grade != self.ctf_grade:
                inst.apply_gradient_compensation(self.ctf_grade, verbose=False)
            info["ctf_grade_after"] = int(inst.compensation_grade)
        self.qc_.update(info)
        return inst


def _maxwell_file(value: Any, raw: BaseRaw, kind: str, system: str) -> Any:
    """Resolve a cross-talk / fine-calibration argument (``"auto"`` looks in BIDS)."""
    if value != "auto":
        return value
    if system != "neuromag":
        return None  # these files exist for Neuromag/MEGIN systems only
    from pathlib import Path

    from ..io.read import as_bids_path

    fname = raw.filenames[0] if raw.filenames else None
    bids_path = as_bids_path(Path(fname)) if fname is not None else None
    if bids_path is None:
        raise FileNotFoundError(
            f"{kind}='auto' looks for the file in the BIDS dataset of the recording, but "
            f"{fname} is not in a BIDS dataset; pass the file, or {kind}=None to detect "
            "bad channels without it (less accurate)."
        )
    found = (
        bids_path.meg_crosstalk_fpath if kind == "cross_talk" else bids_path.meg_calibration_fpath
    )
    if found is None:
        raise FileNotFoundError(
            f"No {kind.replace('_', '-')} file for sub-{bids_path.subject} in {bids_path.root}; "
            f"add it (mne_bids.write_meg_{'crosstalk' if kind == 'cross_talk' else 'calibration'}), "
            f"or pass {kind}=None to detect bad channels without it (less accurate)."
        )
    return found
