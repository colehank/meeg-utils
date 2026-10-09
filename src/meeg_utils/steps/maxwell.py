"""Maxwell filtering (SSS / tSSS) with optional head-movement compensation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ..core import Step
from ._plotting import plot_psd_comparison, psd_summary
from ._utils import head_origin
from .channels import _maxwell_file
from .head import _as_dev_head_t, _movement

HEAD_POS_SOURCES = ("chpi",)


class Maxwell(Step):
    """Signal space separation (SSS) and its temporal extension (tSSS).

    Wraps :func:`mne.preprocessing.maxwell_filter` (Taulu & Kajola, 2005,
    J Appl Phys 97:124905; Taulu & Simola, 2006, Phys Med Biol 51:1759) for
    Neuromag/MEGIN systems:

    - external interference is removed and bad MEG channels are
      reconstructed, so mark them first (:class:`BadChannels` with
      ``method="maxwell"``): an unmarked noisy channel spreads its noise to
      every channel;
    - with ``st_duration``, tSSS also removes interference from sources close
      to the sensors (e.g. dental work, stimulators);
    - with ``head_pos``, the data are corrected for head movement;
    - with ``destination``, the data are transformed to another head position,
      e.g. the average over runs (:func:`meeg_utils.io.average_dev_head_t`).

    Parameters
    ----------
    st_duration : float | None
        tSSS buffer length in seconds; ``None`` (default) applies SSS only.
        MaxFilter's default is 10 s. The buffer must hold an even number of
        samples (MNE's overlapping windows); the step says which duration to
        use otherwise.
    st_correlation : float
        tSSS subspace correlation limit (MaxFilter default 0.98).
    origin : "auto" | tuple of float
        Expansion origin (m) in ``coord_frame``; ``"auto"`` fits a sphere to
        the head digitization (or uses (0, 0, 0.04) with a warning when there
        are no head-shape points).
    int_order, ext_order : int
        Orders of the internal and external expansions (defaults 8 and 3,
        as MaxFilter).
    cross_talk, calibration : "auto" | str | path-like | None
        Cross-talk compensation and fine-calibration files. ``"auto"``
        (default) finds them in the BIDS dataset of the recording and raises
        if they are missing; ``None`` filters without them, which leaves
        more residual interference.
    head_pos : "chpi" | str | path-like | ndarray | None
        Head positions for movement compensation: ``"chpi"`` estimates them
        from the cHPI coils (as :class:`meeg_utils.qc.HeadMovement` does), a
        string or path reads a head-position file (MaxFilter ``.pos``), an
        array of shape (n, 10) is used as is; ``None`` (default) applies no
        movement compensation.
    destination : str | path-like | array, shape (4, 4) | Transform | None
        Head position to transform the data to (a device-to-head transform,
        or a recording whose ``dev_head_t`` is used). ``None`` keeps the
        recording's ``dev_head_t`` (with ``head_pos``: the initial position).
    coord_frame : {"head", "meg"}
        Frame of the expansion. ``"meg"`` is for data without head
        digitization (e.g. empty-room recordings); ``head_pos`` and
        ``destination`` then cannot be used.
    regularize : "in" | None
        Regularize the internal basis by information content (MaxFilter's
        default) or not.
    remove_chpi : bool
        If cHPI was on, subtract the coil signals
        (:func:`mne.chpi.filter_chpi`) after estimating head positions and
        before filtering. SSS does not remove them (the coils are inside
        the sensor array).
    bad_condition : {"error", "warning", "info", "ignore"}
        What to do when the basis is ill conditioned.
    skip_by_annotation : list of str
        Annotations whose segments are filtered separately (e.g. acquisition
        skips).

    Attributes
    ----------
    cross_talk_, calibration_ : str | None
        Files used.
    head_pos_ : ndarray | None
        Head positions used for movement compensation in the last transform
        (MNE's format, see :func:`mne.chpi.read_head_pos`; save them with
        :func:`mne.chpi.write_head_pos`). MNE adds them to the data as
        ``CHPI`` channels; this step drops those, so the channels are
        unchanged.
    qc_ : dict
        ``reconstructed`` (bad MEG channels rebuilt), ``n_basis`` (number of
        internal components kept), ``tsss``, ``chpi_removed``, ``projs_removed``
        (MEG projectors dropped, as they no longer apply), per channel type
        ``power_change_db`` (broadband power after relative to before; strongly
        negative values mean much interference was removed), with
        ``head_pos`` the head movement (``max_displacement_mm``, ...) and with
        ``destination`` the ``destination_shift_mm`` and
        ``destination_rotation_deg``.
    psd_ : dict
        Per channel type, PSDs of the good channels before and after the last
        transform.

    Notes
    -----
    Maxwell filtering lowers the rank of the MEG data to ``n_basis``
    (typically 60-80); :class:`ICA` lowers an integer ``n_components`` to
    the rank. Run it before filtering and resampling, as MaxFilter does.

    Figures (:meth:`plot`): ``"psd"`` (before and after) and, with
    ``head_pos``, ``"head_positions"`` and ``"displacement"``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    systems: ClassVar[frozenset[str] | None] = frozenset({"neuromag"})
    plot_kinds: ClassVar[dict[str, bool]] = {
        "psd": False,
        "head_positions": False,
        "displacement": False,
    }

    def __init__(
        self,
        *,
        st_duration: float | None = None,
        st_correlation: float = 0.98,
        origin: str | tuple[float, float, float] = "auto",
        int_order: int = 8,
        ext_order: int = 3,
        cross_talk: Any = "auto",
        calibration: Any = "auto",
        head_pos: Any = None,
        destination: Any = None,
        coord_frame: str = "head",
        regularize: str | None = "in",
        remove_chpi: bool = True,
        bad_condition: str = "error",
        skip_by_annotation: tuple[str, ...] | list[str] = ("edge", "bad_acq_skip"),
    ) -> None:
        self.st_duration = st_duration
        self.st_correlation = st_correlation
        self.origin = origin
        self.int_order = int_order
        self.ext_order = ext_order
        self.cross_talk = cross_talk
        self.calibration = calibration
        self.head_pos = head_pos
        self.destination = destination
        self.coord_frame = coord_frame
        self.regularize = regularize
        self.remove_chpi = remove_chpi
        self.bad_condition = bad_condition
        self.skip_by_annotation = skip_by_annotation

    def _fit(self, inst: BaseRaw) -> None:
        if self.coord_frame not in ("head", "meg"):
            raise ValueError(f"coord_frame must be 'head' or 'meg', got {self.coord_frame!r}.")
        if self.coord_frame == "meg" and (
            self.head_pos is not None or self.destination is not None
        ):
            raise ValueError("head_pos and destination need coord_frame='head'.")
        if isinstance(self.head_pos, str) and not Path(self.head_pos).suffix:
            if self.head_pos not in HEAD_POS_SOURCES:
                raise ValueError(
                    f"head_pos must be 'chpi', a head-position file or an array, "
                    f"got {self.head_pos!r}."
                )
        if self.coord_frame == "head" and inst.info["dev_head_t"] is None:
            raise ValueError(
                "The recording has no device-to-head transform; use coord_frame='meg' "
                "(e.g. for empty-room data)."
            )
        self.cross_talk_ = _as_str(_maxwell_file(self.cross_talk, inst, "cross_talk", "neuromag"))
        self.calibration_ = _as_str(
            _maxwell_file(self.calibration, inst, "calibration", "neuromag")
        )
        self.destination_ = None if self.destination is None else _as_dev_head_t(self.destination)
        self.head_pos_: np.ndarray | None = None

    def _transform(self, inst: BaseRaw) -> BaseRaw:
        from mne.preprocessing import maxwell_filter

        from ..qc.head import _chpi_source

        info = inst.info
        meg_bads = [ch for ch in info["bads"] if _is_meg(info, ch)]
        picks = _picks(info)
        before = {t: psd_summary(inst.get_data(p), info["sfreq"]) for t, p in picks.items()}
        power_before = {t: float(np.mean(inst.get_data(p) ** 2)) for t, p in picks.items()}
        n_projs = len(info["projs"])

        self.qc_.clear()
        self.head_pos_ = self._head_positions(inst)
        has_chpi = _chpi_source(inst) == "neuromag"
        if self.remove_chpi and has_chpi:
            mne.chpi.filter_chpi(inst, include_line=False, verbose=False)

        if self.st_duration is not None:
            _check_st_duration(self.st_duration, info["sfreq"])
        out = maxwell_filter(
            inst,
            origin=head_origin(self.origin, info) if self.coord_frame == "head" else self.origin,
            int_order=self.int_order,
            ext_order=self.ext_order,
            calibration=self.calibration_,
            cross_talk=self.cross_talk_,
            st_duration=self.st_duration,
            st_correlation=self.st_correlation,
            coord_frame=self.coord_frame,
            destination=self.destination_,
            regularize=self.regularize,
            bad_condition=self.bad_condition,
            head_pos=self.head_pos_,
            skip_by_annotation=list(self.skip_by_annotation),
            verbose=False,
        )
        added = [ch for ch in out.ch_names if ch not in info.ch_names]
        if added:  # MaxFilter-style head-position channels; the positions are in head_pos_
            out.drop_channels(added)

        sss_info = out.info["proc_history"][0]["max_info"]["sss_info"]
        after_picks = {
            t: [out.ch_names.index(info.ch_names[i]) for i in p] for t, p in picks.items()
        }
        self.psd_ = {
            t: {
                "freqs": before[t][0],
                "before": before[t][1],
                "after": psd_summary(out.get_data(after_picks[t]), info["sfreq"])[1],
            }
            for t in picks
        }
        self.qc_.update(
            reconstructed=meg_bads,
            n_basis=int(sss_info["nfree"]),
            tsss=self.st_duration is not None,
            chpi_removed=bool(self.remove_chpi and has_chpi),
            projs_removed=n_projs - len(out.info["projs"]),
            power_change_db={
                t: round(
                    10 * np.log10(np.mean(out.get_data(after_picks[t]) ** 2) / power_before[t]), 2
                )
                for t in picks
                if power_before[t] > 0
            },
            cross_talk=self.cross_talk_,
            calibration=self.calibration_,
        )
        if self.destination_ is not None:
            shift, rotation = _movement(info["dev_head_t"], self.destination_, _origin(out))
            self.qc_.update(
                destination_shift_mm=round(1000 * shift, 2),
                destination_rotation_deg=round(rotation, 2),
            )
        if self.check_ is not None:
            self.qc_["head_movement"] = dict(self.check_.metrics_)
        return out

    def _head_positions(self, raw: BaseRaw) -> np.ndarray | None:
        """Head positions for movement compensation (estimated before cHPI removal)."""
        from ..qc import HeadMovement

        self.check_: Any = None
        if self.head_pos is None:
            return None
        source = (
            None if isinstance(self.head_pos, str) and self.head_pos == "chpi" else self.head_pos
        )
        check = HeadMovement(head_pos=source)
        reason = check.not_applicable(raw)
        if reason is not None:
            raise ValueError(f"Cannot estimate head positions: {reason}.")
        self.check_ = check.compute(raw)
        return np.asarray(self.check_.pos_)

    # ------------------------------------------------------------------

    def _plot_unavailable(self, kind: str) -> str | None:
        if kind == "psd" and not hasattr(self, "psd_"):
            return "the step has not transformed data yet (use fit_transform)"
        if kind in ("head_positions", "displacement") and getattr(self, "check_", None) is None:
            return "no head movement compensation (head_pos=None)"
        return None

    def _plot_psd(self, inst: Any) -> Any:
        title = "tSSS" if self.qc_.get("tsss") else "SSS"
        return plot_psd_comparison(self.psd_, title=f"Spectra before and after {title}")

    def _plot_head_positions(self, inst: Any) -> Any:
        return self.check_.plot("positions")["positions"]

    def _plot_displacement(self, inst: Any) -> Any:
        return self.check_.plot("displacement")["displacement"]


def _check_st_duration(duration: float, sfreq: float) -> None:
    """MNE's overlapping tSSS windows need an even number of samples per buffer."""
    n = round(duration * sfreq)
    if n % 2:
        raise ValueError(
            f"st_duration={duration:g} s is {n} samples at {sfreq:g} Hz; tSSS with overlapping "
            f"windows needs an even number of samples, e.g. st_duration={(n + 1) / sfreq:.6f}."
        )


def _picks(info: mne.Info) -> dict[str, np.ndarray]:
    from ._utils import picks_by_type

    return {t: p for t, p in picks_by_type(info).items() if t in ("mag", "grad")}


def _is_meg(info: mne.Info, ch: str) -> bool:
    return mne.channel_type(info, info["ch_names"].index(ch)) in ("mag", "grad")


def _origin(raw: BaseRaw) -> np.ndarray:
    origin = raw.info["proc_history"][0]["max_info"]["sss_info"]["origin"]
    return np.asarray(origin, float)


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
