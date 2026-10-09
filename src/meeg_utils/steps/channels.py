"""Bad-channel detection, interpolation and referencing."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst
from ..io import get_datatypes

BAD_CHANNEL_METHODS = ("auto", "prep", "maxwell")
_MAXWELL_SYSTEMS = frozenset({"neuromag", "ctf"})


class BadChannels(Step):
    """Detect bad channels and mark them in ``info["bads"]``.

    The data themselves are not changed; follow with :class:`Interpolate`
    to repair the channels. Channels already marked bad stay bad.

    Parameters
    ----------
    method : {"auto", "prep", "maxwell"}
        - ``"prep"``: PREP noisy-channel detection for EEG (Bigdely-Shamlo et
          al., 2015) via PyPREP, running all PREP criteria: NaN/flat,
          deviation, high-frequency noise, correlation, low SNR and,
          optionally, RANSAC.
        - ``"maxwell"``: :func:`mne.preprocessing.find_bad_channels_maxwell`
          for MEG (Neuromag; CTF via a grade-0 compensated copy).
        - ``"auto"`` (default): PREP for EEG channels and Maxwell for MEG
          channels of Neuromag/CTF systems; MEG from other systems raises,
          because no validated default exists for them.
    ransac : bool
        PREP: also run RANSAC, which needs electrode positions.
    random_state : int | None
        PREP: seed for RANSAC.
    reject_by_annotation : {"omit", None}
        PREP: whether to ignore ``BAD_`` annotated segments.
    origin : str | tuple
        Maxwell: head origin, ``"auto"`` fits it to the digitization.
    cross_talk, calibration : str | None
        Maxwell: Neuromag cross-talk and fine-calibration files.
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
        ``fraction`` and ``by_criterion``.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)

    def __init__(
        self,
        method: str = "auto",
        *,
        ransac: bool = True,
        random_state: int | None = 42,
        reject_by_annotation: str | None = None,
        origin: str | tuple[float, float, float] = "auto",
        cross_talk: str | None = None,
        calibration: str | None = None,
        limit: float = 7.0,
        h_freq: float | None = 40.0,
    ) -> None:
        self.method = method
        self.ransac = ransac
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
        from pyprep.find_noisy_channels import NoisyChannels

        eeg = inst.copy().pick("eeg", exclude=[])
        if self.ransac:
            pos = np.array([ch["loc"][:3] for ch in eeg.info["chs"]])
            if not np.isfinite(pos).all() or np.allclose(pos, 0):
                raise ValueError(
                    "PREP's RANSAC needs electrode positions: set a montage "
                    "(raw.set_montage) or use BadChannels(ransac=False)."
                )
        manual = [ch for ch in inst.info["bads"] if ch in eeg.ch_names]
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
        by_criterion = {
            key.removeprefix("bad_by_"): sorted(map(str, chs))
            for key, chs in finder.get_bads(as_dict=True).items()
            if key not in ("bad_all", "bad_by_manual")
        }
        found = sorted({ch for chs in by_criterion.values() for ch in chs} - set(manual))
        return found, by_criterion, len(eeg.ch_names)

    def _maxwell(self, inst: BaseRaw) -> tuple[list[str], dict[str, list[str]], int]:
        from mne.preprocessing import find_bad_channels_maxwell

        meg = inst.copy().pick(["meg", "ref_meg"], exclude=[])
        if self.system_ == "ctf" and meg.compensation_grade != 0:
            meg.apply_gradient_compensation(0, verbose=False)  # required by Maxwell filtering
        noisy, flat = find_bad_channels_maxwell(
            meg,
            limit=self.limit,
            origin=self.origin,
            cross_talk=self.cross_talk,
            calibration=self.calibration,
            h_freq=self.h_freq,
            verbose=False,
        )
        found = sorted(set(noisy) | set(flat))
        found = [ch for ch in found if ch not in inst.info["bads"]]
        n_meg = len(meg.copy().pick("meg", exclude=[]).ch_names)
        return found, {"noisy": sorted(noisy), "flat": sorted(flat)}, n_meg

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
        Head origin; ``"auto"`` fits it to the digitization.
    reset_bads : bool
        Whether to clear the interpolated channels from ``info["bads"]``.
    exclude : list of str
        Bad channels to leave as they are.

    Attributes
    ----------
    qc_ : dict
        ``interpolated`` (channel names) and ``fraction`` of channels.
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)

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
        n_data = len(inst.copy().pick(["meg", "eeg"], exclude=[]).ch_names)
        self.qc_.update(interpolated=bads, fraction=len(bads) / n_data if n_data else 0.0)
        if bads:
            inst.interpolate_bads(
                reset_bads=self.reset_bads,
                mode="accurate",
                origin=self.origin,
                method=self.method or {"eeg": "spline", "meg": "MNE"},
                exclude=list(self.exclude),
                verbose=False,
            )
        return inst


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
