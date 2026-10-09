"""MEG head-position alignment."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
from mne import Evoked, Transform
from mne.epochs import BaseEpochs
from mne.io import BaseRaw

from ..core import Step
from ..core.step import Inst
from ..io.system import MEG_SYSTEMS

#: Head displacement (m) above which a warning is emitted: in simulations
#: (tests/test_steps/test_head.py) the mapping error grows from ~1-3 % at
#: 10 mm to ~2-6 % at 20 mm, and the minimum-norm mapping is not designed
#: for larger movements.
WARN_SHIFT = 0.02


class HeadAlign(Step):
    """Map MEG data to a common head position (device-to-head transform).

    Different runs of a session rarely share the head position, so the same
    brain source projects onto different sensors. This step computes, from
    the sensor geometry alone, what the sensors *would have* measured with
    the head at ``destination`` (minimum-norm field mapping in a spherical
    harmonic basis; Knösche, 2002, NeuroImage 17:1307), applies it to the
    good MEG channels and sets ``info["dev_head_t"]`` to the destination.

    For Neuromag data, Maxwell filtering with a destination
    (:func:`mne.preprocessing.maxwell_filter`) is the usual alternative; this
    step works for any MEG system with a head transform (CTF, KIT, BTi,
    Neuromag, ...) but not for on-head OPM arrays, whose sensors move with
    the head.

    Parameters
    ----------
    destination : str | path-like | array, shape (4, 4) | Transform | dict
        The head position to map to, as a device-to-head transform, or a
        FIF file (or any recording MNE can read) whose ``dev_head_t`` is
        used. For several runs, a common choice is the average over runs,
        see :func:`meeg_utils.io.average_dev_head_t`.
    origin : "auto" | tuple of float
        Origin of the spherical harmonic expansion in the head frame (m);
        ``"auto"`` fits a sphere to the head digitization.
    mode : {"accurate", "fast"}
        Accuracy of the Legendre expansion used for the field mapping.

    Attributes
    ----------
    mapping_ : ndarray, shape (n_good_meg, n_good_meg)
        The mapping from the measured to the destination sensor positions.
    picks_ : ndarray
        Indices of the mapped channels (good MEG channels; reference
        channels and bad channels are left unchanged).
    dev_head_t_ : Transform
        The destination transform.
    qc_ : dict
        ``shift_mm`` (displacement of the head origin relative to the
        sensors), ``rotation_deg``, ``n_channels`` and ``unmapped_bads``.

    Notes
    -----
    - Bad channels are not mapped and stay marked bad; interpolating them
      afterwards (:class:`Interpolate`) uses the destination geometry.
    - CTF data are mapped in their current compensation grade, as MNE does
      for bad-channel interpolation; reference sensors are unchanged.
    - SSP projectors are kept; compute new ones after alignment if they
      depend on the head position.
    - A fitted step only transforms data recorded at the same head position
      with the same channels (normally the data it was fitted on).

    Figures (:meth:`plot`): ``"positions"`` (sensor positions relative to the
    head before and after, in three views).
    """

    accepts: ClassVar[tuple[type, ...]] = (BaseRaw, BaseEpochs, Evoked)
    systems: ClassVar[frozenset[str] | None] = frozenset(MEG_SYSTEMS) - {"opm"}
    plot_kinds: ClassVar[dict[str, bool]] = {"positions": False}

    def __init__(
        self,
        destination: Any,
        *,
        origin: str | tuple[float, float, float] = "auto",
        mode: str = "accurate",
    ) -> None:
        self.destination = destination
        self.origin = origin
        self.mode = mode

    def _fit(self, inst: Inst) -> None:
        from mne.bem import _check_origin
        from mne.forward._field_interpolation import _map_meg_or_eeg_channels

        info = inst.info
        if info["dev_head_t"] is None:
            raise ValueError("The data have no device-to-head transform (info['dev_head_t']).")
        picks = mne.pick_types(info, meg=True, ref_meg=False, exclude="bads")
        if len(picks) == 0:
            raise ValueError("HeadAlign needs good MEG channels.")
        dest = _as_dev_head_t(self.destination)

        try:
            origin = _check_origin(self.origin, info)
        except Exception as exc:
            raise ValueError(
                "Could not fit the head origin to the digitization; pass origin=(x, y, z) "
                f"in metres, head frame (e.g. (0.0, 0.0, 0.04)). Original error: {exc}"
            ) from exc

        info_from = mne.pick_info(info, picks)
        info_to = info_from.copy()
        with info_to._unlock():
            info_to["dev_head_t"] = dest
        with mne.use_log_level("warning"):
            self.mapping_ = _map_meg_or_eeg_channels(
                info_from, info_to, mode=self.mode, origin=origin
            )
        self.picks_ = picks
        self.origin_ = origin
        self.dev_head_t_ = dest
        self.source_dev_head_t_ = info["dev_head_t"].copy()
        self.ch_names_ = list(info["ch_names"])
        self.sensors_ = np.array([info["chs"][p]["loc"][:3] for p in picks])  # device frame

        shift, rotation = _movement(self.source_dev_head_t_, dest, origin)
        bads = [ch for ch in info["bads"] if _is_meg(info, ch)]
        self.qc_.update(
            shift_mm=round(1000 * shift, 2),
            rotation_deg=round(rotation, 2),
            n_channels=len(picks),
            unmapped_bads=bads,
        )
        if shift > WARN_SHIFT:
            warnings.warn(
                f"The head moves {1000 * shift:.1f} mm relative to the sensors; field "
                "mapping is less accurate for such large movements.",
                stacklevel=4,
            )

    def _transform(self, inst: Inst) -> Inst:
        info = inst.info
        if list(info["ch_names"]) != self.ch_names_ or not np.allclose(
            info["dev_head_t"]["trans"], self.source_dev_head_t_["trans"]
        ):
            raise ValueError(
                "HeadAlign was fitted on data with a different head position or channels; "
                "fit it on this recording."
            )
        data = inst._data
        if isinstance(inst, BaseEpochs):
            data[:, self.picks_] = np.einsum("ij,ejt->eit", self.mapping_, data[:, self.picks_])
        else:
            data[self.picks_] = self.mapping_ @ data[self.picks_]
        with info._unlock():
            info["dev_head_t"] = self.dev_head_t_.copy()
            if info.get("ctf_head_t") is not None and info.get("dev_ctf_t") is not None:
                # keep the CTF transforms consistent: dev_head = ctf_head @ dev_ctf
                ctf_head = info["ctf_head_t"]["trans"]
                info["dev_ctf_t"] = Transform(
                    info["dev_ctf_t"]["from"],
                    info["dev_ctf_t"]["to"],
                    np.linalg.inv(ctf_head) @ self.dev_head_t_["trans"],
                )
        return inst

    def _plot_positions(self, inst: Inst | None) -> Any:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        before = mne.transforms.apply_trans(self.source_dev_head_t_, self.sensors_) * 1000
        after = mne.transforms.apply_trans(self.dev_head_t_, self.sensors_) * 1000
        origin = np.asarray(self.origin_) * 1000
        views = (("top", 0, 1), ("side", 1, 2), ("front", 0, 2))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
        for ax, (name, i, j) in zip(axes, views, strict=True):
            segments = np.stack([before[:, [i, j]], after[:, [i, j]]], axis=1)
            ax.add_collection(LineCollection(list(segments), colors="0.3", linewidths=0.6))
            ax.scatter(before[:, i], before[:, j], s=8, color="0.6", label="measured")
            ax.scatter(after[:, i], after[:, j], s=8, color="C0", label="destination")
            ax.plot(*origin[[i, j]], "+", color="C3", ms=12, mew=2, label="head origin")
            ax.set(
                title=name,
                xlabel="xyz"[i] + " (mm, head)",
                ylabel="xyz"[j] + " (mm, head)",
                aspect="equal",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
        fig.suptitle(
            f"Sensors relative to the head: shift {self.qc_['shift_mm']:.1f} mm, "
            f"rotation {self.qc_['rotation_deg']:.1f}°"
        )
        return fig


def _as_dev_head_t(destination: Any) -> Transform:
    """Return ``destination`` as a validated MEG-device-to-head Transform."""
    if destination is None:
        raise ValueError("HeadAlign needs a destination head position.")
    if isinstance(destination, str | Path):
        info = mne.io.read_info(destination, verbose=False)
        if info["dev_head_t"] is None:
            raise ValueError(f"{destination} has no device-to-head transform.")
        trans = np.asarray(info["dev_head_t"]["trans"], float)
    elif isinstance(destination, dict):  # Transform, or its serialized form
        if "trans" not in destination:
            raise ValueError("A destination dict must be a Transform (with a 'trans' entry).")
        frames = (destination.get("from"), destination.get("to"))
        expected = (
            mne.io.constants.FIFF.FIFFV_COORD_DEVICE,
            mne.io.constants.FIFF.FIFFV_COORD_HEAD,
        )
        if frames != expected:
            raise ValueError("The destination must transform MEG device to head coordinates.")
        trans = np.asarray(destination["trans"], float)
    else:
        trans = np.asarray(destination, float)
    if trans.shape != (4, 4):
        raise ValueError(f"The destination must be a 4x4 transform, got shape {trans.shape}.")
    rot = trans[:3, :3]
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rot), 1):
        raise ValueError("The destination is not a rigid transform (rotation + translation).")
    if np.linalg.norm(trans[:3, 3]) > 1:
        raise ValueError("The destination places the head more than 1 m from the device.")
    return Transform("meg", "head", trans)


def _movement(source: Transform, dest: Transform, origin: np.ndarray) -> tuple[float, float]:
    """Head-origin displacement relative to the sensors (m) and rotation (deg)."""
    o = np.append(origin, 1.0)
    shift = np.linalg.norm(
        (np.linalg.inv(source["trans"]) @ o - np.linalg.inv(dest["trans"]) @ o)[:3]
    )
    rel = dest["trans"][:3, :3] @ source["trans"][:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    return float(shift), float(angle)


def _is_meg(info: mne.Info, ch: str) -> bool:
    idx = info["ch_names"].index(ch)
    return bool(info["chs"][idx]["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH)
