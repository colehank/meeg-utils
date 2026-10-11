"""Head position, head movement and digitization checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
from mne.io import BaseRaw

from ..io import detect_system, get_datatypes
from ._base import Check

#: Plausible head radius of a sphere fit, in m (the range MNE accepts).
HEAD_RADIUS_RANGE = (0.050, 0.1085)


class HeadMovement(Check):
    """Head movement during an MEG recording, from continuous head localization.

    Head positions are estimated from the cHPI coils with MNE
    (:func:`mne.chpi.compute_chpi_amplitudes`, :func:`~mne.chpi.compute_chpi_locs`
    and :func:`~mne.chpi.compute_head_pos` for Neuromag;
    :func:`~mne.chpi.extract_chpi_locs_ctf` for CTF), or read from a
    head-position file (e.g. MaxFilter's ``.pos``). Movement is the
    displacement of ``origin`` relative to the sensors, compared with the
    head position in ``info["dev_head_t"]`` (the one source analysis uses).

    Parameters
    ----------
    head_pos : str | Path | ndarray | None
        Precomputed head positions (``.pos`` file or array of shape (n, 10));
        ``None`` estimates them from the cHPI signals.
    warn_mm : float | None
        Maximum displacement that is flagged. There is no formal standard
        and the acceptable movement depends on the participants (children
        and patients move more) and on whether movement is compensated
        later; 5 mm is a common exclusion or movement-correction criterion.
        ``None`` (default) reports the displacement without a verdict.
    origin : tuple of float
        Point (head frame, m) whose displacement is measured.
    gof_limit, dist_limit : float
        Coil fits with a goodness of fit below ``gof_limit`` or further than
        ``dist_limit`` (m) from their digitized position are not used (MNE
        defaults 0.98 and 0.005).
    min_good_fraction : float
        A coil with good fits in less than this fraction of the time is
        flagged (loose, off, or not digitized correctly).
    weighted : bool
        Weight coil fits by their goodness of fit (MNE's current default,
        ``False``).

    Attributes
    ----------
    metrics_ : dict
        ``max_displacement_mm``, ``median_displacement_mm``,
        ``fraction_above`` (of time above ``warn_mm``, when set),
        ``max_rotation_deg``, ``n_positions`` and, when estimated from cHPI,
        ``coil_good_fraction``.

    Notes
    -----
    Figures: ``"positions"`` (:func:`mne.viz.plot_head_positions`, relative
    to ``dev_head_t``) and ``"displacement"`` (displacement over time).
    """

    name: ClassVar[str] = "head_movement"
    modalities: ClassVar[frozenset[str]] = frozenset({"meg"})
    plot_kinds: ClassVar[dict[str, bool]] = {"positions": False, "displacement": False}

    def __init__(
        self,
        *,
        head_pos: str | Path | np.ndarray | None = None,
        warn_mm: float | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.04),
        gof_limit: float = 0.98,
        dist_limit: float = 0.005,
        min_good_fraction: float = 0.9,
        weighted: bool = False,
    ) -> None:
        self.head_pos = head_pos
        self.warn_mm = warn_mm
        self.origin = origin
        self.gof_limit = gof_limit
        self.dist_limit = dist_limit
        self.min_good_fraction = min_good_fraction
        self.weighted = weighted

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
        reason = super().not_applicable(raw)
        if reason is not None or self.head_pos is not None:
            return reason
        if raw.info["dev_head_t"] is None:
            return "needs a device-to-head transform"
        if _chpi_source(raw) is None:
            return "found no continuous head localization (cHPI) in the recording"
        return None

    def _compute(self, raw: BaseRaw) -> None:
        info = raw.info
        if self.head_pos is not None:
            pos = (
                mne.chpi.read_head_pos(self.head_pos)
                if isinstance(self.head_pos, str | Path)
                else np.asarray(self.head_pos, float)
            )
            coil_good = None
        else:
            if _chpi_source(raw) == "neuromag":
                amplitudes = mne.chpi.compute_chpi_amplitudes(raw, verbose=False)
                locs = mne.chpi.compute_chpi_locs(info, amplitudes, verbose=False)
            else:
                locs = mne.chpi.extract_chpi_locs_ctf(raw, verbose=False)
            pos = mne.chpi.compute_head_pos(
                info, locs, gof_limit=self.gof_limit, dist_limit=self.dist_limit,
                weighted=self.weighted, verbose=False,
            )  # fmt: skip
            coil_good = (np.asarray(locs["gofs"]) >= self.gof_limit).mean(axis=0)
        if len(pos) == 0:
            raise ValueError("No head position could be estimated (check the cHPI coils).")

        displacement, rotation = _movement(
            pos, info["dev_head_t"]["trans"], np.asarray(self.origin)
        )
        self.pos_ = pos
        self.info_ = mne.pick_info(info, mne.pick_types(info, meg=True, exclude=[]))
        self.displacement_ = displacement.astype(np.float32)
        self.metrics_.update(
            max_displacement_mm=round(float(displacement.max() * 1000), 2),
            median_displacement_mm=round(float(np.median(displacement) * 1000), 2),
            max_rotation_deg=round(float(rotation.max()), 2),
            n_positions=len(pos),
        )
        detail = ""
        if self.warn_mm is not None:
            above = float((displacement * 1000 > self.warn_mm).mean())
            self.metrics_["fraction_above"] = round(above, 4)
            detail = f"above the limit {100 * above:.0f} % of the time"
        self._judge(
            "max_displacement_mm",
            self.metrics_["max_displacement_mm"],
            warn=self.warn_mm,
            what="maximum head displacement",
            unit=" mm",
            detail=detail,
            typical="5 mm is a common limit",
        )
        if coil_good is not None:
            self.metrics_["coil_good_fraction"] = [round(float(g), 3) for g in coil_good]
            for i, good in enumerate(coil_good):
                self._judge(
                    f"coil{i + 1}_good_fraction",
                    round(float(good), 3),
                    warn=self.min_good_fraction,
                    below=True,
                    what=f"fraction of good fits of HPI coil {i + 1}",
                    detail="coil loose, off or misdigitized",
                )

    def _plot_positions(self, inst: Any) -> Any:
        return mne.viz.plot_head_positions(
            self.pos_, mode="traces", destination=self.info_["dev_head_t"], info=self.info_,
            show=False,
        )  # fmt: skip

    def _plot_displacement(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt

        t = self.pos_[:, 0] - self.pos_[0, 0]
        fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
        ax.plot(t, self.displacement_ * 1000, color="C0")
        if self.warn_mm is not None:
            ax.axhline(self.warn_mm, color="C3", ls="--", lw=1, label=f"{self.warn_mm:g} mm")
            ax.legend(frameon=False)
        ax.set(
            xlabel="Time (s)",
            ylabel="Displacement (mm)",
            title=f"Head displacement relative to dev_head_t (max {self.metrics_['max_displacement_mm']:.1f} mm)",
        )
        return fig


class Digitization(Check):
    """Electrode positions, head digitization and head-to-sensor distance.

    - EEG channels without positions cannot be interpolated or plotted.
    - A sphere fitted to the digitized head shape
      (:func:`mne.bem.fit_sphere_to_headshape`) should have a plausible
      radius (50-108.5 mm, the range MNE accepts) and a centre near the
      head-frame origin (within 20 mm in x/y, where MNE warns).
    - For MEG, the device-to-head transform must have been measured, and the
      distance between the head (sphere surface) and the sensors is
      reported: a large gap lowers the signal-to-noise ratio.

    Attributes
    ----------
    metrics_ : dict
        ``n_points`` by kind, ``eeg_without_position``, ``head_radius_mm``,
        ``origin_mm``, and for MEG ``min_gap_mm`` / ``median_gap_mm``.

    Notes
    -----
    Figures: ``"headshape"`` (digitized points, fitted sphere and MEG
    sensors in three views).
    """

    name: ClassVar[str] = "digitization"
    plot_kinds: ClassVar[dict[str, bool]] = {"headshape": False}

    def _compute(self, raw: BaseRaw) -> None:
        info = raw.info
        datatypes = get_datatypes(raw)
        kinds = {
            int(mne.io.constants.FIFF.FIFFV_POINT_CARDINAL): "fiducials",
            int(mne.io.constants.FIFF.FIFFV_POINT_HPI): "hpi",
            int(mne.io.constants.FIFF.FIFFV_POINT_EEG): "eeg",
            int(mne.io.constants.FIFF.FIFFV_POINT_EXTRA): "headshape",
        }
        counts = dict.fromkeys(kinds.values(), 0)
        for d in info["dig"] or []:
            if int(d["kind"]) in kinds:
                counts[kinds[int(d["kind"])]] += 1
        self.metrics_["n_points"] = counts
        self.info_ = info.copy()

        if "eeg" in datatypes:
            missing = [
                info["ch_names"][i]
                for i in mne.pick_types(info, eeg=True, exclude=[])
                if not np.isfinite(info["chs"][i]["loc"][:3]).all()
                or not np.any(info["chs"][i]["loc"][:3])
            ]
            self.metrics_["eeg_without_position"] = missing
            self._judge(
                "n_eeg_without_position",
                len(missing),
                warn=0,
                what="EEG channels without a position",
                detail="set a montage or read the digitization: " + ", ".join(missing[:15]),
            )

        self.sphere_ = None
        try:
            radius, origin, _ = mne.bem.fit_sphere_to_headshape(info, units="m", verbose=False)
        except (RuntimeError, ValueError) as exc:
            self._note("head_radius_mm", None, f"no sphere fit to the head shape ({exc})", "warn")
        else:
            self.sphere_ = (float(radius), np.asarray(origin, float))
            lo, hi = HEAD_RADIUS_RANGE
            self.metrics_.update(
                head_radius_mm=round(radius * 1000, 1),
                origin_mm=[round(o * 1000, 1) for o in origin],
            )
            self._judge(
                "head_radius_mm_low",
                round(radius * 1000, 1),
                warn=lo * 1000,
                below=True,
                what="fitted head radius",
                unit=" mm",
                detail="implausible; check the digitization",
            )
            self._judge(
                "head_radius_mm_high",
                round(radius * 1000, 1),
                warn=hi * 1000,
                what="fitted head radius",
                unit=" mm",
                detail="implausible; check the digitization",
            )
            self._judge(
                "origin_xy_mm",
                round(float(np.linalg.norm(origin[:2]) * 1000), 1),
                warn=20.0,
                what="sphere centre distance from the head-frame origin (x/y)",
                unit=" mm",
                detail="fiducials or head shape probably misdigitized",
            )

        if "meg" in datatypes:
            trans = info["dev_head_t"]
            if trans is None or np.allclose(trans["trans"], np.eye(4)):
                self._note(
                    "dev_head_t", None, "device-to-head transform missing or identity "
                    "(head position not measured)", "fail",
                )  # fmt: skip
            elif self.sphere_ is not None and detect_system(raw) != "opm":
                radius, origin = self.sphere_
                meg = mne.pick_types(info, meg=True, ref_meg=False, exclude=[])
                sensors = mne.transforms.apply_trans(
                    trans, np.array([info["chs"][i]["loc"][:3] for i in meg])
                )
                gaps = np.linalg.norm(sensors - origin, axis=1) - radius
                self.metrics_.update(
                    min_gap_mm=round(float(gaps.min() * 1000), 1),
                    median_gap_mm=round(float(np.median(gaps) * 1000), 1),
                )
                self._note(
                    "median_gap_mm",
                    self.metrics_["median_gap_mm"],
                    f"head-to-sensor distance: {self.metrics_['min_gap_mm']:.0f} mm (closest), "
                    f"{self.metrics_['median_gap_mm']:.0f} mm (median)",
                )

    def _plot_headshape(self, inst: Any) -> Any:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        info = self.info_
        colors = {"fiducials": "C3", "hpi": "C2", "eeg": "C0", "headshape": "0.6"}
        kinds = {1: "fiducials", 2: "hpi", 3: "eeg", 4: "headshape"}
        points: dict[str, list[np.ndarray]] = {k: [] for k in colors}
        for d in info["dig"] or []:
            if (
                int(d["kind"]) in kinds
                and int(d["coord_frame"]) == mne.io.constants.FIFF.FIFFV_COORD_HEAD
            ):
                points[kinds[int(d["kind"])]].append(np.asarray(d["r"]) * 1000)
        meg = mne.pick_types(info, meg=True, ref_meg=False, exclude=[])
        sensors = None
        if len(meg) and info["dev_head_t"] is not None:
            sensors = (
                mne.transforms.apply_trans(
                    info["dev_head_t"], np.array([info["chs"][i]["loc"][:3] for i in meg])
                )
                * 1000
            )
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
        for ax, (title, i, j) in zip(
            axes, (("top", 0, 1), ("side", 1, 2), ("front", 0, 2)), strict=True
        ):
            if sensors is not None:
                ax.scatter(sensors[:, i], sensors[:, j], s=4, color="0.85", label="MEG sensors")
            for kind, pts in points.items():
                if pts:
                    arr = np.array(pts)
                    ax.scatter(arr[:, i], arr[:, j], s=10, color=colors[kind], label=kind)
            if self.sphere_ is not None:
                radius, origin = self.sphere_
                ax.add_patch(
                    Circle(
                        (origin[i] * 1000, origin[j] * 1000), radius * 1000,
                        fill=False, ls="--", color="C1", label="fitted sphere",
                    )
                )  # fmt: skip
            ax.set(
                title=title, xlabel="xyz"[i] + " (mm)", ylabel="xyz"[j] + " (mm)", aspect="equal"
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False)
        radius_text = (
            f"sphere radius {self.metrics_['head_radius_mm']:.0f} mm"
            if "head_radius_mm" in self.metrics_
            else "no sphere fit"
        )
        fig.suptitle(f"Digitization (head frame): {radius_text}")
        return fig


def _chpi_source(raw: BaseRaw) -> str | None:
    """Which continuous head localization the recording has, if any."""
    system = detect_system(raw)
    if system == "neuromag":
        freqs = mne.chpi.get_chpi_info(raw.info, on_missing="ignore", verbose=False)[0]
        return "neuromag" if len(freqs) else None
    if system == "ctf" and any(name.startswith("HLC00") for name in raw.ch_names):
        return "ctf"
    return None


def _movement(
    pos: np.ndarray, reference: np.ndarray, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement (m) of ``origin`` relative to the sensors, and rotation (deg)."""
    trans, rot, _ = mne.chpi.head_pos_to_trans_rot_t(pos)
    # origin in device coordinates: R^T (o - t)
    dev = np.einsum("nji,nj->ni", rot, origin - trans)
    ref_rot, ref_trans = reference[:3, :3], reference[:3, 3]
    ref_dev = ref_rot.T @ (origin - ref_trans)
    displacement = np.linalg.norm(dev - ref_dev, axis=1)
    rel = np.einsum("ij,njk->nik", ref_rot.T, rot)
    angle = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return displacement, angle
