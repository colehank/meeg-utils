"""Plotting helpers shared by the steps."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

#: Scale to display units and their labels, per channel type.
UNITS: dict[str, tuple[float, str]] = {
    "eeg": (1e6, "µV"),
    "mag": (1e15, "fT"),
    "grad": (1e13, "fT/cm"),
}


def psd_summary(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of each channel (4 s windows), compact enough to store on a step."""
    from scipy.signal import welch  # type: ignore[import-untyped]

    nperseg = int(min(data.shape[-1], 4 * sfreq))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
    if psd.ndim == 3:  # epochs: average over epochs
        psd = psd.mean(axis=0)
    return freqs, psd.astype(np.float32)


def plot_psd_comparison(
    spectra: dict[str, dict[str, Any]],
    *,
    title: str,
    marks: Iterable[float] = (),
    labels: tuple[str, str] = ("before", "after"),
) -> Any:
    """Plot PSDs (dB) before and after a step, one panel per channel type.

    Lines are the median over channels and bands the 5-95th percentiles, so
    single outlying channels do not distort the picture; flat channels
    (zero power) are left out.

    Parameters
    ----------
    spectra : dict
        Channel type mapped to ``{"freqs", "before", "after"}`` (PSDs of
        shape (n_channels, n_freqs), in SI units squared per Hz).
    title : str
        Figure title.
    marks : iterable of float
        Frequencies to mark with vertical lines (e.g. line-noise harmonics).
    labels : tuple of str
        Legend labels.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    marks = list(marks)
    fig, axes = plt.subplots(
        1, len(spectra), figsize=(4.5 * len(spectra), 3.2), squeeze=False, layout="constrained"
    )
    for ax, (ch_type, spec) in zip(axes[0], spectra.items(), strict=True):
        scale, unit = UNITS.get(ch_type, (1.0, "a.u."))
        live = (spec["before"] > 0).any(axis=1) & (spec["after"] > 0).any(axis=1)
        for key, label, color, ls in (
            ("before", labels[0], "0.45", "--"),
            ("after", labels[1], "C0", "-"),
        ):
            psd_db = 10 * np.log10(np.maximum(spec[key][live] * scale**2, 1e-30))
            lo, mid, hi = np.percentile(psd_db, [5, 50, 95], axis=0)
            ax.plot(spec["freqs"], mid, color=color, lw=1.2, ls=ls, label=label)
            ax.fill_between(spec["freqs"], lo, hi, color=color, alpha=0.15, lw=0)
        for f in marks:
            ax.axvline(f, color="C3", lw=0.6, ls=":", alpha=0.7)
        ax.set(
            title=ch_type.upper(),
            xlabel="Frequency (Hz)",
            ylabel=f"PSD (dB re 1 ({unit})²/Hz)" if "/" in unit else f"PSD (dB re 1 {unit}²/Hz)",
            xlim=(spec["freqs"][1], spec["freqs"][-1]),
        )
        ax.set_xscale("log")
    axes[0, 0].legend(frameon=False, fontsize="small")
    fig.suptitle(title)
    return fig


def plot_sensor_groups(info: Any, groups: dict[str, list[str]], *, title: str) -> Any:
    """Plot sensor positions, highlighting named groups of channels.

    Parameters
    ----------
    info : Info
        Measurement info with sensor positions.
    groups : dict
        Label mapped to channel names to highlight (e.g. ``{"bad": [...]}``).
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    import mne

    ch_types = [t for t in ("eeg", "mag", "grad") if t in info.get_channel_types(unique=True)]
    fig, axes = plt.subplots(
        1, len(ch_types), figsize=(4 * len(ch_types), 4.4), squeeze=False, layout="constrained"
    )
    colors = [plt.get_cmap("tab10")(i) for i in (3, 1, 2, 4, 5)]
    for ax, ch_type in zip(axes[0], ch_types, strict=True):
        picks = mne.pick_types(
            info, meg=ch_type if ch_type != "eeg" else False, eeg=ch_type == "eeg", exclude=[]
        )
        sub = mne.pick_info(info, picks).copy()
        with sub._unlock():
            sub["bads"] = []
        mne.viz.plot_sensors(sub, kind="topomap", axes=ax, show=False, verbose=False)
        xy = np.asarray(ax.collections[0].get_offsets())  # one point per channel, in order
        for color, (name, chs) in zip(colors, groups.items(), strict=False):
            idx = [sub.ch_names.index(ch) for ch in chs if ch in sub.ch_names]
            ax.scatter(
                xy[idx, 0], xy[idx, 1], s=70, color=color, zorder=5, label=f"{name} ({len(idx)})"
            )
            for i in idx:
                ax.annotate(
                    sub.ch_names[i],
                    xy[i],
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    zorder=6,
                )
        ax.set_title(ch_type.upper())
        if groups:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False, ncols=len(groups)
            )
    fig.suptitle(title)
    return fig
