"""Ready-made pipelines.

Recommended presets (``eeg-erp``, ``eeg-rest``, ``meg-erp``, ``meg-rest``)
take every parameter from a published default; the sources are listed in
``docs/presets/recommended.md``. Dataset presets reproduce the
preprocessing of a published dataset with its known problems fixed; every
difference from the original code is listed in ``docs/presets/<name>.md``.
Use them through :meth:`meeg_utils.Pipeline.preset`, or :func:`get` and
:func:`available`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ._had_meeg import had_meeg
from ._recommended import eeg_erp, eeg_rest, meg_erp, meg_rest

if TYPE_CHECKING:
    from ..core import Pipeline

#: Preset name mapped to its factory; each factory documents its parameters.
_PRESETS: dict[str, Callable[..., Pipeline]] = {
    "eeg-erp": eeg_erp,
    "eeg-rest": eeg_rest,
    "meg-erp": meg_erp,
    "meg-rest": meg_rest,
    "had-meeg": had_meeg,
}


def available() -> dict[str, str]:
    """Return the preset names with the first line of their description.

    Returns
    -------
    dict
        Preset name mapped to a one-line summary.
    """
    return {
        name: (factory.__doc__ or "").strip().splitlines()[0] for name, factory in _PRESETS.items()
    }


def get(name: str, **kwargs) -> Pipeline:
    """Build the pipeline of a preset.

    Parameters
    ----------
    name : str
        Preset name, see :func:`available`.
    **kwargs
        Preset options, e.g. ``stage``, ``system`` (MEG presets),
        ``event_id`` (ERP epochs) or ``datatype`` (``"had-meeg"``).

    Returns
    -------
    Pipeline
        A new, unfitted pipeline; change parameters with ``set_params``.

    Raises
    ------
    ValueError
        If the preset does not exist.
    """
    if name not in _PRESETS:
        raise ValueError(f"Unknown preset {name!r}; available: {sorted(_PRESETS)}")
    return _PRESETS[name](**kwargs)


def options(name: str) -> list[str]:
    """Return the options a preset accepts.

    Parameters
    ----------
    name : str
        Preset name, see :func:`available`.

    Returns
    -------
    list of str
        Option names (keyword arguments of :func:`get`).
    """
    import inspect

    if name not in _PRESETS:
        raise ValueError(f"Unknown preset {name!r}; available: {sorted(_PRESETS)}")
    return list(inspect.signature(_PRESETS[name]).parameters)


__all__ = ["available", "get", "options"]
