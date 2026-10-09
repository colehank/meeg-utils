"""The plot() interface shared by steps and quality checks."""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.exceptions import NotFittedError


class PlotMixin:
    """Draw figures declared in ``plot_kinds`` with ``_plot_<kind>`` methods.

    Subclasses set ``plot_kinds`` (kind -> whether the figure needs data) and
    ``_fitted_attr``, the attribute that exists once the object is fitted.
    """

    #: Figures the object can draw once fitted: kind -> whether it needs data
    #: (``inst``). Each kind is drawn by a ``_plot_<kind>(inst, **kwargs)`` method.
    plot_kinds: ClassVar[dict[str, bool]] = {}
    _fitted_attr: ClassVar[str] = "system_"

    def plot(self, kind: str | None = None, *, inst: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Draw quality-control figures.

        Figures are returned, not shown. Most are drawn from compact
        diagnostics stored while fitting, so they need no data; kinds marked
        ``True`` in :attr:`plot_kinds` need ``inst``.

        Parameters
        ----------
        kind : str | None
            The figure to draw (see :attr:`plot_kinds`). ``None`` draws every
            figure that the available inputs allow.
        inst : Raw | Epochs | Evoked | None
            Data for the figures that need it.
        **kwargs
            Passed to the plotting method of ``kind``; only valid with ``kind``.

        Returns
        -------
        dict
            Kind mapped to its :class:`matplotlib.figure.Figure` (or a list of
            figures).
        """
        name = type(self).__name__
        if not hasattr(self, self._fitted_attr):
            raise NotFittedError(
                f"This {name} instance is not fitted yet; there is nothing to plot."
            )
        if kind is None:
            if kwargs:
                raise TypeError("Plot options can only be passed together with kind.")
            kinds = [
                k for k, needs_data in self.plot_kinds.items() if inst is not None or not needs_data
            ]
        else:
            if kind not in self.plot_kinds:
                available = sorted(self.plot_kinds) or "none"
                raise ValueError(f"{name} has no {kind!r} plot; available: {available}.")
            if self.plot_kinds[kind] and inst is None:
                raise ValueError(f"The {kind!r} plot of {name} needs inst=<data>.")
            kinds = [kind]

        import matplotlib.pyplot as plt

        figures: dict[str, Any] = {}
        with plt.ioff():
            for k in kinds:
                figures[k] = getattr(self, f"_plot_{k}")(inst, **kwargs)
        return figures
