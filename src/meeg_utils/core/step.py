"""Base class for processing steps."""

from __future__ import annotations

from typing import Any, ClassVar, Self, TypeAlias

from mne import Evoked
from mne.epochs import BaseEpochs
from mne.io import BaseRaw
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..io import detect_system

Inst: TypeAlias = BaseRaw | BaseEpochs | Evoked


class Step(BaseEstimator):
    """Base class for all processing steps.

    A step follows scikit-learn conventions:

    - All parameters are declared as keyword arguments of ``__init__`` and
      stored unmodified as attributes; ``__init__`` does no work.
    - :meth:`fit` learns state from the data and returns ``self``. Learned
      state is stored in attributes with a trailing underscore.
    - :meth:`transform` applies the learned state without refitting and
      returns the processed data.
    - :meth:`fit_transform` fits and transforms the same data.

    Subclasses implement :meth:`_fit` (optional, for stateful steps) and
    :meth:`_transform`, and may record quality-control metrics in
    ``self.qc_`` during either.

    Subclasses must not drop channels, nor change ``first_samp``,
    ``meas_date`` or the timing of annotations, unless they declare it with
    ``changes_channels = True`` / ``changes_times = True``.

    Attributes
    ----------
    system_ : str
        Acquisition system the step was fitted on (see
        :func:`meeg_utils.io.detect_system`).
    qc_ : dict
        Quality-control metrics recorded while fitting and transforming.
    """

    #: Data types the step accepts.
    accepts: ClassVar[tuple[type, ...]] = (BaseRaw,)
    #: Data type the step returns; ``None`` means the same type as its input.
    returns: ClassVar[type | None] = None
    #: Acquisition systems the step applies to; ``None`` means all systems.
    systems: ClassVar[frozenset[str] | None] = None
    #: Whether the step may add, drop or reorder channels.
    changes_channels: ClassVar[bool] = False
    #: Whether the step may change the time axis (e.g. resampling, cropping).
    changes_times: ClassVar[bool] = False
    #: Figures the step can draw once fitted: kind -> whether it needs data
    #: (``inst``). Each kind is drawn by a ``_plot_<kind>(inst, **kwargs)`` method.
    plot_kinds: ClassVar[dict[str, bool]] = {}

    def fit(self, inst: Inst, y: Any = None, *, system: str | None = None) -> Self:
        """Learn the step's state from the data.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data to fit on. It is not modified.
        y : None
            Ignored; present for scikit-learn API compatibility.
        system : str | None
            Acquisition system of the data. If ``None``, it is detected from
            ``inst``. Pipelines detect it once and pass it to every step.

        Returns
        -------
        self : Step
            The fitted step.
        """
        self._check_input(inst)
        self.system_ = detect_system(inst) if system is None else system
        if self.systems is not None and self.system_ not in self.systems:
            raise ValueError(
                f"{type(self).__name__} applies to {sorted(self.systems)} data, "
                f"but the data come from a {self.system_!r} system."
            )
        self.qc_: dict[str, Any] = {}
        self._fit(inst)
        return self

    def transform(self, inst: Inst, *, copy: bool = True) -> Inst:
        """Apply the fitted step to the data.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data to transform.
        copy : bool
            If ``True`` (default), ``inst`` is left untouched and a processed
            copy is returned. If ``False``, ``inst`` may be modified in place
            to save memory; use the return value either way.

        Returns
        -------
        Raw | Epochs | Evoked
            The processed data.
        """
        check_is_fitted(self, "system_")
        self._check_input(inst)
        if copy:
            inst = inst.copy()
        return self._transform(inst)

    def fit_transform(
        self,
        inst: Inst,
        y: Any = None,
        *,
        system: str | None = None,
        copy: bool = True,
    ) -> Inst:
        """Fit the step on the data and transform the same data.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data.
        y : None
            Ignored; present for scikit-learn API compatibility.
        system : str | None
            Acquisition system of the data; detected if ``None``.
        copy : bool
            Whether to leave ``inst`` untouched (see :meth:`transform`).

        Returns
        -------
        Raw | Epochs | Evoked
            The processed data.
        """
        return self.fit(inst, system=system).transform(inst, copy=copy)

    def plot(
        self, kind: str | None = None, *, inst: Inst | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Draw quality-control figures of the fitted step.

        Figures are returned, not shown. Most are drawn from compact
        diagnostics stored while fitting and transforming, so they need no
        data; kinds marked ``True`` in :attr:`plot_kinds` need ``inst``.

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
            Kind mapped to its :class:`matplotlib.figure.Figure`.
        """
        check_is_fitted(self, "system_")
        if kind is None:
            if kwargs:
                raise TypeError("Plot options can only be passed together with kind.")
            kinds = [
                k for k, needs_data in self.plot_kinds.items() if inst is not None or not needs_data
            ]
        else:
            if kind not in self.plot_kinds:
                available = sorted(self.plot_kinds) or "none"
                raise ValueError(
                    f"{type(self).__name__} has no {kind!r} plot; available: {available}."
                )
            if self.plot_kinds[kind] and inst is None:
                raise ValueError(f"The {kind!r} plot of {type(self).__name__} needs inst=<data>.")
            kinds = [kind]

        import matplotlib.pyplot as plt

        figures: dict[str, Any] = {}
        with plt.ioff():
            for k in kinds:
                figures[k] = getattr(self, f"_plot_{k}")(inst, **kwargs)
        return figures

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _fit(self, inst: Inst) -> None:
        """Learn state from ``inst``; stateless steps need not override this."""

    def _transform(self, inst: Inst) -> Inst:
        """Process ``inst``, which the step owns and may modify in place."""
        raise NotImplementedError

    # ------------------------------------------------------------------

    def _check_input(self, inst: Inst) -> None:
        if not isinstance(inst, self.accepts):
            accepted = ", ".join(cls.__name__ for cls in self.accepts)
            raise TypeError(f"{type(self).__name__} accepts {accepted}, got {type(inst).__name__}.")
        if isinstance(inst, BaseRaw | BaseEpochs) and not inst.preload:
            raise ValueError(
                f"{type(self).__name__} needs preloaded data; call inst.load_data() first."
            )
