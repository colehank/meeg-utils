"""Tests for the Step base class and the step contract checker."""

from __future__ import annotations

import numpy as np
import pytest
from mne.io import BaseRaw
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from meeg_utils.testing import check_step

from ._steps import (
    CTFOnly,
    Demean,
    DropsChannel,
    ModifiesInputInFit,
    RebuildsRaw,
    Scale,
    ToEpochs,
    WithTuple,
)


class TestStepAPI:
    """The scikit-learn style API of Step."""

    def test_fit_returns_self_and_sets_state(self, small_raw: BaseRaw) -> None:
        """Fit returns the step and sets system_, qc_ and learned state."""
        step = Demean()
        assert step.fit(small_raw) is step
        assert step.system_ == "eeg"
        assert step.means_.shape == (4, 1)
        assert step.qc_["max_abs_mean"] > 0

    def test_transform_before_fit_raises(self, small_raw: BaseRaw) -> None:
        """Transform on an unfitted step raises NotFittedError."""
        with pytest.raises(NotFittedError):
            Scale().transform(small_raw)

    def test_transform_uses_fitted_state(self, small_raw: BaseRaw) -> None:
        """Transform applies state learned on other data, without refitting."""
        other = small_raw.copy()
        other._data += 1e-3
        step = Demean().fit(small_raw)
        out = step.transform(other)
        expected = other.get_data(picks="eeg") - step.means_
        np.testing.assert_allclose(out.get_data(picks="eeg"), expected)

    def test_copy_true_leaves_input_untouched(self, small_raw: BaseRaw) -> None:
        """With copy=True the input is not modified."""
        before = small_raw.get_data().copy()
        out = Scale(3.0).fit_transform(small_raw)
        np.testing.assert_array_equal(small_raw.get_data(), before)
        np.testing.assert_allclose(out.get_data(), before * 3.0)

    def test_copy_false_works_in_place(self, small_raw: BaseRaw) -> None:
        """With copy=False the input may be modified in place."""
        before = small_raw.get_data().copy()
        out = Scale(3.0).fit_transform(small_raw, copy=False)
        assert out is small_raw
        np.testing.assert_allclose(small_raw.get_data(), before * 3.0)

    def test_rejects_wrong_input_type(self, small_raw: BaseRaw) -> None:
        """A step refuses data types it does not accept."""
        epochs = ToEpochs().fit_transform(small_raw)
        with pytest.raises(TypeError, match="accepts BaseRaw"):
            Scale().fit(epochs)

    def test_rejects_unloaded_data(self, small_raw: BaseRaw, tmp_path) -> None:
        """A step refuses data that are not preloaded."""
        import mne

        fname = tmp_path / "raw.fif"
        small_raw.save(fname)
        unloaded = mne.io.read_raw_fif(fname, preload=False, verbose=False)
        with pytest.raises(ValueError, match="preloaded"):
            Scale().fit(unloaded)

    def test_rejects_wrong_system(self, small_raw: BaseRaw) -> None:
        """A system-specific step refuses other systems, loudly."""
        with pytest.raises(ValueError, match=r"applies to \['ctf'\].*'eeg'"):
            CTFOnly().fit(small_raw)

    def test_explicit_system_overrides_detection(self, small_raw: BaseRaw) -> None:
        """A system passed by the pipeline is used as is."""
        assert CTFOnly().fit(small_raw, system="ctf").system_ == "ctf"

    def test_params_and_clone(self) -> None:
        """Parameters behave like scikit-learn estimator parameters."""
        step = Scale(5.0)
        assert step.get_params() == {"factor": 5.0}
        step.set_params(factor=7.0)
        cloned = clone(step)
        assert cloned is not step
        assert cloned.factor == 7.0
        assert repr(step) == "Scale(factor=7.0)"


class TestCheckStep:
    """The contract checker accepts good steps and catches bad ones."""

    @pytest.mark.parametrize("step", [Scale(), Demean(), WithTuple(), ToEpochs()])
    def test_good_steps_pass(self, step, small_raw: BaseRaw) -> None:
        """Well-behaved steps pass every check."""
        check_step(step, small_raw)

    def test_dropping_a_channel_fails(self, small_raw: BaseRaw) -> None:
        """Silently dropping channels is caught."""
        with pytest.raises(AssertionError, match="channels changed"):
            check_step(DropsChannel(), small_raw)

    @pytest.mark.filterwarnings("ignore:Omitted .* annotation")
    def test_losing_first_samp_fails(self, small_raw: BaseRaw) -> None:
        """Rebuilding a Raw without first_samp is caught."""
        with pytest.raises(AssertionError, match="first_samp 1000 -> 0"):
            check_step(RebuildsRaw(), small_raw)

    def test_modifying_input_in_fit_fails(self, small_raw: BaseRaw) -> None:
        """Modifying the input while fitting is caught."""
        with pytest.raises(AssertionError, match="fit modified its input"):
            check_step(ModifiesInputInFit(), small_raw)
