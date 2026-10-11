"""Tests for Pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from mne.epochs import BaseEpochs
from mne.io import BaseRaw
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from meeg_utils.core import Pipeline

from ._steps import CTFOnly, Demean, Recorder, Scale, ToEpochs, Warns, WithTuple


@pytest.fixture
def pipe() -> Pipeline:
    """A two-step pipeline."""
    return Pipeline([("scale", Scale(2.0)), ("demean", Demean())])


class TestFitTransform:
    """fit / transform / fit_transform."""

    def test_html_repr_lists_steps(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """Notebooks and the docs show one row per step, not scikit-learn's diagram."""
        html = pipe._repr_html_()
        assert all(f"<b>{name}</b>" in html for name, _ in pipe.steps)
        assert "fitted" not in html
        pipe.fit(small_raw)
        assert "fitted" in pipe._repr_html_()
        assert pipe._repr_mimebundle_()["text/html"] == pipe._repr_html_()

    def test_fit_transform_chains_steps(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """Each step is fitted on the output of the previous one."""
        out = pipe.fit_transform(small_raw)
        scaled = small_raw.get_data(picks="eeg") * 2.0
        expected = scaled - scaled.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(out.get_data(picks="eeg"), expected, atol=1e-20)
        np.testing.assert_allclose(pipe.named_steps["demean"].means_, scaled.mean(1, keepdims=True))

    def test_fit_transforms_all_but_last(self, small_raw: BaseRaw) -> None:
        """fit() fits every step but only transforms up to the last one."""
        Recorder.calls.clear()
        Pipeline([("a", Recorder("a")), ("b", Recorder("b"))]).fit(small_raw)
        assert Recorder.calls == ["fit:a", "transform:a", "fit:b"]

    def test_transform_reuses_fitted_state(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """transform() applies the state learned in fit() to new data."""
        pipe.fit(small_raw)
        other = small_raw.copy()
        other._data += 1e-3
        out = pipe.transform(other)
        expected = other.get_data(picks="eeg") * 2.0 - pipe["demean"].means_
        np.testing.assert_allclose(out.get_data(picks="eeg"), expected)

    def test_transform_before_fit_raises(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """An unfitted pipeline cannot transform."""
        with pytest.raises(NotFittedError):
            pipe.transform(small_raw)

    def test_transform_rejects_other_system(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """A pipeline fitted on one system refuses data from another."""
        import mne

        pipe.fit(small_raw)
        meg = mne.io.RawArray(np.zeros((2, 100)), mne.create_info(2, 100.0, "mag"), verbose=False)
        with pytest.raises(ValueError, match="fitted on 'eeg' data"):
            pipe.transform(meg)

    def test_copy_true_leaves_input_untouched(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """By default the input is never modified, by fit or transform."""
        before = small_raw.get_data().copy()
        pipe.fit(small_raw)
        pipe.transform(small_raw)
        pipe.fit_transform(small_raw)
        np.testing.assert_array_equal(small_raw.get_data(), before)

    def test_copy_false_works_in_place(self, small_raw: BaseRaw) -> None:
        """With copy=False the input is processed in place."""
        before = small_raw.get_data().copy()
        out = Pipeline([("scale", Scale(2.0))], copy=False).fit_transform(small_raw)
        assert out is small_raw
        np.testing.assert_allclose(small_raw.get_data(), before * 2.0)

    def test_data_type_can_change(self, small_raw: BaseRaw) -> None:
        """Steps may turn Raw into Epochs."""
        out = Pipeline([("demean", Demean()), ("epochs", ToEpochs(2.0))]).fit_transform(small_raw)
        assert isinstance(out, BaseEpochs)
        assert len(out) == 10

    def test_type_mismatch_fails_before_running(self, small_raw: BaseRaw) -> None:
        """An impossible type chain is rejected before any step runs."""
        Recorder.calls.clear()
        bad = Pipeline([("rec", Recorder()), ("epochs", ToEpochs()), ("scale", Scale())])
        with pytest.raises(TypeError, match=r"'scale' .* accepts BaseRaw, but receives BaseEpochs"):
            bad.fit(small_raw)
        assert Recorder.calls == []

    def test_system_restriction_is_enforced(self, small_raw: BaseRaw) -> None:
        """System-specific steps fail loudly on other systems."""
        with pytest.raises(ValueError, match="applies to"):
            Pipeline([("ctf", CTFOnly())]).fit(small_raw)

    @pytest.mark.parametrize(
        ("steps", "error", "match"),
        [
            ([("a", Scale()), ("a", Scale())], ValueError, "unique"),
            ([("a__b", Scale())], ValueError, "'__'"),
            ([("a", object())], TypeError, "must be a meeg_utils Step"),
        ],
    )
    def test_invalid_steps(self, steps, error, match, small_raw: BaseRaw) -> None:
        """Invalid step lists are rejected."""
        with pytest.raises(error, match=match):
            Pipeline(steps).fit(small_raw)


class TestSideOutputs:
    """qc_ and provenance_."""

    def test_qc_collects_step_metrics(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """qc_ holds each step's metrics, keyed by step name."""
        pipe.fit(small_raw)
        assert pipe.qc_["scale"] == {}
        assert pipe.qc_["demean"]["max_abs_mean"] > 0

    def test_provenance(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """provenance_ records versions, system and step details."""
        pipe.fit(small_raw)
        prov = pipe.provenance_
        assert prov["system"] == "eeg"
        assert {"python", "mne", "meeg_utils"} <= prov["versions"].keys()
        assert [s["name"] for s in prov["steps"]] == ["scale", "demean"]
        assert prov["steps"][0]["class"].endswith(":Scale")
        assert prov["steps"][0]["params"] == {"factor": 2.0}
        assert prov["steps"][0]["duration_s"] >= 0

    def test_warnings_are_recorded_and_still_emitted(self, small_raw: BaseRaw) -> None:
        """Warnings raised by a step are recorded and still reach the user."""
        pipe = Pipeline([("warns", Warns())])
        with pytest.warns(RuntimeWarning, match="something looks off"):
            pipe.fit(small_raw)
        assert pipe.provenance_["steps"][0]["warnings"] == ["RuntimeWarning: something looks off"]


class TestParams:
    """Nested parameters, clone and editing."""

    def test_get_params_is_nested(self, pipe: Pipeline) -> None:
        """Deep parameters include each step and its parameters."""
        params = pipe.get_params()
        assert params["scale"] is pipe.steps[0][1]
        assert params["scale__factor"] == 2.0
        assert params["demean__picks"] == ("eeg",)
        assert "scale" not in pipe.get_params(deep=False)

    def test_set_nested_param(self, pipe: Pipeline) -> None:
        """<step>__<param> sets a step parameter."""
        pipe.set_params(scale__factor=10.0, copy=False)
        assert pipe["scale"].factor == 10.0
        assert pipe.copy is False

    def test_set_params_replaces_step(self, pipe: Pipeline) -> None:
        """<step>=new_step replaces a step."""
        new = Scale(4.0)
        pipe.set_params(scale=new)
        assert pipe["scale"] is new

    def test_set_unknown_param_raises(self, pipe: Pipeline) -> None:
        """Unknown parameters are rejected."""
        with pytest.raises(ValueError):
            pipe.set_params(scale__nope=1)

    def test_clone_gives_unfitted_independent_copy(
        self, pipe: Pipeline, small_raw: BaseRaw
    ) -> None:
        """clone() copies parameters but not fitted state."""
        pipe.fit(small_raw)
        cloned = clone(pipe)
        assert cloned["scale"] is not pipe["scale"]
        assert cloned["scale"].factor == 2.0
        assert not hasattr(cloned, "system_")
        assert not hasattr(cloned["demean"], "means_")

    def test_indexing_and_slicing(self, pipe: Pipeline, small_raw: BaseRaw) -> None:
        """Steps are reachable by position and name; slices are pipelines."""
        assert pipe[0] is pipe["scale"]
        assert len(pipe) == 2
        head = pipe[:1]
        assert isinstance(head, Pipeline)
        assert head.steps == pipe.steps[:1]
        out = head.fit_transform(small_raw)
        np.testing.assert_allclose(out.get_data(), small_raw.get_data() * 2.0)

    def test_slice_of_fitted_pipeline_can_transform(
        self, pipe: Pipeline, small_raw: BaseRaw
    ) -> None:
        """A slice shares the fitted steps and can transform directly."""
        pipe.fit(small_raw)
        out = pipe[:1].transform(small_raw)
        np.testing.assert_allclose(out.get_data(), small_raw.get_data() * 2.0)

    def test_editing(self, pipe: Pipeline) -> None:
        """Steps can be inserted, replaced and removed by name."""
        pipe.insert_after("scale", "tuple", WithTuple())
        pipe.insert_before("scale", "first", Scale(1.0))
        assert [n for n, _ in pipe.steps] == ["first", "scale", "tuple", "demean"]
        pipe.replace("tuple", Scale(3.0)).remove("first")
        assert [n for n, _ in pipe.steps] == ["scale", "tuple", "demean"]
        assert pipe["tuple"].factor == 3.0
        with pytest.raises(KeyError, match="No step named 'nope'"):
            pipe.remove("nope")


class TestSerialization:
    """Configuration round trips."""

    def test_dict_round_trip(self, small_raw: BaseRaw) -> None:
        """to_dict/from_dict reproduce the configuration."""
        pipe = Pipeline([("scale", Scale(3.0)), ("tuple", WithTuple((0.5, 30.0)))], copy=False)
        config = pipe.to_dict()
        assert config["steps"][1] == {
            "name": "tuple",
            "class": "tests.test_core._steps:WithTuple",
            "params": {"band": [0.5, 30.0]},
        }
        restored = Pipeline.from_dict(config)
        assert restored.copy is False
        assert [n for n, _ in restored.steps] == ["scale", "tuple"]
        assert restored["scale"].factor == 3.0
        np.testing.assert_allclose(  # copy=False: give each pipeline its own input
            restored.fit_transform(small_raw.copy()).get_data(),
            pipe.fit_transform(small_raw.copy()).get_data(),
        )

    def test_yaml_round_trip(self, pipe: Pipeline, tmp_path) -> None:
        """to_yaml/from_yaml reproduce the configuration."""
        fname = tmp_path / "config.yaml"
        pipe.to_yaml(fname)
        restored = Pipeline.from_yaml(fname)
        assert restored.get_params(deep=False)["copy"] is True
        assert restored["scale"].factor == 2.0
        assert list(restored["demean"].picks) == ["eeg"]

    def test_from_dict_rejects_non_steps(self) -> None:
        """Only Step subclasses can be instantiated from a configuration."""
        config = {"steps": [{"name": "x", "class": "builtins:dict", "params": {}}]}
        with pytest.raises(TypeError, match="not a meeg_utils Step"):
            Pipeline.from_dict(config)

    def test_unserializable_param_raises(self) -> None:
        """Parameters that are not plain data cannot be serialized."""
        with pytest.raises(TypeError, match="cannot be serialized"):
            Pipeline([("scale", Scale(object()))]).to_dict()  # type: ignore[arg-type]
