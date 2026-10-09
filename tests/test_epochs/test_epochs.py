"""Epoch, Baseline, DropChannels and AutoReject against simulated ERPs."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

import meeg_utils as meu
from meeg_utils import steps as S
from meeg_utils.testing import check_step

from ..simulation import P300, make_erp


@pytest.fixture
def erp():
    return make_erp()


@pytest.fixture
def epochs(erp):
    return S.Epoch("stim", -0.2, 0.8, baseline=None).fit_transform(erp)


class TestContracts:
    @pytest.mark.parametrize(
        "step",
        [S.Epoch("stim", -0.2, 0.8), S.Epoch(None, -0.1, 0.5, baseline=None, decim=1)],
        ids=repr,
    )
    def test_epoch(self, step, erp):
        check_step(step, erp)

    @pytest.mark.parametrize(
        "step",
        [
            S.Baseline(),
            S.Baseline((-0.2, 0.0), "zscore", scale="pooled"),
            S.Baseline((-0.2, 0.0), "percent"),
            S.DropChannels(["Fp1", "Fp2"]),
            S.AutoReject("global", random_state=0),
        ],
        ids=repr,
    )
    def test_epochs_steps(self, step, epochs):
        check_step(step, epochs)

    def test_drop_channels_raw(self, erp):
        check_step(S.DropChannels(["EOG"]), erp)


class TestEpoch:
    def test_conditions_and_erp(self, erp):
        step = S.Epoch("stim", -0.2, 0.8)
        epochs = step.fit_transform(erp)
        assert set(step.event_id_) == {"stim/target", "stim/standard"}
        assert len(epochs) == 60
        assert sum(step.qc_["n_events"].values()) == 60
        assert step.qc_["fraction_kept"] == 1.0
        # epochs are cut at the right samples: minus the same recording without
        # the ERP, every epoch is exactly the simulated P300
        background = S.Epoch("stim", -0.2, 0.8).fit(erp).transform(make_erp(erp=False))
        for condition, amplitude in P300.items():
            diff = epochs[condition].get_data(["Pz"]) - background[condition].get_data(["Pz"])
            peak = diff[..., np.argmin(np.abs(epochs.times - 0.3))]
            np.testing.assert_allclose(peak, amplitude, rtol=0.02)

    def test_select_by_name_and_dict(self, erp):
        assert set(S.Epoch("target").fit(erp).event_id_) == {"stim/target"}
        assert S.Epoch({"stim/target": 7}).fit(erp).event_id_ == {"stim/target": 7}
        with pytest.raises(ValueError, match="No events match"):
            S.Epoch("response").fit(erp)

    def test_codes_are_fixed_at_fit(self, erp):
        step = S.Epoch("stim", -0.2, 0.8).fit(erp)
        other = make_erp(seed=1)
        targets = other.annotations.description == "stim/target"
        other.annotations.delete(np.flatnonzero(targets))
        out = step.transform(other)
        assert out.event_id == {"stim/standard": step.event_id_["stim/standard"]}

    def test_bad_annotations_drop_epochs(self, erp):
        erp.annotations.append(
            10.0,
            3.0,
            "BAD_movement",
        )
        step = S.Epoch("stim", -0.2, 0.8)
        epochs = step.fit_transform(erp)
        assert step.qc_["n_dropped"] > 0
        assert step.qc_["drop_reasons"] == {"BAD_movement": step.qc_["n_dropped"]}
        assert len(epochs) == 60 - step.qc_["n_dropped"]

    def test_stim_channel(self, erp):
        events, _ = mne.events_from_annotations(erp, verbose=False)
        stim = np.zeros((1, erp.n_times))
        stim[0, events[:, 0] - erp.first_samp] = events[:, 2]
        info = mne.create_info(["STI"], erp.info["sfreq"], "stim")
        erp.add_channels([mne.io.RawArray(stim, info, first_samp=erp.first_samp, verbose=False)],
                         force_update_info=True)  # fmt: skip
        step = S.Epoch(None, -0.1, 0.5, events="stim")
        epochs = step.fit_transform(erp)
        assert len(epochs) == 60
        assert set(step.event_id_) == {str(c) for c in np.unique(events[:, 2])}

    @pytest.mark.filterwarnings(
        "ignore:Converting data files to BrainVision format",
        "ignore:There are channels without locations:RuntimeWarning",
        "ignore:Not setting position of 1 eog channel:RuntimeWarning",
        'ignore:Encountered data in "double" format:RuntimeWarning',
    )
    def test_bids_metadata(self, erp, tmp_path):
        bids_path = BIDSPath(subject="01", task="oddball", datatype="eeg", root=tmp_path / "bids")
        write_raw_bids(erp, bids_path, allow_preload=True, format="BrainVision", verbose=False)
        raw = meu.io.read(bids_path)
        epochs = S.Epoch("stim", -0.2, 0.8, metadata="bids").fit_transform(raw)
        names = {code: name for name, code in epochs.event_id.items()}
        assert list(epochs.metadata["trial_type"]) == [names[c] for c in epochs.events[:, 2]]
        assert len(epochs["target"]) == (epochs.metadata["trial_type"] == "stim/target").sum()

    def test_invalid(self, erp):
        with pytest.raises(ValueError, match="tmin"):
            S.Epoch("stim", 0.5, 0.1).fit(erp)
        with pytest.raises(ValueError, match="events must be"):
            S.Epoch("stim", events="trigger").fit(erp)

    def test_plots(self, erp):
        import matplotlib.pyplot as plt

        step = S.Epoch("stim", -0.2, 0.8)
        step.fit_transform(erp)
        figs = step.plot()
        assert set(figs) == {"drop_log", "evoked"}
        plt.close("all")


class TestBaseline:
    def test_mean(self, epochs):
        out = S.Baseline((-0.2, 0.0)).fit_transform(epochs)
        sel = (out.times >= -0.2) & (out.times <= 0.0)
        np.testing.assert_allclose(out.get_data("eeg")[..., sel].mean(axis=-1), 0, atol=1e-12)

    def test_short_baseline_cannot_be_z_scored_per_epoch(self, epochs):
        """HAD-MEEG z-scored each epoch on 100 ms at 200 Hz (21 samples)."""
        short = epochs.copy().resample(200.0).crop(-0.1, None)
        with pytest.raises(ValueError, match=r"21 samples.*16% error"):
            S.Baseline((None, 0.0), "zscore").fit(short)
        # pooled over epochs, the scale is precise
        out = S.Baseline((None, 0.0), "zscore", scale="pooled").fit_transform(short)
        sel = out.times <= 0
        x = out.get_data("eeg")[..., sel]
        sd = (x - x.mean(axis=-1, keepdims=True)).std(axis=(0, 2), ddof=1)
        np.testing.assert_allclose(sd, 1.0, rtol=1e-6)

    def test_zscore_per_epoch_with_long_baseline(self, epochs):
        out = S.Baseline((-0.2, 0.0), "zscore").fit_transform(epochs)
        sel = (out.times >= -0.2) & (out.times <= 0.0)
        np.testing.assert_allclose(out.get_data("eeg")[..., sel].std(axis=-1), 1.0, rtol=1e-6)

    def test_evoked(self, epochs):
        evoked = epochs.average()
        out = S.Baseline((-0.2, 0.0)).fit_transform(evoked)
        sel = (out.times >= -0.2) & (out.times <= 0.0)
        np.testing.assert_allclose(out.data[:, sel].mean(axis=-1), 0, atol=1e-12)

    def test_invalid(self, epochs):
        with pytest.raises(ValueError, match="mode must be"):
            S.Baseline(mode="median").fit(epochs)
        with pytest.raises(ValueError, match="only applies to mode='zscore'"):
            S.Baseline(mode="mean", scale="pooled").fit(epochs)


class TestDropChannels:
    def test_missing(self, epochs):
        with pytest.raises(ValueError, match="M1"):
            S.DropChannels(["M1", "Fp1"]).fit(epochs)
        step = S.DropChannels(["M1", "Fp1"], on_missing="ignore")
        out = step.fit_transform(epochs)
        assert "Fp1" not in out.ch_names
        assert step.qc_ == {"dropped": ["Fp1"], "missing": ["M1"]}


class TestAutoReject:
    @pytest.fixture
    def dirty(self, epochs):
        """Epochs 3 and 7: artifacts on all channels; epochs 10-14: Cz jumps."""
        data = epochs._data
        data[[3, 7], :32] += 400e-6 * np.sin(np.linspace(0, 20, data.shape[-1]))
        data[10:15, epochs.ch_names.index("Cz")] += 300e-6
        data[10:15, epochs.ch_names.index("Cz"), : data.shape[-1] // 2] -= 300e-6
        return epochs

    def test_global_drops_artifact_epochs(self, dirty):
        step = S.AutoReject("global", random_state=0)
        out = step.fit_transform(dirty)
        dropped = set(range(len(dirty))) - set(out.selection)
        assert {3, 7} <= dropped
        assert step.qc_["n_dropped"] == len(dropped)

    def test_local_repairs_channels_and_drops_epochs(self, dirty):
        step = S.AutoReject("local", n_interpolate=[1, 4], cv=4, random_state=0)
        out = step.fit_transform(dirty)
        labels = step.reject_log_.labels
        cz = step.reject_log_.ch_names.index("Cz")
        assert step.reject_log_.bad_epochs[[3, 7]].all()
        assert (labels[10:15, cz] == 2).all()  # Cz interpolated, epochs kept
        assert not step.reject_log_.bad_epochs[10:15].any()
        assert len(out) == len(dirty) - step.qc_["n_dropped"]
        assert step.qc_["n_interpolated"] > 0

    def test_plots(self, dirty):
        import matplotlib.pyplot as plt

        for method in ("global", "local"):
            kwargs = {"n_interpolate": [1, 4], "cv": 3} if method == "local" else {}
            step = S.AutoReject(method, random_state=0, **kwargs).fit(dirty)
            assert set(step.plot()) == {"reject_log", "thresholds"}
            plt.close("all")
