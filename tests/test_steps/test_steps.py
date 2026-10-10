"""Tests for the processing steps: the step contract and scientific behaviour."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from mne.io import BaseRaw
from scipy.signal import welch

from meeg_utils import Pipeline, io
from meeg_utils import steps as S
from meeg_utils.testing import check_step

from ..simulation import EEG_CHANNELS, FLAT, FRONTAL, NOISY, SFREQ, make_eeg


def _band_power(raw: BaseRaw, picks, lo: float, hi: float) -> float:
    freqs, psd = welch(raw.get_data(picks=picks), fs=raw.info["sfreq"], nperseg=1000)
    band = (freqs >= lo) & (freqs <= hi)
    return float(psd[:, band].mean())


# ----------------------------------------------------------------------
# Contract


@pytest.mark.parametrize(
    "step",
    [
        S.Filter(1.0, 40.0),
        S.Filter(None, 30.0, method="iir"),
        S.Resample(100.0),
        S.LineNoise("zapline"),
        S.LineNoise("zapline-plus"),
        S.LineNoise("notch"),
        S.BadChannels("prep", ransac=False),
        S.Reference(),
        S.ICA(n_components=8, labeler=None, max_iter=200),
    ],
    ids=repr,
)
def test_step_contract(step, eeg: BaseRaw) -> None:
    """Every step honours the step contract."""
    check_step(step, eeg)


def test_interpolate_contract(eeg: BaseRaw) -> None:
    """Interpolate honours the contract on data with bad channels."""
    eeg.info["bads"] = [NOISY]
    check_step(S.Interpolate(), eeg)


# ----------------------------------------------------------------------
# Filter / Resample


class TestFilterResample:
    """Filter and Resample."""

    def test_filter_band(self, eeg: BaseRaw) -> None:
        """A 1-40 Hz band-pass removes 50 Hz and records the new band."""
        step = S.Filter(1.0, 40.0)
        out = step.fit_transform(eeg)
        assert step.qc_ == {"highpass": 1.0, "lowpass": 40.0}
        assert _band_power(out, "eeg", 48, 52) < 1e-3 * _band_power(eeg, "eeg", 48, 52)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"l_freq": None, "h_freq": None}, "l_freq and/or h_freq"),
            ({"l_freq": 1.0, "h_freq": 200.0}, "below the Nyquist"),
            ({"l_freq": 40.0, "h_freq": 1.0}, "must be below h_freq"),
        ],
    )
    def test_filter_validation(self, kwargs, match, eeg: BaseRaw) -> None:
        """Impossible filters are rejected before filtering."""
        with pytest.raises(ValueError, match=match):
            S.Filter(**kwargs).fit(eeg)

    def test_resample_keeps_timing(self, eeg: BaseRaw) -> None:
        """Resampling keeps the time of the first sample and of annotations."""
        out = S.Resample(100.0).fit_transform(eeg)
        assert out.info["sfreq"] == 100.0
        assert out.first_samp / 100.0 == pytest.approx(eeg.first_samp / SFREQ)
        np.testing.assert_allclose(out.annotations.onset, eeg.annotations.onset)


# ----------------------------------------------------------------------
# LineNoise


class TestLineNoise:
    """LineNoise."""

    @pytest.mark.parametrize(
        ("method", "tolerance_db"), [("zapline-plus", 0.5), ("zapline", 0.5), ("notch", 1.5)]
    )
    def test_removes_line_noise_without_distortion(
        self, method: str, tolerance_db: float, eeg: BaseRaw
    ) -> None:
        """The 50 Hz band returns to the line-free ground truth; 1-45 Hz is left intact.

        ZapLine removes only the line noise; a notch also removes the brain
        signal at 50 Hz, hence its wider tolerance.
        """
        step = S.LineNoise(method)
        out = step.fit_transform(eeg)
        qc = step.qc_["eeg"]
        assert step.fline_ == 50.0  # from info["line_freq"]
        assert qc["suppression_db"] > 10
        assert qc["distortion_db"] < 0.5
        assert qc["underclean_fraction"] == 0
        truth = make_eeg(line=False)  # same seed: identical except for the line noise
        residual_db = 10 * np.log10(
            _band_power(out, "eeg", 48, 52) / _band_power(truth, "eeg", 48, 52)
        )
        assert abs(residual_db) < tolerance_db
        before, after = _band_power(eeg, "eeg", 8, 12), _band_power(out, "eeg", 8, 12)
        assert after == pytest.approx(before, rel=0.05)  # alpha preserved
        np.testing.assert_array_equal(out.get_data("EOG"), eeg.get_data("EOG"))

    def test_unknown_line_frequency_raises(self, eeg: BaseRaw) -> None:
        """No line frequency in info and none given: refuse to guess."""
        eeg.info["line_freq"] = None
        with pytest.raises(ValueError, match="power line frequency is unknown"):
            S.LineNoise().fit(eeg)
        assert S.LineNoise(fline=50.0).fit(eeg).fline_ == 50.0

    def test_bad_channels_are_left_alone(self, eeg: BaseRaw) -> None:
        """Channels marked bad are neither used nor modified."""
        eeg.info["bads"] = [NOISY]
        out = S.LineNoise("zapline").fit_transform(eeg)
        np.testing.assert_array_equal(out.get_data(NOISY), eeg.get_data(NOISY))

    def test_zapline_transform_reuses_filters(self, eeg: BaseRaw) -> None:
        """Standard ZapLine applies the filters learned in fit to new data."""
        step = S.LineNoise("zapline").fit(eeg)
        other = make_eeg(seed=1)
        out = step.transform(other)
        assert _band_power(out, "eeg", 48, 52) < 0.05 * _band_power(other, "eeg", 48, 52)

    def test_channel_types_processed_separately(self, neuromag: BaseRaw) -> None:
        """MEG magnetometers, gradiometers and EEG are cleaned separately."""
        neuromag.info["line_freq"] = 60.0
        step = S.LineNoise("zapline")
        out = step.fit_transform(neuromag)
        assert set(step.qc_) == {"mag", "grad", "eeg"}
        assert out.ch_names == neuromag.ch_names
        np.testing.assert_array_equal(out.get_data("stim"), neuromag.get_data("stim"))


# ----------------------------------------------------------------------
# BadChannels / Interpolate


class TestBadChannels:
    def test_electrode_next_to_reference_is_not_bad(self):
        """Recorded against a reference next to Cz, Cz carries almost no signal. On the
        recording's own reference PREP's correlation criterion flags it; after PREP's
        robust average reference (the default) it is a normal channel."""
        raw = make_eeg(blinks=False)
        rng = np.random.default_rng(0)
        cz = raw.ch_names.index("Cz")
        raw._data[:32] -= raw._data[cz].copy()
        raw._data[cz] += rng.normal(0, 0.4e-6, raw.n_times)
        assert S.BadChannels("prep", robust_reference=False).fit(raw).bads_ == ["Cz"]
        step = S.BadChannels("prep").fit(raw)
        assert step.bads_ == []
        assert step.qc_["eeg_before_reference"] == ["Cz"]

    """BadChannels and Interpolate."""

    def test_psd_criterion_is_opt_in(self, eeg_bad: BaseRaw) -> None:
        """PyPREP's band-power criterion is not PREP's: off unless asked, and PyPREP is
        left as it was."""
        from pyprep.find_noisy_channels import NoisyChannels

        original = NoisyChannels.find_bad_by_PSD
        step = S.BadChannels("prep", ransac=False).fit(eeg_bad)
        assert "PREP psd" not in step.scores_
        assert step.qc_["eeg"]["by_criterion"]["psd"] == []
        assert NoisyChannels.find_bad_by_PSD is original
        step = S.BadChannels("prep", ransac=False, psd=True).fit(eeg_bad)
        assert "PREP psd" in step.scores_
        assert NoisyChannels.find_bad_by_PSD is original

    @pytest.mark.parametrize("ransac", [False, True])
    def test_prep_finds_noisy_and_flat(self, ransac: bool, eeg_bad: BaseRaw) -> None:
        """PREP flags the simulated noisy and flat channels, and only marks them."""
        step = S.BadChannels("prep", ransac=ransac)
        out = step.fit_transform(eeg_bad)
        assert set(step.bads_) == {NOISY, FLAT}  # and no false positives
        assert set(out.info["bads"]) == set(step.bads_)
        np.testing.assert_array_equal(out.get_data(), eeg_bad.get_data())  # data untouched
        assert "flat" in step.qc_["eeg"]["by_criterion"]

    def test_existing_bads_are_kept(self, eeg_bad: BaseRaw) -> None:
        """Channels already marked bad stay bad and are not reported as new."""
        eeg_bad.info["bads"] = ["Fz"]
        step = S.BadChannels("prep", ransac=False)
        out = step.fit_transform(eeg_bad)
        assert "Fz" in out.info["bads"]
        assert "Fz" not in step.bads_
        assert step.qc_["previous_bads"] == ["Fz"]

    def test_ransac_needs_positions(self, eeg_bad: BaseRaw) -> None:
        """RANSAC without electrode positions fails loudly."""
        eeg_bad.set_montage(None)
        with pytest.raises(ValueError, match="electrode positions"):
            S.BadChannels("prep", ransac=True).fit(eeg_bad)

    def test_maxwell_on_neuromag(self, neuromag: BaseRaw) -> None:
        """Maxwell detection finds MNE sample data's known bad channel, MEG 2443."""
        step = S.BadChannels("maxwell", cross_talk=None, calibration=None)
        step.fit(neuromag)
        assert "MEG 2443" in step.bads_
        assert step.qc_["meg"]["fraction"] < 0.05
        assert step.qc_["maxwell_files"] == {"cross_talk": None, "calibration": None}
        # the scores figure shows what decided: detected channels at or above the line
        for score in step.scores_.values():
            for ch, value in zip(score["ch_names"], score["values"], strict=True):
                if ch in step.qc_["meg"]["by_criterion"]["noisy"]:
                    assert value >= score["threshold"]
                else:
                    assert value < score["threshold"]

    def test_maxwell_files_must_be_found(self, neuromag: BaseRaw) -> None:
        """cross_talk='auto' on a recording outside BIDS fails loudly."""
        with pytest.raises(FileNotFoundError, match="not in a BIDS dataset"):
            S.BadChannels("maxwell").fit(neuromag)

    def test_unsupported_meg_system_raises(self) -> None:
        """MEG systems without a validated default fail loudly."""
        info = mne.create_info(4, 250.0, "mag")
        with info._unlock():
            for ch in info["chs"]:
                ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_KIT_GRAD
        raw = mne.io.RawArray(np.zeros((4, 2500)), info, verbose=False)
        with pytest.raises(NotImplementedError, match="'kit'"):
            S.BadChannels().fit(raw)

    def test_interpolate_repairs_and_resets(self, eeg_bad: BaseRaw) -> None:
        """Interpolation replaces the noisy channel with a plausible signal."""
        eeg_bad.info["bads"] = [NOISY]
        step = S.Interpolate()
        out = step.fit_transform(eeg_bad)
        assert out.info["bads"] == []
        assert step.qc_["interpolated"] == [NOISY]
        clean = make_eeg()  # same seed, without the added noise
        r = np.corrcoef(out.get_data(NOISY)[0], clean.get_data(NOISY)[0])[0, 1]
        assert r > 0.8
        assert out.get_data(NOISY).std() < 3 * clean.get_data(NOISY).std()


# ----------------------------------------------------------------------
# Reference


class TestReference:
    """Reference."""

    def test_average_reference(self, eeg: BaseRaw) -> None:
        """After average referencing, the EEG channels sum to zero."""
        step = S.Reference()
        out = step.fit_transform(eeg)
        np.testing.assert_allclose(out.get_data("eeg").mean(axis=0), 0, atol=1e-18)
        np.testing.assert_array_equal(out.get_data("EOG"), eeg.get_data("EOG"))
        assert step.qc_ == {"eeg_reference": "average"}

    def test_average_reference_excludes_bads(self, eeg_bad: BaseRaw) -> None:
        """Bad channels do not enter the average reference."""
        eeg_bad.info["bads"] = [NOISY]
        out = S.Reference().fit_transform(eeg_bad)
        good = [ch for ch in EEG_CHANNELS if ch != NOISY]
        np.testing.assert_allclose(out.get_data(good).mean(axis=0), 0, atol=1e-18)

    def test_no_ctf_compensation_for_other_systems(self, neuromag: BaseRaw) -> None:
        """Gradient compensation only applies to CTF; Neuromag MEG is untouched."""
        step = S.Reference()
        out = step.fit_transform(neuromag)
        assert "ctf_grade_after" not in step.qc_
        np.testing.assert_array_equal(out.get_data("meg"), neuromag.get_data("meg"))


# ----------------------------------------------------------------------
# ICA


class TestICA:
    """ICA."""

    @pytest.mark.filterwarnings("ignore:FastICA did not converge")
    def test_rank_caps_n_components(self, eeg: BaseRaw) -> None:
        """An integer n_components above the rank (after average reference) is lowered."""
        ref = S.Reference().fit_transform(eeg.crop(0, 20))
        step = S.ICA(n_components=40, method="fastica", labeler=None, max_iter=50)
        step.fit(ref)
        assert step.qc_["rank"] == len(EEG_CHANNELS) - 1
        assert step.ica_.n_components_ == len(EEG_CHANNELS) - 1
        assert "lowered" in step.qc_["note"]

    def test_manual_relabel_removes_blinks(self, eeg: BaseRaw) -> None:
        """Excluding the blink component removes blinks and keeps alpha."""
        data = S.Filter(1.0, 40.0).fit_transform(eeg)
        step = S.ICA(n_components=15, labeler=None).fit(data)
        assert step.exclude_ == []
        sources = step.ica_.get_sources(data).get_data()
        eog = data.get_data("EOG")[0]
        blink_ic = int(np.argmax([abs(np.corrcoef(s, eog)[0, 1]) for s in sources]))

        step.relabel({blink_ic: "eye blink"}, inst=data)
        assert step.exclude_ == [blink_ic]
        assert step.qc_["excluded_variance"]["eeg"] > 0.1
        out = step.transform(data)

        frontal_r = abs(np.corrcoef(out.get_data(FRONTAL).mean(0), eog)[0, 1])
        assert frontal_r < 0.3
        alpha = _band_power(out, ["O1", "Oz", "O2"], 9, 11) / _band_power(
            data, ["O1", "Oz", "O2"], 9, 11
        )
        assert alpha > 0.8

    def test_threshold(self, eeg: BaseRaw) -> None:
        """Artifact labels below the probability threshold are not excluded."""
        step = S.ICA(n_components=5, labeler=None, max_iter=200, threshold=0.9).fit(eeg)
        step.labeler_ = "iclabel"
        step.labels_ = ["eye blink", "eye blink", "brain", "muscle artifact", "other"]
        step.proba_ = np.array([0.95, 0.6, 0.99, 0.91, 0.99])
        step._update_exclude(None)
        assert step.exclude_ == [0, 3]

    def test_iclabel_runs(self, eeg: BaseRaw) -> None:
        """ICLabel labels every component of a properly prepared EEG decomposition."""
        ref = S.Reference().fit_transform(eeg)
        step = S.ICA(n_components=10).fit(ref)
        assert step.labeler_ == "iclabel"
        assert len(step.labels_) == 10
        assert all(0 <= p <= 1 for p in step.proba_)
        assert set(step.exclude_) <= set(range(10))

    def test_meg_plus_eeg_needs_explicit_picks(self, neuromag: BaseRaw) -> None:
        """For MEG+EEG recordings, ICA must be told which modality to decompose."""
        with pytest.raises(ValueError, match="one ICA step per modality"):
            S.ICA(labeler=None).fit(neuromag)
        step = S.ICA(n_components=5, picks="eeg", labeler=None, max_iter=100).fit(neuromag)
        assert step.ica_.ch_names == neuromag.copy().pick("eeg", exclude="bads").ch_names

    @pytest.mark.filterwarnings("ignore:FastICA did not converge")
    def test_neuromag_components_figure(self, neuromag: BaseRaw) -> None:
        """MNE titles Neuromag components "ICA000 (mag)"; the labels are still added."""
        step = S.ICA(n_components=5, picks="meg", method="fastica", labeler=None, max_iter=50)
        step.fit(neuromag)
        figs = step.plot("components")["components"]
        titles = [ax.get_title() for fig in np.atleast_1d(figs) for ax in fig.axes]
        assert any(t.startswith("ICA000") and "\n" in t for t in titles)
        import matplotlib.pyplot as plt

        plt.close("all")

    @pytest.mark.filterwarnings("ignore:FastICA did not converge")
    def test_rank_after_sss(self, neuromag: BaseRaw) -> None:
        """After SSS the data rank can exceed the SSS rank (a later step adds signal outside
        the SSS subspace, as ZapLine does); ICA is capped at the rank in info."""
        meg = neuromag.copy().pick("meg")
        sss = S.Maxwell(cross_talk=None, calibration=None).fit_transform(meg)
        sss._data += np.random.default_rng(0).normal(0, 1e-15, sss._data.shape)
        step = S.ICA(n_components=200, method="fastica", labeler=None, max_iter=20).fit(sss)
        assert step.qc_["rank"] == sum(mne.compute_rank(sss, rank="info").values())


# ----------------------------------------------------------------------
# End to end


def test_eeg_pipeline_end_to_end(eeg_bad: BaseRaw, tmp_path) -> None:
    """A full EEG pipeline runs, keeps all channels and timing, and is saved with provenance."""
    pipe = Pipeline(
        [
            ("filter", S.Filter(0.1, 100.0)),
            ("line", S.LineNoise("zapline")),
            ("bads", S.BadChannels("prep", ransac=False)),
            ("interp", S.Interpolate()),
            ("ref", S.Reference()),
            ("ica", S.ICA(n_components=15, max_iter=300)),
        ]
    )
    out = pipe.fit_transform(eeg_bad)

    assert out.ch_names == eeg_bad.ch_names
    assert out.first_samp == eeg_bad.first_samp
    np.testing.assert_allclose(out.annotations.onset, eeg_bad.annotations.onset)
    assert set(pipe.qc_["bads"]["eeg"]["bads"]) == {NOISY, FLAT}
    assert pipe.qc_["interp"]["interpolated"]
    assert pipe.qc_["line"]["eeg"]["suppression_db"] > 15  # before bad-channel removal
    assert np.isfinite(pipe.qc_["line"]["eeg"]["distortion_db"])  # flat channel skipped
    assert pipe.qc_["line"]["eeg"]["n_flat_skipped"] == 1
    assert [s["name"] for s in pipe.provenance_["steps"]] == [n for n, _ in pipe.steps]

    fname = io.save_derivative(out, "/data/sub-01_task-rest_eeg.vhdr", tmp_path, pipeline=pipe)
    back = mne.io.read_raw_fif(fname, verbose=False)
    assert back.ch_names == out.ch_names


# ----------------------------------------------------------------------
# Figures


class TestPlotInterface:
    """The plot() interface shared by steps and pipelines."""

    def test_errors(self, eeg: BaseRaw) -> None:
        """Unfitted steps, unknown kinds, missing data and stray options are rejected."""
        from sklearn.exceptions import NotFittedError

        step = S.ICA(n_components=5, labeler=None, max_iter=100)
        with pytest.raises(NotFittedError):
            step.plot()
        step.fit(eeg)
        with pytest.raises(ValueError, match="no 'nope' plot; available"):
            step.plot("nope")
        with pytest.raises(ValueError, match="needs inst=<data>"):
            step.plot("overlay")
        with pytest.raises(TypeError, match="only be passed together with kind"):
            step.plot(colorbar=True)
        assert set(step.plot("components", colorbar=True)) == {"components"}

    def test_steps_without_figures(self, eeg: BaseRaw) -> None:
        """Steps without figures return an empty dict."""
        assert S.Reference().fit(eeg).plot() == {}

    def test_pipeline_plot_gives_each_step_its_input(self, eeg_bad: BaseRaw) -> None:
        """pipe.plot(inst=...) replays the fitted steps so each plots its own input."""
        pipe = Pipeline(
            [
                ("bads", S.BadChannels("prep", ransac=False)),
                ("interp", S.Interpolate()),
                ("ica", S.ICA(n_components=6, labeler=None, max_iter=100)),
            ]
        )
        pipe.fit(eeg_bad)
        assert set(pipe.plot()) == {"bads", "interp", "ica"}  # fit transforms all but the last
        figures = pipe.plot(inst=eeg_bad)
        assert set(figures["ica"]) == {"components", "labels", "properties", "overlay"}
        assert set(figures["interp"]) == {"sensors"}
        import matplotlib.pyplot as plt

        plt.close("all")
