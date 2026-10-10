"""Second-batch acquisition checks, against recordings with known defects."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

import meeg_utils as meu
from meeg_utils import qc
from meeg_utils.testing import check_check

from ..simulation import make_chpi, make_eeg, make_erp


def _levels(check: qc.Check) -> dict[str, str]:
    return {f.metric: f.level for f in check.findings_}


class TestSquidJumps:
    def test_finds_steps_not_spikes(self, neuromag):
        raw = neuromag.copy()
        sfreq = raw.info["sfreq"]
        jump_channels = ["MEG 0113", "MEG 1642"]
        for ch in jump_channels:
            i = raw.ch_names.index(ch)
            raw._data[i, int(3 * sfreq) :] += 50 * raw._data[i].std()  # lasting step
        spike = raw.ch_names.index("MEG 0122")
        raw._data[spike, int(5 * sfreq)] += 50 * raw._data[spike].std()  # one sample
        check = qc.SquidJumps().compute(raw)
        assert sorted(check.metrics_["channels"]) == jump_channels
        assert check.metrics_["times"][0] == pytest.approx(3.0, abs=0.01)
        assert check.level == "warn"
        check_check(qc.SquidJumps(), raw)

    def test_clean_recording(self, neuromag):
        check = qc.SquidJumps().compute(neuromag)
        assert check.metrics_["n_jumps"] == 0 and check.level == "ok"

    def test_not_for_eeg(self, eeg):
        assert qc.SquidJumps().not_applicable(eeg) is not None


class TestChpiSNR:
    @pytest.fixture(scope="class")
    def chpi(self, neuromag_fname):
        return make_chpi(neuromag_fname)[0]

    def test_coils_ok(self, chpi):
        check = qc.ChpiSNR().compute(chpi)
        assert len(check.metrics_["mag"]) == 4
        assert check.level == "info"
        assert qc.ChpiSNR(max_drop_db=10).compute(chpi).level == "ok"
        check_check(qc.ChpiSNR(), chpi)

    def test_silent_coil(self, chpi):
        raw = chpi.copy()
        with raw.info._unlock():  # the 4th coil "drives" a frequency with no signal
            raw.info["hpi_meas"][0]["hpi_coils"][3]["coil_freq"] = 77.0
        check = qc.ChpiSNR(max_drop_db=10).compute(raw)
        assert _levels(check)["mag_coil4_drop_db"] == "warn"
        assert _levels(check)["mag_coil1_drop_db"] == "ok"

    def test_needs_chpi(self, neuromag):
        assert "cHPI" in qc.ChpiSNR().not_applicable(neuromag)


class TestPhotodiode:
    @pytest.fixture
    def recording(self):
        """Oddball EEG with a photodiode: 20 ms delay, 1 ms jitter, one stimulus not shown."""
        raw = make_erp()
        rng = np.random.default_rng(0)
        sfreq = raw.info["sfreq"]
        pd = np.zeros(raw.n_times)
        onsets = raw.annotations.onset - raw.first_time
        for k, onset in enumerate(onsets):
            if k == 5:
                continue
            start = round((onset + 0.020 + rng.normal(0, 0.001)) * sfreq)
            pd[start : start + int(0.1 * sfreq)] = 1.0
        raw.add_channels(
            [mne.io.RawArray(pd[None], mne.create_info(["PD"], sfreq, "misc"), verbose=False)],
            force_update_info=True,
        )
        return raw

    def test_delay_and_jitter(self, recording):
        check = qc.Photodiode("PD", max_jitter_ms=5.0).compute(recording)
        assert check.metrics_["delay_ms"] == pytest.approx(20, abs=4)  # 4 ms samples
        assert check.metrics_["jitter_ms"] < 4
        assert check.metrics_["n_triggers"] - check.metrics_["n_matched"] == 1
        assert _levels(check) == {"n_unmatched": "warn", "jitter_ms": "ok"}
        check_check(qc.Photodiode("PD"), recording)

    def test_missing_channel(self, recording):
        assert "photodiode" in qc.Photodiode("nope").not_applicable(recording)


@pytest.mark.filterwarnings(
    "ignore:Converting data files to BrainVision format",
    "ignore:There are channels without locations",
    "ignore:Not setting position",
    "ignore:Encountered data in",
)
class TestBidsMetadata:
    @pytest.fixture
    def bids_path(self, tmp_path):
        path = BIDSPath(subject="01", task="rest", datatype="eeg", root=tmp_path)
        write_raw_bids(make_eeg(), path, allow_preload=True, format="BrainVision", verbose=False)
        return path.copy().update(suffix="eeg", extension=".vhdr")

    def _sidecar(self, bids_path) -> Path:
        return Path(str(bids_path.copy().update(extension=".json").fpath))

    def test_consistent(self, bids_path):
        check = qc.BidsMetadata().compute(meu.io.read(bids_path))
        assert check.metrics_["mismatches"] == {} and check.level == "ok"

    def test_wrong_sidecar(self, bids_path):
        fname = self._sidecar(bids_path)
        sidecar = json.loads(fname.read_text())
        sidecar["SamplingFrequency"] = 500
        sidecar["EEGChannelCount"] = 64
        del sidecar["PowerLineFrequency"]
        fname.write_text(json.dumps(sidecar))
        raw = mne.io.read_raw(str(bids_path.fpath), preload=True, verbose=False)
        check = qc.BidsMetadata().compute(raw)
        assert set(check.metrics_["mismatches"]) == {"SamplingFrequency", "EEGChannelCount"}
        assert check.metrics_["missing"] == ["PowerLineFrequency"]
        assert _levels(check)["SamplingFrequency"] == "fail"

    def test_not_bids(self, eeg):
        assert "BIDS" in qc.BidsMetadata().not_applicable(eeg)


class TestEmptyRoom:
    @pytest.fixture
    def empty_room(self, neuromag):
        """An empty room: sensor noise only; one magnetometer 10x noisier."""
        er = neuromag.copy().pick(["meg"])
        rng = np.random.default_rng(0)
        for ch_type, level in (("mag", 3e-15), ("grad", 3e-13)):
            picks = mne.pick_types(er.info, meg=ch_type)
            er._data[picks] = rng.normal(
                0, level * np.sqrt(er.info["sfreq"] / 2), (len(picks), er.n_times)
            )
        noisy = er.ch_names.index("MEG 0111")
        er._data[noisy] *= 10
        return er

    def test_noise_floor_and_noisy_sensor(self, neuromag, empty_room):
        check = qc.EmptyRoom(empty_room=empty_room).compute(neuromag)
        assert check.metrics_["mag"]["noise_floor"] == pytest.approx(3.0, rel=0.1)  # fT/√Hz
        assert check.metrics_["grad"]["noise_floor"] == pytest.approx(3.0, rel=0.1)  # fT/cm/√Hz
        assert check.metrics_["mag"]["outlier_channels"] == ["MEG 0111"]
        assert check.metrics_["grad"]["outlier_channels"] == []
        assert check.metrics_["mag"]["recording_above_db"] > 10
        assert _levels(check)["mag_outlier_channels"] == "warn"
        check_check(qc.EmptyRoom(empty_room=empty_room), neuromag)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_found_in_bids(self, neuromag, empty_room, tmp_path):
        recording = BIDSPath(subject="01", task="faces", datatype="meg", root=tmp_path)
        write_raw_bids(neuromag, recording, allow_preload=True, format="FIF", verbose=False)
        date = neuromag.info["meas_date"].strftime("%Y%m%d")
        er_path = BIDSPath(
            subject="emptyroom", session=date, task="noise", datatype="meg", root=tmp_path
        )
        write_raw_bids(empty_room, er_path, allow_preload=True, format="FIF", verbose=False)
        raw = meu.io.read(recording.copy().update(suffix="meg", extension=".fif"))
        check = qc.EmptyRoom()
        assert check.not_applicable(raw) is None
        check.compute(raw)
        assert "sub-emptyroom" in check.metrics_["empty_room"]
        assert check.metrics_["days_apart"] < 1
        assert qc.EmptyRoom().not_applicable(neuromag) is not None  # not in BIDS
