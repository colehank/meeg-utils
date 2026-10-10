"""
.. _tut-epochs:

Epochs, artifact rejection and combining runs
=============================================

Preprocessing works on each run separately. Epoching is a second stage,
applied to each preprocessed run; the epochs of a session's runs are then
combined. This tutorial cuts MNE's ``sample`` EEG into epochs around the
auditory and visual stimuli, repairs or rejects the bad ones with
autoreject, and combines two runs.
"""

# sphinx_gallery_thumbnail_number = 5
# sphinx_gallery_multi_image = "single"

# %%
import mne

import meeg_utils as meu
from meeg_utils import steps as S

mne.set_log_level("ERROR")

data_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(data_path / "MEG" / "sample" / "sample_audvis_raw.fif")
raw.pick(["eeg", "eog", "stim"]).load_data()
raw.info["line_freq"] = 60.0

# %%
# The sample recording is a single run. To show how runs are combined, we
# treat its two halves as two runs, each preprocessed on its own (here only
# a high-pass filter and an average reference, to keep the tutorial fast; in
# practice the ``eeg-erp`` preset).
pre = meu.Pipeline([("highpass", S.Filter(0.1, None)), ("reference", S.Reference("average"))])
runs = [raw.copy().crop(0, 138), raw.copy().crop(138, None)]
clean_runs = [pre.fit_transform(run) for run in runs]

# %%
# The epochs stage
# ----------------
# The ``eeg-erp`` epochs stage low-passes the data at 40 Hz, cuts epochs from
# -0.2 to 0.5 s with a pre-stimulus baseline, and runs local autoreject (all
# MNE-BIDS-Pipeline defaults). Only the events need to be given:
event_id = {"auditory/left": 1, "auditory/right": 2, "visual/left": 3, "visual/right": 4}
ep = meu.Pipeline.preset("eeg-erp", stage="epochs", event_id=event_id)
ep

# %%
epochs_per_run = [ep.fit_transform(run) for run in clean_runs]
epochs_per_run

# %%
# The last fitted run: why epochs were dropped, and the average per
# condition:
print(ep.qc_["epoch"])
figs = ep["epoch"].plot("evoked")

# %%
# autoreject
# ----------
# Local autoreject (Jas et al., 2017) learns a peak-to-peak threshold per
# channel. Within an epoch it interpolates the few channels above their
# threshold; epochs with too many bad channels are dropped. Each row is a
# channel and each column an epoch:
print({k: ep.qc_["autoreject"][k] for k in ("n_dropped", "fraction_dropped")})
figs = ep["autoreject"].plot("reject_log")

# %%
# and the learned thresholds:
figs = ep["autoreject"].plot("thresholds")

# %%
# Combine the runs
# ----------------
# :func:`meeg_utils.epochs.combine` concatenates the runs after checking
# that they agree: the same event codes, the same channels (a channel bad in
# one run is marked bad in all, ``bads="union"``), and for MEG a head
# position within 2 mm (otherwise map the runs to a common head position
# first with :class:`~meeg_utils.steps.HeadAlign`).
epochs = meu.epochs.combine(epochs_per_run)
epochs

# %%
evokeds = {cond: epochs[cond].average() for cond in ("auditory", "visual")}
mne.viz.plot_compare_evokeds(evokeds, picks="EEG 021", show=False)

# %%
evokeds["auditory"].plot_joint(times=[0.09, 0.2], title="Auditory stimuli", show=False)

# %%
# Resting state
# -------------
# Without events, the ``*-rest`` presets cut consecutive windows of a given
# length (the frequency resolution of later spectra is 1 / length):
rest = meu.Pipeline.preset("eeg-rest", stage="epochs", epoch_duration=2.0)
windows = rest.fit_transform(clean_runs[0])
windows.compute_psd(fmax=40).plot(average=True, amplitude=False, show=False)
