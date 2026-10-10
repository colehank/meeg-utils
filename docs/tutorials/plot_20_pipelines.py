"""
.. _tut-pipelines:

Build and adjust a pipeline
===========================

A pipeline is a list of named steps. Each step is a small scikit-learn
style estimator: its parameters are set when it is created, ``fit`` learns
what it needs from a recording, ``transform`` applies it, and afterwards it
reports what it did (``qc_``) and draws it (``plot``).

This tutorial builds the recommended MEG pipeline for a Neuromag system
step by step on MNE's ``sample`` recording, looks at every step, and then
changes it.
"""

# sphinx_gallery_thumbnail_number = 5
# sphinx_gallery_multi_image = "single"

# %%
import tempfile
from pathlib import Path

import mne

import meeg_utils as meu
from meeg_utils import steps as S

mne.set_log_level("ERROR")

data_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(data_path / "MEG" / "sample" / "sample_audvis_raw.fif")
raw.pick(["meg", "eog", "stim"]).load_data()
raw.info["line_freq"] = 60.0
meu.io.detect_system(raw)

# %%
# The steps
# ---------
# Neuromag systems come with a cross-talk and a fine-calibration file, which
# SSS needs. In a BIDS dataset they are found automatically; here we pass
# them.
cross_talk = data_path / "SSS" / "ct_sparse_mgh.fif"
calibration = data_path / "SSS" / "sss_cal_mgh.dat"

pipe = meu.Pipeline(
    [
        ("bads", S.BadChannels("maxwell", cross_talk=cross_talk, calibration=calibration)),
        ("sss", S.Maxwell(cross_talk=cross_talk, calibration=calibration)),
        ("highpass", S.Filter(0.1, None)),
        ("resample", S.Resample(250.0)),
        ("line_noise", S.LineNoise("zapline-plus")),
        ("ica", S.ICA(20, picks="meg", method="infomax", labeler="megnet")),
    ]
)
pipe

# %%
# This is the ``meg-erp`` preset; ``meu.Pipeline.preset("meg-erp",
# system="neuromag")`` builds the same pipeline and finds the two files in
# the BIDS dataset of the recording.
clean = pipe.fit_transform(raw)

# %%
# Bad channels
# ------------
# :func:`mne.preprocessing.find_bad_channels_maxwell` scores every channel
# against its SSS reconstruction. ``MEG 2443`` was already marked bad in the
# file:
print(pipe.qc_["bads"]["meg"])
figs = pipe["bads"].plot("scores")

# %%
# SSS
# ---
# Signal-space separation keeps the fields from inside the head and rebuilds
# the bad channels. The spectra before and after:
print({k: pipe.qc_["sss"][k] for k in ("reconstructed", "n_basis", "power_change_db")})
figs = pipe["sss"].plot("psd")

# %%
# Line noise
# ----------
# ZapLine-plus removes the 60 Hz components and their harmonics, chunk by
# chunk, without notching the spectrum:
figs = pipe["line_noise"].plot("psd")

# %%
# ICA with MEGnet
# ---------------
# MEGnet labels the 20 components (it was trained on 20 Infomax components
# of 250 Hz data, which is why the pipeline resamples first). Components
# whose artifact probability is at least 0.8 are removed:
print(pipe.qc_["ica"]["labels"])
figs = pipe["ica"].plot("labels")

# %%
figs = pipe["ica"].plot("components")

# %%
# The excluded components, with their spectrum and time course (these
# figures need the data):
figs = pipe["ica"].plot("properties", inst=raw)

# %%
# If you disagree with a label, correct it and transform again; the
# decomposition is kept:
#
# .. code-block:: python
#
#    pipe["ica"].relabel({4: "brain"})
#    clean = pipe.transform(raw)
#
# Change the pipeline
# -------------------
# Parameters are addressed as ``<step>__<parameter>``, as in scikit-learn:
pipe.set_params(highpass__l_freq=1.0, ica__threshold=0.9)
pipe["highpass"]

# %%
# Steps can be added, replaced and removed by name, and a pipeline can be
# sliced:
pipe.insert_after("line_noise", "lowpass", S.Filter(None, 40.0))
pipe.remove("resample")
list(pipe.named_steps), pipe[:2]

# %%
# A pipeline is saved as YAML, to rerun it or to publish it with a paper:
fname = Path(tempfile.mkdtemp()) / "pipeline.yaml"
pipe.to_yaml(fname)
print(fname.read_text())

# %%
# ``meu.Pipeline.from_yaml(fname)`` reads it back, and ``meu run
# pipeline.yaml --sources bids/ --out derivatives/`` runs it from the command
# line.
#
# MEG and EEG together
# --------------------
# :class:`~meeg_utils.steps.ByChannelType` runs separate steps on each
# channel type and writes the results back into one recording, e.g. SSS for
# the MEG and PREP plus an average reference for the EEG:
combined = meu.Pipeline(
    [
        (
            "by_type",
            S.ByChannelType(
                {
                    "meg": [("sss", S.Maxwell(cross_talk=cross_talk, calibration=calibration))],
                    "eeg": [("bads", S.BadChannels("prep")), ("reference", S.Reference())],
                }
            ),
        ),
        ("highpass", S.Filter(0.1, None)),
    ]
)
combined
