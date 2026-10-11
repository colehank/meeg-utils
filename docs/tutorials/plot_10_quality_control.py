"""
.. _tut-qc:

Acquisition quality control
===========================

Before cleaning a recording, check how well it was made: a flat channel, a
loose EOG electrode, a head that moved, or a sidecar that disagrees with the
data are better found the day the data were recorded than months later.

:mod:`meeg_utils.qc` answers these questions without changing the data. This
tutorial runs it on MNE's ``sample`` recording (Neuromag MEG and EEG).
"""

# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_multi_image = "single"

# %%
import mne

import meeg_utils as meu
from meeg_utils import qc

mne.set_log_level("ERROR")

data_path = mne.datasets.sample.data_path()
meg_dir = data_path / "MEG" / "sample"
raw = mne.io.read_raw_fif(meg_dir / "sample_audvis_raw.fif").load_data()
raw.info["line_freq"] = 60.0

# %%
# Run every check
# ---------------
# :func:`~meeg_utils.qc.inspect` picks the checks that apply to the data
# (MEG, EEG, the acquisition system, BIDS) and lists the others with the
# reason they were skipped:
report = qc.inspect(raw)
print(report)

# %%
# Each finding has a level:
#
# ``fail`` / ``warn``
#     A problem by definition (a flat channel, a SQUID jump, a bridged
#     electrode, a sidecar that disagrees with the data) or a channel that
#     stands out from the others in this recording.
# ``ok``
#     The criterion was checked and passed.
# ``info``
#     A measurement without a verdict: head movement, blink rate, heart
#     rate, muscle activity, ... Their normal range depends on the
#     participants and the setup (children move more than adults, passive
#     electrodes have higher impedances than active ones), so meeg-utils
#     does not pretend to know it.
#
# ``report.flags`` lists the problems, worst first:
for finding in report.flags:
    print(f"{finding.level:4}  {finding.check}: {finding.message}")

# %%
# Look at what was measured
# -------------------------
# Every check draws what it measured. Channels that stand out from the
# others (Local Outlier Factor, per channel type):
figs = report.checks["outlier_channels"].plot()

# %%
# Blinks show whether the EOG electrodes work; the average blink should look
# like one:
figs = report.checks["blinks"].plot()

# %%
# Without an ECG channel, MNE builds a synthetic ECG from the magnetometers.
# meeg-utils judges the heartbeat detection by how regular the intervals
# between beats are, and does not report a rate it cannot trust:
print(report.checks["heart_rate"].metrics_)
figs = report.checks["heart_rate"].plot()

# %%
# The digitized head shape, the sphere fitted to it, and the MEG sensors:
figs = report.checks["digitization"].plot()

# %%
# The events found in the recording:
figs = report.checks["events"].plot()

# %%
# The empty room
# --------------
# The empty-room recording of the session shows the noise of the sensors and
# the room on the day. In a BIDS dataset it is found automatically; here we
# pass the file:
er = qc.EmptyRoom(empty_room=meg_dir / "ernoise_raw.fif").compute(raw)
print(er.metrics_["mag"], er.metrics_["grad"], sep="\n")
figs = er.plot()

# %%
# Your own thresholds
# -------------------
# The measurements reported as ``info`` become verdicts when you give them a
# threshold that fits your participants and equipment. Every threshold is a
# parameter of its check:
report = qc.inspect(
    raw,
    checks=[
        qc.Blinks(min_rate=5),  # blinks per minute
        qc.Muscle(warn_fraction=0.05),  # share of the recording
        qc.Events(expected={"1": 72, "2": 73, "3": 73, "4": 71}),
    ],
)
print(report)

# %%
# A whole dataset
# ---------------
# Across many recordings the ``info`` measurements get a reference: the rest
# of the dataset. :func:`~meeg_utils.qc.inspect_dataset` runs the checks on
# every recording of a BIDS dataset and flags the recordings that stand out
# on any metric (modified z-score above 3.5):
#
# .. code-block:: python
#
#    result = meu.qc.inspect_dataset("bids/", n_jobs=8)
#    result.outliers          # recording, metric, value, z
#    result.to_csv("qc/summary.csv")
#    result.plot()["levels"]  # recordings x checks
#
# or from the command line: ``meu qc bids/ --out qc/``, which also writes an
# HTML report per recording with :func:`meeg_utils.report.build`:
html = meu.report.build(qc=qc.inspect(raw), title="QC of sample_audvis_raw")
html
