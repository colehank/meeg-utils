"""
.. _tut-datasets:

Whole BIDS datasets
===================

For a study, the same steps run on every recording of a BIDS dataset:
check them all, preprocess every run, epoch it, and combine the runs of each
session. :class:`meeg_utils.Dataset` selects the recordings and
:func:`meeg_utils.process_dataset` takes them through every stage, resuming
an interrupted batch and collecting failures. The ``meu`` command does the
same from a terminal.

We first write MNE's ``sample`` recording as a small BIDS dataset, with two
runs (its two halves), the empty-room recording and the Neuromag
calibration files.
"""

# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_multi_image = "single"

# %%
import tempfile
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_meg_calibration, write_meg_crosstalk, write_raw_bids

import meeg_utils as meu
from meeg_utils import steps as S

mne.set_log_level("ERROR")

data_path = mne.datasets.sample.data_path()
meg_dir = data_path / "MEG" / "sample"
raw = mne.io.read_raw_fif(meg_dir / "sample_audvis_raw.fif").load_data()
raw.info["line_freq"] = 60.0
event_id = {"auditory/left": 1, "auditory/right": 2, "visual/left": 3, "visual/right": 4}
all_events = {**event_id, "smiley": 5, "button": 32}

bids_root = Path(tempfile.mkdtemp()) / "bids"
for run, (tmin, tmax) in enumerate([(0, 138), (138, None)], start=1):
    part = raw.copy().crop(tmin, tmax)
    part_events = mne.find_events(part)
    path = BIDSPath(subject="01", session="01", task="audvis", run=run, root=bids_root)
    write_raw_bids(
        part, path, events=part_events, event_id=all_events, format="FIF",
        allow_preload=True, overwrite=True,
    )  # fmt: skip

er = mne.io.read_raw_fif(meg_dir / "ernoise_raw.fif").load_data()
er.info["line_freq"] = 60.0
er_path = BIDSPath(
    subject="emptyroom", session=er.info["meas_date"].strftime("%Y%m%d"), task="noise",
    root=bids_root,
)  # fmt: skip
write_raw_bids(er, er_path, format="FIF", allow_preload=True, overwrite=True)

meg_path = BIDSPath(subject="01", session="01", datatype="meg", root=bids_root)
write_meg_crosstalk(data_path / "SSS" / "ct_sparse_mgh.fif", meg_path)
write_meg_calibration(data_path / "SSS" / "sss_cal_mgh.dat", meg_path)
sorted(str(p.relative_to(bids_root)) for p in bids_root.rglob("*") if p.is_file())[:14]

# %%
# Select recordings
# -----------------
# :class:`~meeg_utils.Dataset` selects recordings by BIDS entities and
# groups the runs of each session and task. Empty-room recordings are left
# out unless asked for:
ds = meu.Dataset(bids_root, datatype="meg")
ds.table()

# %%
# Check them all
# --------------
# :func:`~meeg_utils.qc.inspect_dataset` runs the acquisition checks on every
# recording. In a BIDS dataset the checks also use the sidecars, the
# calibration files and the session's empty room:
result = meu.qc.inspect_dataset(bids_root)
result

# %%
# One row per recording and check: grey is measured only, green passed,
# amber and red need attention. With five or more recordings, a dot marks
# a recording that stands out from the rest of the dataset on a metric:
figs = result.plot()

# %%
first = next(iter(result.reports.values()))
print(first.checks["empty_room"].metrics_["mag"])

# %%
# Process every run
# -----------------
# Preprocessing runs on every run with its own copy of the pipeline; the
# epochs stage runs on each preprocessed run, after mapping the MEG runs of a
# session to their average head position; then the runs are combined.
# Here the preprocessing is the front of the ``meg-erp`` preset (SSS, with
# the calibration files found in the dataset), to keep the tutorial short:
preprocessing = meu.Pipeline(
    [
        ("bads", S.BadChannels("maxwell")),
        ("sss", S.Maxwell()),
        ("highpass", S.Filter(0.1, None)),
        ("resample", S.Resample(250.0)),
    ]
)
epochs = meu.Pipeline(
    [
        ("lowpass", S.Filter(None, 40.0)),
        ("epoch", S.Epoch(list(event_id), tmin=-0.2, tmax=0.5)),
    ]
)
deriv_root = bids_root / "derivatives" / "meu"
batch = meu.process_dataset(ds, deriv_root, preprocessing=preprocessing, epochs=epochs)
batch

# %%
# Every output is a BIDS derivative with a JSON sidecar holding the
# configuration, software versions, warnings and QC metrics of every step:
sorted(str(p.relative_to(deriv_root)) for p in deriv_root.rglob("*.fif"))

# %%
# Calling :func:`~meeg_utils.process_dataset` again skips what exists, so an
# interrupted batch resumes and failed runs are retried. ``batch.failed``
# lists the failures with their errors.
#
# Review the results
# ------------------
# :func:`meeg_utils.report.summarize` collects the QC metrics of every
# derivative into one table, with the warnings of every run:
summary = meu.report.summarize(deriv_root, desc="preproc")
summary.to_records()

# %%
# and :func:`meeg_utils.report.from_derivative` rebuilds the report of a
# single output (pipeline, parameters, QC metrics, provenance and the data):
fname = next(deriv_root.rglob("*_desc-preproc_meg.fif"))
meu.report.from_derivative(fname)

# %%
# The same from the command line
# ------------------------------
# .. code-block:: bash
#
#    meu qc bids/ --out qc/
#    meu run --preset meg-erp --epochs \
#        --option "event_id=[auditory/left, auditory/right, visual/left, visual/right]" \
#        --sources bids/ --out bids/derivatives/meu
#    meu report bids/derivatives/meu --desc preproc
