Command line
============

Installing meeg-utils provides the ``meu`` command.

Quality control
---------------

.. code-block:: bash

   meu qc /data/bids --out qc/ -j 8            # every MEG/EEG recording of a BIDS dataset
   meu qc sub-01_task-rest_eeg.vhdr --out qc/  # selected recordings
   meu qc /data/bids --out qc/ --checks amplitude,bridging,impedance
   meu qc /data/bids --out qc/ --fail-on fail  # exit status 2 if any check fails

For each recording ``meu qc`` writes ``<name>_qc.json`` (metrics and
findings) and ``<name>_qc.html`` (report with figures; ``--no-report``
skips it); for the dataset, ``qc_summary.csv`` and ``qc_overview.png``.
Outlying recordings and errors are printed.

Preprocessing
-------------

.. code-block:: bash

   meu run --preset had-meeg --datatype eeg --sources /data/bids \
       --out /data/bids/derivatives/meu -j 8
   meu run pipeline.yaml --sources rec1_raw.fif rec2_raw.fif --out derivatives/
   meu run pipeline.yaml --sources /data/bids --datatype meg --out derivatives/ --skip-existing

Preset options are passed with ``--option KEY=VALUE`` (values are read as
YAML). For the MEG presets the system is detected from the first recording
unless ``--option system=...`` is given:

.. code-block:: bash

   meu run --preset meg-erp --sources /data/bids --datatype meg --out derivatives/
   meu run --preset meg-erp --option stage=epochs --option "event_id=[face, house]" \
       --sources derivatives/sub-*/meg/*_desc-preproc_meg.fif --out derivatives/ --desc epochs
   meu run --preset eeg-rest --option stage=epochs --option epoch_duration=2 ...

With ``--epochs`` the sources must be one BIDS root, and every session is
taken through preprocessing, epoching and combination of its runs
(:func:`meeg_utils.process_dataset`; existing outputs are skipped, so the
command resumes). ``--epochs`` alone uses the epochs stage of ``--preset``;
``--epochs epochs.yaml`` a configuration file. The log of every run and
session is written to ``<out>/meu_batch.csv``.

``pipeline.yaml`` is written by :meth:`meeg_utils.Pipeline.to_yaml`. Each
recording is processed separately (:func:`meeg_utils.process`); the exit
status is 1 if any recording failed.

``-v`` shows debug messages, ``-q`` only warnings and errors.

Presets
-------

.. code-block:: bash

   meu preset                                         # names, summaries, options
   meu preset meg-erp --option system=ctf -o pipeline.yaml
   meu run pipeline.yaml --sources /data/bids --datatype meg --out derivatives/

``meu preset NAME`` writes the preset as a pipeline configuration (to the
screen, or to ``-o FILE``) so it can be edited before ``meu run``.

Reports of saved results
------------------------

.. code-block:: bash

   meu report derivatives/sub-01/meg/sub-01_task-faces_run-01_desc-preproc_meg.fif
   meu report derivatives/ --desc preproc

For one derivative, ``meu report`` writes an HTML report next to it: the
pipeline and its parameters, every step's QC metrics and warnings, the
provenance and views of the data (the fitted steps are not saved, so their
own figures are only in reports built right after processing, see
:func:`meeg_utils.report.build`). For a folder, it collects the QC metrics
of every derivative into ``derivatives_summary.csv`` and
``derivatives_summary.html``, flagging runs whose value of a metric is an
outlier (modified z > 3.5), e.g. a run with many more bad channels or
rejected epochs than the others.
