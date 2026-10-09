Quality control
===============

:mod:`meeg_utils.qc` answers *how well was this recording made?*, so that
problems can be fixed during the session or reported to whoever acquired
the data. Checks never modify the data.

.. code-block:: python

   import meeg_utils as meu

   report = meu.qc.inspect(raw)      # every check that applies to this recording
   print(report)                     # verdicts and the findings that need attention
   report.flags                      # findings that are "warn" or "fail", worst first
   report.skipped                    # checks that did not apply, with the reason
   figs = report.plot()              # {check: {kind: Figure}}

Each check is a small class: parameters (thresholds included) are set in
the constructor, ``compute(raw)`` measures, and the results are in
``metrics_`` and ``findings_`` (one :class:`~meeg_utils.qc.Finding` per
criterion, with a verdict ``"ok"``, ``"warn"`` or ``"fail"`` and a message
that states the threshold). Use your lab's thresholds by passing them:

.. code-block:: python

   meu.qc.inspect(raw, checks=[
       meu.qc.Impedance(warn_kohm=10),            # passive electrodes
       meu.qc.HeadMovement(warn_mm=3),
       meu.qc.Events(expected={"stimulus": 240}),
   ])

Checks
------

Every default threshold comes from MNE's defaults or the literature, and is
documented with its source in the class docstring; where no standard
exists, the docstring says so.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Check
     - What it finds
     - Figures
   * - :class:`~meeg_utils.qc.Amplitude`
     - flat channels, dropouts, clipping (amplifier saturation), NaN samples
     - ``channels``
   * - :class:`~meeg_utils.qc.Bridging`
     - EEG electrodes shorted by gel (MNE's electrical-distance method)
     - ``topomap``, ``distances``
   * - :class:`~meeg_utils.qc.Impedance`
     - electrodes above an impedance limit (BrainVision header or given values)
     - ``channels``
   * - :class:`~meeg_utils.qc.OutlierChannels`
     - channels unlike the others (Local Outlier Factor), any system
     - ``scores``
   * - :class:`~meeg_utils.qc.NarrowbandNoise`
     - line noise, other narrowband noise sources, channels with poor contact
     - ``psd``, ``line_channels``
   * - :class:`~meeg_utils.qc.Muscle`
     - share of time with muscle activity
     - ``scores``
   * - :class:`~meeg_utils.qc.Blinks`
     - blink rate, i.e. whether the EOG works
     - ``blinks``
   * - :class:`~meeg_utils.qc.HeartRate`
     - heart rate, i.e. whether the ECG works
     - ``heartbeats``
   * - :class:`~meeg_utils.qc.HeadMovement`
     - MEG head movement (cHPI or a ``.pos`` file) and HPI coil fit quality
     - ``positions``, ``displacement``
   * - :class:`~meeg_utils.qc.SquidJumps`
     - flux jumps of SQUID sensors (FieldTrip's jump detection)
     - ``jumps``
   * - :class:`~meeg_utils.qc.ChpiSNR`
     - HPI coils much weaker than the others (Neuromag)
     - ``snr``
   * - :class:`~meeg_utils.qc.EmptyRoom`
     - sensor noise floor and noisy sensors in the session's empty-room
       recording (found in BIDS), empty room from another day
     - ``psd``
   * - :class:`~meeg_utils.qc.Digitization`
     - missing electrode positions, implausible head shape, unmeasured head
       position, head-to-sensor distance
     - ``headshape``
   * - :class:`~meeg_utils.qc.Events`
     - event counts against what was expected, duplicated triggers
     - ``timeline``
   * - :class:`~meeg_utils.qc.BidsMetadata`
     - BIDS sidecar disagreeing with the data or ``channels.tsv``
       (sampling rate, duration, channel counts), missing line frequency
     -
   * - :class:`~meeg_utils.qc.Photodiode` (not run by default)
     - delay and jitter between triggers and the photodiode, stimuli not
       shown: ``meu.qc.Photodiode("PD").compute(raw)``
     - ``delays``

Bridged electrodes can also be repaired during preprocessing with
:class:`meeg_utils.steps.BridgedElectrodes`.

A whole dataset
---------------

.. code-block:: python

   result = meu.qc.inspect_dataset("bids/", n_jobs=8)   # or a list of recordings
   result.outliers        # recordings that stand out on a metric (modified z > 3.5)
   result.to_csv("qc/summary.csv")
   result.plot()["levels"]                               # recordings x checks overview

Reports
-------

:func:`meeg_utils.report.build` collects QC findings and figures, and the
metrics and figures of every step of a fitted pipeline, in an
:class:`mne.Report`:

.. code-block:: python

   rep = meu.report.build(pipe, inst=raw, qc=meu.qc.inspect(raw))
   rep.save("sub-01_report.html")
