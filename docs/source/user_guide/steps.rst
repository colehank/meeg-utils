Steps and pipelines
===================

Conventions
-----------

Steps and pipelines follow scikit-learn:

* all parameters are keyword arguments of the constructor, and nothing else
  happens there;
* ``fit``, ``transform`` and ``fit_transform`` are the only ways to process
  data; what ``fit`` learns is stored in attributes ending with ``_``
  (``bads_``, ``ica_``, ``mapping_``, ...) together with ``qc_``;
* ``get_params`` / ``set_params`` (nested as ``step__param`` in a pipeline)
  and :func:`sklearn.base.clone` work as usual.

Pipelines accept a Raw object or a path (read with
:func:`meeg_utils.io.read`), check before running that the output of each
step is a valid input for the next, and record provenance
(``provenance_``: software versions, system, parameters, duration and
warnings of each step). They can be edited and sliced:

.. code-block:: python

   pipe.replace("line_noise", S.LineNoise(method="notch"))
   pipe.insert_after("interpolate", "align", S.HeadAlign(dest))
   pipe.remove("resample")
   first_part = pipe[:3]          # a pipeline of the first three steps

Data integrity
--------------

A step never drops channels, reorders them or shifts the time axis unless
its class declares it (``changes_channels`` / ``changes_times``, e.g.
``Resample``). ``first_samp``, ``meas_date`` and annotations are preserved.
Each step only processes the systems it has been validated for and raises
otherwise. :func:`meeg_utils.testing.check_step` checks all of this, plus the
figures; every built-in step passes it, and custom steps can use it in their
tests.

Available steps
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Step
     - What it does
     - Figures (``plot_kinds``)
   * - :class:`~meeg_utils.steps.Filter`
     - FIR/IIR band-, high- or low-pass (MNE defaults).
     - ``response``, ``psd``
   * - :class:`~meeg_utils.steps.Resample`
     - Change the sampling rate.
     -
   * - :class:`~meeg_utils.steps.LineNoise`
     - ZapLine-plus (default), ZapLine or notch, per channel type; QC
       measures suppression, distortion and over-/under-cleaning.
     - ``psd``
   * - :class:`~meeg_utils.steps.BadChannels`
     - PREP (EEG) or Maxwell (Neuromag, CTF) detection; marks
       ``info["bads"]``.
     - ``sensors``, ``scores``
   * - :class:`~meeg_utils.steps.Interpolate`
     - Repair bad channels (spline for EEG, field mapping for MEG).
     - ``sensors``
   * - :class:`~meeg_utils.steps.Reference`
     - EEG reference; CTF gradient compensation grade.
     -
   * - :class:`~meeg_utils.steps.ICA`
     - ICA with ICLabel / MEGnet labels; removes components whose artifact
       probability reaches ``threshold``; ``relabel`` for manual review.
     - ``components``, ``labels``, ``properties``\*, ``overlay``\*
   * - :class:`~meeg_utils.steps.HeadAlign`
     - Map MEG data to another head position (e.g. the average over runs).
     - ``positions``
   * - :class:`~meeg_utils.steps.BridgedElectrodes`
     - Detect and interpolate bridged EEG electrodes.
     - ``topomap``, ``distances``
   * - :class:`~meeg_utils.steps.DropChannels`
     - Remove named channels (e.g. mastoids before re-referencing).
     -
   * - :class:`~meeg_utils.steps.Epoch`
     - Raw → Epochs around events (annotations or stimulus channel); event
       codes fixed at fit; optional metadata from BIDS ``events.tsv``.
     - ``drop_log``, ``evoked``
   * - :class:`~meeg_utils.steps.Baseline`
     - Baseline correction (mean, ratio, percent, z-score); refuses
       per-epoch z-scores on baselines too short to estimate a standard
       deviation, and offers a pooled scale.
     -
   * - :class:`~meeg_utils.steps.AutoReject`
     - Learned rejection thresholds (autoreject): repair channels and drop
       epochs (local), or one threshold per channel type (global).
     - ``reject_log``, ``thresholds``

\* needs data: ``step.plot(inst=...)``.

Figures
-------

``step.plot()`` draws every figure that needs no data, from summaries kept
on the fitted step; ``step.plot(inst=raw)`` adds the others;
``step.plot("psd", **kwargs)`` draws one kind, passing ``kwargs`` to the
underlying MNE function. Figures are returned, never shown. For a pipeline,
``pipe.plot(inst=raw)`` replays the fitted steps so that every step plots
the data it actually received.

Several runs
------------

Each run is processed on its own: bad channels, line noise and ICA differ
between runs. For MEG, map the runs to a common head position before
combining them:

.. code-block:: python

   dest = meu.io.average_dev_head_t(run_paths)   # weighted by good duration
   pipe.insert_after("ica", "align", S.HeadAlign(dest))

``HeadAlign`` uses minimum-norm field mapping; in simulations it lowers the
error caused by a 10 mm / 5° head movement from 32-49 % to 1-3 %.

After epoching each run, combine them:

.. code-block:: python

   epoch = meu.Pipeline([("epoch", S.Epoch("stimulus", -0.2, 0.8)), ("ar", S.AutoReject())])
   runs = [epoch.fit_transform(run) for run in preprocessed_runs]
   epochs = meu.epochs.combine(runs)              # bad channels: union across runs

:func:`meeg_utils.epochs.combine` checks that the runs share channels,
sampling rate and epoch times, matches events by name (unifying their
codes), and, for MEG, that their head positions differ by less than 2 mm
(otherwise align them first).
