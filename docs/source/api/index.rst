API Reference
=============

.. currentmodule:: meeg_utils

Pipelines and steps
-------------------

.. autosummary::
   :toctree: _autosummary

   Pipeline
   Step
   process
   Dataset
   process_dataset
   BatchResult

Processing steps (``meeg_utils.steps``)
---------------------------------------

.. autosummary::
   :toctree: _autosummary

   steps.Filter
   steps.Resample
   steps.LineNoise
   steps.BadChannels
   steps.Interpolate
   steps.Reference
   steps.ICA
   steps.HeadAlign
   steps.Maxwell
   steps.ByChannelType
   steps.HFC
   steps.Regression
   steps.BadSegments
   steps.ASR
   steps.SNS
   steps.BridgedElectrodes
   steps.DropChannels
   steps.Epoch
   steps.FixedLengthEpochs
   steps.Baseline
   steps.AutoReject
   epochs.combine

Reading and writing (``meeg_utils.io``)
---------------------------------------

.. autosummary::
   :toctree: _autosummary

   io.read
   io.save_derivative
   io.existing_derivatives
   io.detect_system
   io.get_datatypes
   io.average_dev_head_t

Quality control (``meeg_utils.qc``)
-----------------------------------

.. autosummary::
   :toctree: _autosummary

   qc.inspect
   qc.inspect_dataset
   qc.find_recordings
   qc.QCReport
   qc.DatasetQC
   qc.Finding
   qc.Check
   qc.Amplitude
   qc.Bridging
   qc.Impedance
   qc.OutlierChannels
   qc.NarrowbandNoise
   qc.Muscle
   qc.Blinks
   qc.HeartRate
   qc.HeadMovement
   qc.Digitization
   qc.Events
   qc.SquidJumps
   qc.ChpiSNR
   qc.EmptyRoom
   qc.BidsMetadata
   qc.Photodiode
   report.build
   report.from_derivative
   report.summarize
   report.DerivativesSummary

Presets (``meeg_utils.presets``)
--------------------------------

.. autosummary::
   :toctree: _autosummary

   presets.available
   presets.get
   presets.options

Logging and testing
-------------------

.. autosummary::
   :toctree: _autosummary

   setup_logging
   teardown_logging
   log_to_file
   testing.check_step
   testing.check_check
