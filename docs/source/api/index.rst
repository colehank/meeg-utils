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

Presets (``meeg_utils.presets``)
--------------------------------

.. autosummary::
   :toctree: _autosummary

   presets.available
   presets.get

Logging and testing
-------------------

.. autosummary::
   :toctree: _autosummary

   setup_logging
   teardown_logging
   log_to_file
   testing.check_step
