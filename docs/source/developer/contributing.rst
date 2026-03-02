Contributing Guide
==================

Thank you for considering contributing to meeg-utils! This document provides
guidelines and instructions for contributing.

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.11 or higher
* `uv <https://github.com/astral-sh/uv>`_ package manager

Installation
~~~~~~~~~~~~

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/colehank/meeg-utils.git
   cd meeg-utils

2. Install dependencies:

.. code-block:: bash

   uv sync --all-extras --dev

3. Install pre-commit hooks:

.. code-block:: bash

   uv run pre-commit install

Development Workflow
--------------------

Test-Driven Development (TDD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project follows strict TDD principles:

1. **Write tests first** - Always write the test before implementing functionality
2. **Watch it fail (RED)** - Run tests to ensure they fail for the right reason
3. **Write minimal code (GREEN)** - Implement just enough code to pass the test
4. **Refactor** - Clean up code while keeping tests green

See :doc:`testing` for more details.

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   uv run pytest

   # Run specific test file
   uv run pytest tests/test_preprocessing/test_pipeline.py

   # Run with coverage
   uv run pytest --cov=src/meeg_utils --cov-report=html

Code Quality
~~~~~~~~~~~~

Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run ruff linter
   uv run ruff check src/ tests/

   # Auto-fix issues
   uv run ruff check --fix src/ tests/

   # Format code
   uv run ruff format src/ tests/

Type Checking
^^^^^^^^^^^^^

.. code-block:: bash

   # Run mypy
   uv run mypy src/

Security Check
^^^^^^^^^^^^^^

.. code-block:: bash

   # Run bandit
   uv run bandit -r src/ -c pyproject.toml

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Pre-commit hooks will automatically run on ``git commit``:

* Trailing whitespace removal
* End-of-file fixer
* YAML/TOML/JSON validation
* Ruff linting and formatting

To run manually:

.. code-block:: bash

   uv run pre-commit run --all-files

Pull Request Process
--------------------

1. Create a feature branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes
~~~~~~~~~~~~~~~~~~~~

* Follow TDD: write tests first
* Ensure all tests pass
* Add docstrings to all public functions/classes
* Update documentation if needed

3. Commit your changes
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add .
   git commit -m "feat: add new feature"

Follow `Conventional Commits <https://www.conventionalcommits.org/>`_:

* ``feat:`` new feature
* ``fix:`` bug fix
* ``docs:`` documentation changes
* ``style:`` formatting, missing semicolons, etc.
* ``refactor:`` code refactoring
* ``test:`` adding tests
* ``chore:`` maintenance tasks

4. Push to GitHub
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/your-feature-name

5. Create Pull Request
~~~~~~~~~~~~~~~~~~~~~~

* Go to GitHub and create a pull request
* Fill in the PR template
* Link related issues
* Wait for CI checks to pass
* Request review

Code Style Guidelines
---------------------

Python Style
~~~~~~~~~~~~

* Follow PEP 8
* Use type hints for all function signatures
* Maximum line length: 100 characters
* Use double quotes for strings
* Use f-strings for string formatting

Docstrings
~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def process_data(
       raw: BaseRaw,
       highpass: float = 0.1,
       lowpass: float = 100.0,
   ) -> BaseRaw:
       \"\"\"Process MEG/EEG data with filtering.

       Parameters
       ----------
       raw : BaseRaw
           Input raw data.
       highpass : float, optional
           High-pass filter cutoff in Hz. Default is 0.1.
       lowpass : float, optional
           Low-pass filter cutoff in Hz. Default is 100.0.

       Returns
       -------
       BaseRaw
           Processed raw data.

       Examples
       --------
       >>> raw_filtered = process_data(raw, highpass=1.0, lowpass=50.0)
       \"\"\"

Testing Guidelines
~~~~~~~~~~~~~~~~~~

* One test file per source file
* Use descriptive test names: ``test_<what>_<condition>``
* Use fixtures for common test data
* Mock expensive operations (trust library implementations)
* Aim for >80% code coverage

Reporting Issues
----------------

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:

   * OS and version
   * Python version
   * meeg-utils version
   * Relevant package versions

Feature Requests
----------------

Feature requests are welcome! Please:

1. Check existing issues first
2. Clearly describe the use case
3. Explain why this feature would be useful
4. Consider if it fits the project scope

Questions?
----------

* Open a `GitHub Discussion <https://github.com/colehank/meeg-utils/discussions>`_
* Check the documentation

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.
