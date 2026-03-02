CI/CD
=====

Continuous Integration and Deployment setup for meeg-utils.

GitHub Actions
--------------

CI Workflow
~~~~~~~~~~~

Automatically runs on every push and pull request to ``main``, ``master``, or ``develop`` branches.

**Jobs:**

1. **lint** - Ruff linter and formatter checks
2. **type-check** - MyPy static type checking
3. **security** - Bandit security analysis
4. **test** - Pytest with coverage (matrix: Python 3.11/3.12 × Ubuntu/macOS/Windows)
5. **docs** - Docstring coverage with interrogate

Configuration: ``.github/workflows/ci.yml``

Release Workflow
~~~~~~~~~~~~~~~~

Automatically triggered on git tags matching ``v*.*.*``.

**Jobs:**

1. **build** - Build distribution packages (wheel + sdist)
2. **publish-pypi** - Publish to PyPI
3. **github-release** - Create GitHub release

Configuration: ``.github/workflows/release.yml``

Pre-commit Hooks
----------------

Local code quality checks that run before each commit.

**Hooks:**

* Trailing whitespace removal
* End-of-file fixer
* YAML/TOML/JSON validation
* Ruff linting (with auto-fix)
* Ruff formatting

Install:

.. code-block:: bash

   uv run pre-commit install

Run manually:

.. code-block:: bash

   uv run pre-commit run --all-files

Configuration: ``.pre-commit-config.yaml``

Code Quality Checks
-------------------

Ruff
~~~~

Fast Python linter and formatter.

.. code-block:: bash

   # Lint
   uv run ruff check src/ tests/

   # Auto-fix
   uv run ruff check --fix src/ tests/

   # Format
   uv run ruff format src/ tests/

Configuration: ``pyproject.toml`` → ``[tool.ruff]``

MyPy
~~~~

Static type checker.

.. code-block:: bash

   uv run mypy src/

Configuration: ``pyproject.toml`` → ``[tool.mypy]``

Bandit
~~~~~~

Security vulnerability scanner.

.. code-block:: bash

   uv run bandit -r src/ -c pyproject.toml

Configuration: ``pyproject.toml`` → ``[tool.bandit]``

Coverage
~~~~~~~~

Code coverage measurement.

.. code-block:: bash

   uv run pytest --cov=src/meeg_utils --cov-report=html

Configuration: ``pyproject.toml`` → ``[tool.coverage]``

Branch Protection
-----------------

Recommended settings for ``main`` branch:

* ✅ Require status checks to pass before merging
* ✅ Require branches to be up to date before merging
* ✅ Required checks: ``lint``, ``type-check``, ``security``, ``test``, ``docs``
* ✅ Require pull request before merging
* ✅ Require approvals: 1

Secrets Configuration
---------------------

GitHub Secrets (Settings → Secrets and variables → Actions):

* ``CODECOV_TOKEN`` - For coverage reporting (optional)
* PyPI publishing uses Trusted Publishers (no token needed)

Badges
------

Add to README.md:

.. code-block:: markdown

   [![CI](https://github.com/colehank/meeg-utils/workflows/CI/badge.svg)](https://github.com/colehank/meeg-utils/actions)
   [![codecov](https://codecov.io/gh/colehank/meeg-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/colehank/meeg-utils)
   [![PyPI version](https://badge.fury.io/py/meeg-utils.svg)](https://badge.fury.io/py/meeg-utils)
