Release Process
===============

This document describes how to create a new release of meeg-utils.

Version Numbering
-----------------

We follow `Semantic Versioning <https://semver.org/>`_:

* **MAJOR** version for incompatible API changes
* **MINOR** version for new functionality (backwards-compatible)
* **PATCH** version for bug fixes (backwards-compatible)

Format: ``MAJOR.MINOR.PATCH`` (e.g., ``0.1.0``, ``1.2.3``)

Release Checklist
-----------------

1. Update Version Number
~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``pyproject.toml``:

.. code-block:: toml

   [project]
   name = "meeg-utils"
   version = "0.2.0"  # Update this

2. Update Changelog
~~~~~~~~~~~~~~~~~~~

Edit ``CHANGELOG.rst`` (or create if doesn't exist):

.. code-block:: rst

   Changelog
   =========

   [0.2.0] - 2026-03-02
   ---------------------

   Added
   ~~~~~
   * New feature X
   * New feature Y

   Changed
   ~~~~~~~
   * Improved performance of Z

   Fixed
   ~~~~~
   * Bug in component A

3. Run Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run full test suite
   uv run pytest

   # Check coverage
   uv run pytest --cov=src/meeg_utils

   # Run all quality checks
   uv run ruff check src/ tests/
   uv run mypy src/
   uv run bandit -r src/ -c pyproject.toml

4. Build Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make clean
   make html
   # Check docs/build/html/index.html

5. Commit Changes
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add pyproject.toml CHANGELOG.rst
   git commit -m "chore: bump version to 0.2.0"
   git push origin main

6. Create Git Tag
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git tag v0.2.0
   git push origin v0.2.0

7. Automated Release
~~~~~~~~~~~~~~~~~~~~

GitHub Actions will automatically:

1. Build distribution packages
2. Publish to PyPI
3. Create GitHub Release

Monitor progress at: ``https://github.com/colehank/meeg-utils/actions``

8. Verify Release
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check PyPI
   pip install --upgrade meeg-utils
   python -c "import meeg_utils; print(meeg_utils.__version__)"

   # Check GitHub Release
   # Visit: https://github.com/colehank/meeg-utils/releases

Manual Release (if needed)
---------------------------

If automated release fails:

1. Build Package
~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv build

2. Upload to PyPI
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv publish

3. Create GitHub Release
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   gh release create v0.2.0 \
       --title "Release 0.2.0" \
       --notes "See CHANGELOG.rst for details" \
       dist/*

Pre-release
-----------

For testing before official release:

.. code-block:: bash

   # Update version to include rc
   version = "0.2.0rc1"

   # Create tag
   git tag v0.2.0rc1
   git push origin v0.2.0rc1

   # Upload to Test PyPI
   uv publish --repository testpypi

   # Test installation
   pip install --index-url https://test.pypi.org/simple/ meeg-utils

Post-release
------------

After successful release:

1. Announce on GitHub Discussions
2. Update documentation site
3. Notify users via relevant channels

Hotfix Release
--------------

For critical bug fixes:

1. Create hotfix branch from tag:

   .. code-block:: bash

      git checkout -b hotfix/0.1.1 v0.1.0

2. Make fix and test
3. Update version to patch (e.g., ``0.1.1``)
4. Commit and tag:

   .. code-block:: bash

      git commit -m "fix: critical bug in X"
      git tag v0.1.1
      git push origin v0.1.1

5. Merge back to main:

   .. code-block:: bash

      git checkout main
      git merge hotfix/0.1.1
      git push origin main

Rollback
--------

If a release has critical issues:

1. Mark as yanked on PyPI (doesn't delete, but warns users)
2. Create hotfix release
3. Update GitHub Release notes with warning

Do NOT delete releases or tags.
