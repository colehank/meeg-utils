"""Sphinx configuration for meeg-utils documentation."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information -----------------------------------------------------
project = "meeg-utils"
copyright = "2026, meeg-utils contributors"
author = "meeg-utils contributors"
from meeg_utils import __version__ as release  # noqa: E402

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "meeg-utils"

html_theme_options = {
    "github_url": "https://github.com/colehank/meeg-utils",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/meeg-utils",
            "icon": "fa-solid fa-box",
        },
    ],
}

html_context = {
    "github_user": "colehank",
    "github_repo": "meeg-utils",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    # scikit-learn's metadata-routing helpers are not part of the meeg-utils API
    "exclude-members": "__weakref__,set_fit_request,set_transform_request,set_output",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "mne": ("https://mne.tools/stable", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# Sphinx-Gallery: the tutorials are scripts run at build time; their figures
# go into the pages, and each page can be downloaded as .py or .ipynb.
# They use MNE's "sample" dataset (downloaded on first use, 1.6 GB).
sphinx_gallery_conf = {
    "examples_dirs": ["../tutorials"],
    "gallery_dirs": ["auto_tutorials"],
    "filename_pattern": r"\.py$",  # run every tutorial
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
    "abort_on_example_error": True,
    "reset_modules": ("matplotlib",),
    "matplotlib_animations": False,
    "image_scrapers": ("matplotlib",),
    "image_srcset": ["2x"],
    "reference_url": {"meeg_utils": None},
    "doc_module": ("meeg_utils",),
    "backreferences_dir": "api/_backreferences",
    "inspect_global_variables": False,
    # the figures are shown as images; do not also print their repr
    "ignore_repr_types": r"matplotlib\.(figure|axes|text)|mne\.viz|dict_values",
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
