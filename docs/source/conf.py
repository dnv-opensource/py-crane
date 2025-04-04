# ruff: noqa
# mypy: ignore-errors

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("../_ext"))
sys.path.insert(0, os.path.abspath("../../src/crane_fmu"))
shutil.copyfile("../../README.rst", "readme.rst")


# -- Project information -----------------------------------------------------

project = "Crane FMU"
copyright = "2024, DNV"
author = "Siegfried Eisinger and Jorge Mendez"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # to handle README.md
    "sphinx.ext.todo",
    "get_from_code",
    "spec",
    "sphinx.ext.napoleon",  # to read nupy docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
]
todo_include_todos = True
spec_include_specs = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autoclass_content = "both"  # both __init__ and class docstring

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "classic"  # alabaster'
html_theme_options = {
    "rightsidebar": "false",
    "stickysidebar": "true",
    "relbarbgcolor": "black",
    "body_min_width": "700px",
    "body_max_width": "900px",
    "sidebarwidth": "250px",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_preserve_defaults = True

# todo_include_todos = True
# spec_include_specs = True

# add markdown mermaid support
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3
