# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import subprocess

latest_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], text=True).strip()

project = "RDAgent"
copyright = "2024, Microsoft"
author = "Microsoft"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinxcontrib.autodoc_pydantic"]

autodoc_member_order = "bysource"

# The suffix of source filenames.
source_suffix = {".rst": "restructuredtext"}

# The encoding of source files.
source_encoding = "utf-8"

# The main toctree document.
master_doc = "index"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = latest_tag
release = latest_tag

# The language for content autogenerated by Sphinx. Refer to documentation for
# a list of supported languages.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["build"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

try:
    import furo

    html_theme = "furo"
    html_theme_options = {
        "navigation_with_keys": True,
    }
except ImportError:
    html_theme = "default"

html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "source_repository": "https://github.com/deviwei/rdagent_test",
    "source_branch": "main",
    "source_directory": "docs/",
}
