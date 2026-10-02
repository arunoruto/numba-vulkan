import datetime
import sys
import tomllib
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

ROOT = Path(__file__).parent.parent.parent

# The version is read from pyproject.toml rather than from the package:
# importing numba_vulkan needs a Vulkan loader, which docs builders lack.
with (ROOT / "pyproject.toml").open("rb") as fh:
    _project = tomllib.load(fh)["project"]

# -- Project information -----------------------------------------------------

project = "numba-vulkan"
author = "Mirza Arnaut"
copyright = f"{datetime.datetime.now().year}, {author}"
version = _project["version"]
release = version

# Charts of the collected benchmark results (benchmarks/report.py), drawn
# afresh on every build.
_benchmarks = str(ROOT / "benchmarks")
sys.path.insert(0, _benchmarks)
import report  # noqa: E402

report.report(Path(__file__).parent / "_generated" / "benchmarks")
sys.path.remove(_benchmarks)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",  # For parsing NumPy style docstrings
    "autoapi.extension",  # For automated API documentation
    "myst_parser",  # For using Markdown files
    "sphinx_copybutton",  # For adding a copy button to code blocks
    "sphinx.ext.intersphinx",  # Cross-references to Python/NumPy/Numba
]

templates_path = ["_templates"]
# _generated holds fragments that pages pull in with {include}.
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Extension configurations ------------------------------------------------

# AutoAPI configuration
# AutoAPI parses the sources statically, so the package is never imported.
autoapi_dirs = ["../../src/numba_vulkan"]
autoapi_type = "python"
autoapi_add_toctree_entry = True  # Add generated API docs to the TOC
autoapi_generate_api = True
autoapi_python_class_content = "both"  # Include docstrings for class and __init__
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
]
autoapi_keep_files = False

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Napoleon configuration
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_attr_annotations = True
# Render "Attributes" sections as fields; AutoAPI already emits the
# attributes themselves, and describing them twice is an error.
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_css_files = []
html_context = dict(
    github_user="arunoruto",
    github_repo="numba-vulkan",
    github_version="main",
    doc_path="docs/source/",
)

# PyData Theme Options
html_theme_options = {
    "github_url": f"https://github.com/{html_context['github_user']}/{html_context['github_repo']}",
    "use_edit_page_button": True,
    "logo": {"text": "numba-vulkan", "alt_text": "numba-vulkan"},
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

## Intersphinx Configuration: Set up links to external documentation:
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}
