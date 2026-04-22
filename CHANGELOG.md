# Changelog

All notable changes to the [py-crane] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

* -/-


### Dependencies
* .pre-commit-config.yaml: Updated rev of ruff-pre-commit to v0.15.9
* Updated to fmpy>=0.3.29
* Updated to numpy>=2.4
* Updated to plotly>=6.6
* Updated to pytest-cov>=7.1
* Updated to ruff>=0.15.9
* Updated to scipy>=1.17.1
* Updated to sphinx-argparse-cli>=1.21.3


## [0.3.1] - 2026-03-26

### Changed
* GitHub Workflows:
  * Added 'name: Checkout code' to uses of 'actions/checkout', for better readability and consistency across workflow files.
  * Added 'name: Download build artifacts' to uses of 'actions/download-artifact', for better readability and consistency across workflow files.
  * Added 'name: Publish to PyPI' to uses of 'pypa/gh-action-pypi-publish', for better readability and consistency across workflow files.
  * Added 'name: Upload build artifacts' to uses of 'actions/upload-artifact', for better readability and consistency across workflow files.
  * Changed 'uv sync --upgrade' to 'uv sync -U'
  * Ensured that actions 'upload-artifact' and 'download-artifact' uniformly specify 'dist' as (file)name for the artifact uploaded (or downloaded, respectively), for consistency across workflow files.
  * pull_request_to_main.yml and nightly_build.yml: Added 'workflow_dispatch:' in selected workflows to allow manual trigger of the workflow.
  * Removed redundant 'Set up Python' steps (no longer needed, as 'uv sync' will automatically install Python if not present).
  * Replaced 'Build source distribution and wheel' with 'Build source distribution and wheels' (plural) in workflow step names.
  * Replaced 'Run twine check' with 'Check build artifacts' in workflow step names, to better reflect the purpose of the step.
  * Updated the syntax used for the OS and Python matrix in test workflows.
* Sphinx Documentation:
  * Sphinx conf.py: Updated year in copyright statement to 2026
* pyproject.toml:
  * Added default directories to the 'exclude' list for pyright, in section [tool.pyright]
    (Ref note in pyright [docs](https://github.com/microsoft/pylance-release/blob/main/docs/settings/python_analysis_exclude.md#default-behavior)).
* tests/conftest.py, tests/test_crane_on_spring.py: Change destination folder for the call to Model.build(), so that MobileCrane.fmu gets built not in `/examples` but in `/tests/test_working_directory`.
This avoids the need to commit an updated binary in the repository (/examples/MobileCrane.fmu), just because we ran tests.

### Dependencies
* .pre-commit-config.yaml: Updated rev of ruff-pre-commit to v0.15.1
* Updated to docutils>=0.22.4
* Updated to furo>=2025.12
* Updated to mypy>=1.19.1
* Updated to myst-parser>=5.0
* Updated to plotly>=6.5
* Updated to pre-commit>=4.5
* Updated to pyright>=1.1.408
* Updated to pytest>=9.0
* Updated to ruff>=0.15.1
* Updated to scipy>=1.17.0
* Updated to sourcery>=1.43.0
* Updated to sphinx-argparse-cli>=1.20.1
* Updated to sphinx-autodoc-typehints>=3.6
* Updated to sphinx>=9.0
* Updated to sphinxcontrib-mermaid>=2.0


## [0.3.0] - 2026-02-02

### Breaking Change: New package name
* Changed name of the repository, the package, and the associated pypi project from `crane-fmu` to `py-crane`.
* The repository URL changed from <br>
  https://github.com/dnv-opensource/crane-fmu <br>
  to <br>
  https://github.com/dnv-opensource/py-crane
* The documentation URL changed from <br>
  https://dnv-opensource.github.io/crane-fmu/README.html <br>
  to <br>
  https://dnv-opensource.github.io/py-crane/README.html
* The pypi URL for the package has changed from <br>
  https://pypi.org/project/crane-fmu <br>
  to <br>
  https://pypi.org/project/py-crane
* The old project on pypi 'crane-fmu' has been archived.


## [0.2.0] - 2026-02-02

### Added
* Sphinx documentation:
  * Added docs for modules enum.py and animation.py
* Added Visual Studio Code settings

### Changed
* Updated code base with latest changes in python_project_template v0.2.6
* pyproject.toml:
  * Updated supported Python versions to 3.11, 3.12, 3.13, 3.14
  * Updated required Python version to ">= 3.11"
  * Removed deprecated mypy plugin 'numpy.typing.mypy_plugin'
  * Removed leading carets and trailing slashes from 'exclude' paths
  * Removed `[project.scripts]` section because as far as I can see there are no command line scripts contained in the package
* ruff.toml:
  * Updated target Python version to "py311"
* .sourcery.yaml:
  *  Updated the lowest Python version the project supports to '3.11'
* GitHub workflow _test.yml:
  * Updated Python versions in test matrix to 3.11, 3.12, 3.13, 3.14
* GitHub workflow _test_future.yml:
  * Updated Python version in test_future to 3.15.0-alpha - 3.15.0
* GitHub workflow _build_and_publish_documentation.yml:
  * Changed 'uv sync --upgrade' to 'uv sync --frozen' to avoid unintentional package upgrades.
* Sphinx documentation:
  * Updated toctree
  * conf.py: Updated, and removed ruff rule exception on file level

### Dependencies
* Updated to ruff>=0.14.3  (from ruff>=0.6.3)
* Updated to pyright>=1.1.407  (from pyright>=1.1.378)
* Updated to sourcery>=1.40  (from sourcery>=1.22)
* Updated to numpy>=2.3  (from numpy>=2.0)
* Updated to scipy>=1.16  (from scipy>=1.15.1)
* Updated to matplotlib>=3.10  (from matplotlib>=3.10.7)
* Updated to plotly>=6.3  (from plotly>=6.0.1)
* Updated to pytest>=8.4  (from pytest>=8.3)
* Updated to pytest-cov>=7.0  (from pytest-cov>=5.0)
* Updated to Sphinx>=8.2  (from Sphinx>=8.0)
* Updated to sphinx-argparse-cli>=1.20  (from sphinx-argparse-cli>=1.17)
* Updated to sphinx-autodoc-typehints>=3.5  (from sphinx-autodoc-typehints>=2.2)
* Updated to furo>=2025.9  (from furo>=2024.8)
* Updated to pre-commit>=4.3  (from pre-commit>=3.8)
* Updated to mypy>=1.18  (from mypy>=1.11.1)
* Updated to checkout@v5  (from checkout@v4)
* Updated to setup-python@v6  (from setup-python@v5)
* Updated to setup-uv@v7  (from setup-uv@v2)
* Updated to upload-artifact@v5  (from upload-artifact@v4)
* Updated to download-artifact@v5  (from download-artifact@v4)


## [0.1.1] - 2023-12-18

### Changed
* ruff.toml: updated
* pytest.ini : Added option `--duration=10`. <br>
  This will show a table listing the 10 slowest tests at the end of any test session.
* README.md : Added selected paragraphs that were written in latest work in ax-dnv, mvx and axtreme
* pyproject.toml:
  * cleaned up and restructured dependencies
  * Turned 'dev-dependencies' into a dependency group 'dev' in table [dependency-groups]. <br>
    (This is the more modern style to declare project dependencies)
* VS Code settings: Changed "mypy-type-checker.preferDaemon" from 'false' to 'true'
* Sphinx documentation:
  * index.rst : Changed order of toc tree headlines
  * conf.py : updated with latest settings from python_project_template

### Solved
* Sphinx documentation: Resolved issue that documentation of class members was generated twice.
* pre-commit-config.yaml: Corrected how `--fix=auto` gets passed as argument

### Added
* Sphinx documentation: Added extension to support Markdown-based diagrams created with Mermaid.

### Dependencies
* Updated to ruff>=0.8.3  (from ruff>=0.6.3)
* Updated to pyright>=1.1.390  (from pyright>=1.1.378)
* Updated to sourcery>=1.27  (from sourcery>=1.22)
* Updated to jupyter>=1.1  (from jupyter>=1.0)
* Updated to pytest-cov>=6.0  (from pytest-cov>=5.0)
* Updated to Sphinx>=8.1  (from Sphinx>=8.0)
* Updated to sphinx-argparse-cli>=1.19  (from sphinx-argparse-cli>=1.17)
* Updated to sphinx-autodoc-typehints>=2.5  (from sphinx-autodoc-typehints>=2.2)
* Updated to pre-commit>=4.0  (from pre-commit>=3.8)
* Updated to mypy>=1.13  (from mypy>=1.11.1)


## [0.1.0] - 2023-09-27

* Initial release


## [0.0.1] - 2023-02-21

### Added

* added this

### Changed

* changed that

### Dependencies

* updated to some_package_on_pypi>=0.1.0

### Fixed

* fixed issue #12345

### Deprecated

* following features will soon be removed and have been marked as deprecated:
    * function x in module z

### Removed

* following features have been removed:
    * function y in module z


<!-- Markdown link & img dfn's -->
[unreleased]: https://github.com/dnv-opensource/py-crane/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/dnv-opensource/py-crane/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dnv-opensource/py-crane/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/dnv-opensource/py-crane/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dnv-opensource/py-crane/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-opensource/py-crane/releases/tag/v0.0.1
[py-crane]: https://github.com/dnv-opensource/py-crane
