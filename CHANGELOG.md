# Changelog

All notable changes to the [crane-fmu] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

-/-


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
[unreleased]: https://github.com/dnv-innersource/crane-fmu/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/dnv-innersource/crane-fmu/releases/tag/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dnv-innersource/crane-fmu/releases/tag/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-innersource/crane-fmu/releases/tag/v0.0.1
[crane-fmu]: https://github.com/dnv-innersource/crane-fmu
