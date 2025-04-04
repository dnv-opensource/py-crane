import logging
import os
from glob import glob
from pathlib import Path
from shutil import rmtree

import pytest
from component_model.model import Model
from pytest import LogCaptureFixture


@pytest.fixture(scope="package", autouse=True)
def chdir():
    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")


@pytest.fixture(scope="package", autouse=True)
def test_dir():
    return Path(__file__).parent.absolute()


output_dirs = [
    "results",
]
output_files = [
    "*test*.pdf",
]


@pytest.fixture(autouse=True)
def default_setup_and_teardown(caplog: LogCaptureFixture):
    _remove_output_dirs_and_files()
    yield
    _remove_output_dirs_and_files()


@pytest.fixture(scope="session")
def mobile_crane_fmu():
    build_path = Path(__file__).parent / "resources"  # together with other crane files
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(
        str(Path(__file__).parent / "resources" / "mobile_crane.py"),
        project_files=[Path(__file__).parent.parent / "src" / "crane_fmu"],
        dest=build_path,
    )

    return fmu_path


def _remove_output_dirs_and_files():
    for folder in output_dirs:
        rmtree(folder, ignore_errors=True)
    for pattern in output_files:
        for file in glob(pattern):
            _file = Path(file)
            _file.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def setup_logging(caplog: LogCaptureFixture):
    caplog.set_level("INFO")
    caplog.clear()


@pytest.fixture(autouse=True)
def logger():
    return logging.getLogger()


def pytest_addoption(parser):
    parser.addoption("--show", action="store", default=False)


@pytest.fixture(scope="session")
def show(request):
    return request.config.getoption("--show") == "True"
