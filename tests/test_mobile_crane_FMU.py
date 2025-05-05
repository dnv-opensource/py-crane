import logging
from math import cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pytest
from example.mobile_crane import MobileCrane
from fmpy import dump, plot_result, simulate_fmu
from fmpy.validation import validate_fmu
from matplotlib.pyplot import set_loglevel
from component_model.utils.xml import read_xml

np.set_printoptions(formatter={"float_kind": "{:.4f}".format})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)
set_loglevel(level="warning")


def arrays_equal(arr1, arr2, eps=1e-7):
    assert len(arr1) == len(arr2), "Length not equal!"

    for i in range(len(arr1)):
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


def mass_center(xs: tuple):
    """Calculate the total mass center of a number of point masses provided as 4-tuple"""
    M, c = 0.0, np.array((0, 0, 0), float)
    for x in xs:
        M += x[0]
        c += x[0] * np.array(x[1:], float)
    return (M, c / M)


def get_result_column( name:str, fmu: Path):
        ell = read_xml(fmu).findall('.//ScalarVariable')
        idx = 0
        for el in ell:
            if el.attrib['causality'] == 'output':
                idx += 1
                if el.attrib['name'] == name:
                    return idx
        return None


def test_mass_center():
    def do_test(Mc, _M, _c):
        assert Mc[0] == _M, f"Mass not as expected: {Mc[0]} != {_M}"
        arrays_equal(Mc[1], _c, 1e-10)

    do_test(mass_center(((1, -1, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0))), 4, (0, 0, 0))
    do_test(
        mass_center(((1, 1, 1, 0), (1, 1, -1, 0), (1, -1, -1, 0), (1, -1, 1, 0))),
        4,
        (0, 0, 0),
    )


def make_mobile_crane_fmu(only_path: bool = True):
    if not only_path:
        build_path = Path(__file__).parent.parent / "examples"  # together with other crane files
        build_path.mkdir(exist_ok=True)
        fmu_path = MobileCrane.build(
            str(Path(__file__).parent.parent / "examples" / "mobile_crane.py"),
            project_files=[Path(__file__).parent.parent / "src" / "crane_fmu"],
            dest=build_path,
        )
    else:
        fmu_path = Path(__file__).parent.parent / "examples" / "MobileCrane.fmu"
        assert fmu_path.exists(), "FMU not found"
    return fmu_path


def test_mobilecrane_fmu(mobile_crane_fmu, show: bool = False):
    """The mobileCrane is build within the fixture 'mobile_crane_fmu'.
    Validate the FMU here and dump its interface.
    """
    val = validate_fmu(str(mobile_crane_fmu))
    assert not len(val), (
        f"Validation of the modelDescription of {mobile_crane_fmu.name} was not successful. Errors: {val}"
    )
    if show:
        dump(mobile_crane_fmu)


# @pytest.mark.skip("Run the FMU")
def test_run_mobilecrane_static(mobile_crane_fmu, show: bool):
    result = simulate_fmu(  # static run
        str(mobile_crane_fmu),
        stop_time=0.1,
        step_size=0.1,
        output_interval=0.1,  # if not set, output_interval=DE.stepSize + at least 2 steps
        validate=True,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "pedestal.mass": 10000.0,
            "pedestal.boom[0]": 3.0,
            "pedestal.boom[2]": 0.0,
            "boom.mass": 1000.0,
            "boom.boom[0]": 8,
            "boom.boom[1]": 45.0,  # input as deg, internal: rad
            "rope.boom[0]": 1e-6,
        },
    )
    # result is a list of tuples. Each tuple contains (time, output-variables)
    # assert abs(result[0][19] - 8) < 1e-9, f"Default start value {result[0][19]}. Default start value of boom end!"
    assert result[0][0] == 0.0
    assert result[1][0] == 0.1, "This works only if output_interval is properly set (not None)!"
    col = get_result_column('boom.end[2]', mobile_crane_fmu)
    assert abs(result[1][col] - 3 - 8 / sqrt(2)) < 1e-14, f"Initial setting {result[1][col]} visible only after first step!"
    M, c = mass_center(
        (
            (10000, -1, 0, 1.5),
            (1000, 4 / sqrt(2), 0, 3 + 4 / sqrt(2)),
            (50, 8 / sqrt(2), 0, 3 + 8 / sqrt(2)),
        )
    )


def test_run_mobilecrane_move(mobile_crane_fmu, show: bool):
    result = simulate_fmu(
        str(mobile_crane_fmu),
        stop_time=10.0,
        step_size=1,
        output_interval=1.0,
        solver="Euler",
        debug_logging=True,
        logger=print,  # fmi_call_logger=print,
        start_values={
            "pedestal.mass": 10000.0,
            "pedestal.boom[0]": 3.0,
            "pedestal.boom[2]": 0.0,
            "boom.mass": 1000.0,
            "boom.boom[0]": 8,
            "boom.boom[1]": 45.0,  # input as deg, internal: rad
            "der(pedestal.boom[2])": radians(1.0),  # azimuthal movement 1 deg per time step
            "rope.boom[0]": 1e-6,
#            "der(fixation.boom[1])": 0.0,
#            "der(fixation.boom[2])": 0.0,
        },
    )
    if show:
        plot_result(result)
    col = get_result_column( 'boom.end[0]', mobile_crane_fmu)
    for i, row in enumerate(result):
        assert abs(row[0] - i) < 1e-9
        print(i, row[0], row[col])
        #assert abs(row[col] - 8 / sqrt(2) * cos(radians(row[0]))) < 1e-9
    assert abs(result[10][col+0] - 8 / sqrt(2) * cos(radians(10))) < 1e-9, f"Final position of boom {result[10][col+0]}"
    assert abs(result[10][col+1] - 8 / sqrt(2) * sin(radians(10))) < 1e-9, f"Final position of boom {result[10][col+1]}"
    assert abs(result[10][col+2] - 3 - 8 / sqrt(2)) < 1e-9, f"Final position of boom {result[10][col+2]}"


if __name__ == "__main__":
    retcode = 0#pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    crane_fmu = make_mobile_crane_fmu(only_path=False)
    # test_mass_center()
    # test_mobilecrane_fmu( crane_fmu, show=True)
    # test_run_mobilecrane_static(crane_fmu, show=True)
    test_run_mobilecrane_move(crane_fmu, show=False)
