import logging
import sys
from math import cos, isinf, radians, sin, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest  # noqa: F401
from component_model.utils.xml import read_xml
from component_model.variable import Variable
from fmpy import dump, plot_result, simulate_fmu
from fmpy.validation import validate_fmu
from pythonfmu.enums import Fmi2Causality

import py_crane  # noqa: F401  # Ensure import machinery of the `py_crane` package has run before `Model.build` reaches out directly into modules inside the package.

np.set_printoptions(formatter={"float_kind": "{:.4f}".format})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)


def mass_center(xs: tuple[tuple[float, ...], ...]) -> tuple[float, np.ndarray]:
    """Calculate the total mass center of a number of point masses provided as 4-tuple"""
    M, c = 0.0, np.array((0, 0, 0), float)
    for x in xs:
        M += x[0]
        c += x[0] * np.array(x[1:], float)
    return (M, c / M)


def get_result_column(name: str, fmu: Path):
    ell = read_xml(fmu).findall(".//ScalarVariable")
    idx = 0
    for el in ell:
        if el.attrib["causality"] == "output":
            idx += 1
            if el.attrib["name"] == name:
                return idx
    return None


def test_mass_center():
    def do_test(Mc: tuple[float, np.ndarray], _M: float, _c: tuple[float, ...]):
        assert Mc[0] == _M, f"Mass not as expected: {Mc[0]} != {_M}"
        np.allclose(Mc[1], _c)

    do_test(mass_center(((1, -1, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0))), 4, (0, 0, 0))
    do_test(
        mass_center(((1, 1, 1, 0), (1, 1, -1, 0), (1, -1, -1, 0), (1, -1, 1, 0))),
        4,
        (0, 0, 0),
    )


def _mobile_crane_fmu() -> Path:
    from component_model.model import Model

    build_path = Path(__file__).parent.parent / "examples"  # together with other crane files
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(  # MobileCrane.build(
        str(Path(__file__).parent.parent / "src" / "py_crane" / "mobile_crane.py"),
        project_files=[Path(__file__).parent.parent / "src" / "py_crane"],
        dest=build_path,
    )
    return fmu_path


def test_mobilecrane_fmu(mobile_crane_fmu: Path, show: bool = False):
    """The mobileCrane is build within the fixture 'mobile_crane_fmu'.
    Validate the FMU here and dump its interface.
    """
    val = validate_fmu(str(mobile_crane_fmu))
    assert not len(val), (
        f"Validation of the modelDescription of {mobile_crane_fmu.name} was not successful. Errors: {val}"
    )
    if show:
        dump(str(mobile_crane_fmu))


def test_fmu():
    """Test the FMU object itself."""
    sys.path.insert(0, str((Path(__file__).parent.parent).absolute()))
    from py_crane.mobile_crane import MobileCrane

    def test_vals(v: Variable, k: int, val: float, rng: tuple[float, float], typ: type) -> tuple[float, ...]:
        """Identify test values with respect to the value and range."""
        res: tuple[float, ...] = ()
        if rng is None:
            res = (val,)
        else:
            vmin, vmax = rng
            if typ is float:
                if abs(vmax - vmin) < 1e-15:  # fixed value
                    assert abs(val - vmin) < 1e-15, "Value of {v.name}[{k}] shall be fixed to {vmin}. Found {val}"
                    res = (val,)
                else:
                    lval = (val - 10) * 10 if isinf(vmin) else vmin
                    rval = (val + 10) * 10 if isinf(vmax) else vmax
                    if vmin < 0.0 < vmax and val != 0.0:
                        res = (val, lval, rval, 0.0)
                    else:
                        res = (val, lval, rval)
            elif typ is int:  # range is mandatory
                _res = [val]
                if val != vmin:
                    _res.append(vmin)
                if val != vmax:
                    _res.append(vmax)
                res = tuple(_res)
        return res

    fmu = MobileCrane()
    for _i, v in fmu.vars.items():
        if v is not None:
            #             print(f"units:({v.unit[0]}, {v.unit[1]}, {v.unit[2]})")
            #             v.setter( (1, 1, 1))
            #             print(fmu.d_angular)
            #             print(f"getter() after (1,1,1): {v.getter()}")
            #
            val = v.getter()
            for k, x in enumerate(val):
                assert np.issubdtype(type(x), np.floating) or type(x) is v.typ, (
                    f"Wrong variable type detected for {v.name}[{k}]. {type(x)} != {v.typ}"
                )
            if v.causality in (Fmi2Causality.input, Fmi2Causality.parameter):
                for k in range(len(v)):
                    logger.info(test_vals(v, k, val[k], v.range[k].rng, v.typ))
                    if v.name == "wire.boom" and k > 0:
                        pass  # cannot set the angle of the wire
                    else:
                        for newval in test_vals(v, k, val[k], v.range[k].rng, v.typ):
                            arr: list[None | float] = [None] * len(v)
                            arr[k] = newval
                            # print(f"Set {v.name}[{k}] with {arr}")
                            v.setter(arr, -1)
                            readback = v.getter()[k]
                            assert abs(newval - readback) < 1e-12, (
                                f"{readback} != {newval}. Diff {abs(newval - readback)} after {v.name}[{k}] change."
                            )


# @pytest.mark.skip("Run the FMU")
def test_run_mobilecrane_static(mobile_crane_fmu: Path, show: bool = False):
    """Load the FMU and run it using fmpy."""
    logger.info("Start simulation")
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
            "wire.boom[0]": 1,
        },
    )
    # result is a list of tuples. Each tuple contains (time, output-variables)
    # assert abs(result[0][19] - 8) < 1e-9, f"Default start value {result[0][19]}. Default start value of boom end!"
    assert result[0][0] == 0.0
    assert result[1][0] == 0.1, "This works only if output_interval is properly set (not None)!"
    expected_end = (
        (0, 0, 3),  # pedestal
        (8 / np.sqrt(2), 0.0, 3 + 8 / np.sqrt(2)),  # boom
        (8 / np.sqrt(2), 0.0, 3 + 8 / np.sqrt(2) - 1),  # wire
    )
    for i, b in enumerate(("pedestal", "boom", "wire")):
        col = get_result_column(f"{b}.end[0]", mobile_crane_fmu)
        assert col is not None, f"Variable {b}.end[0] not found"
        assert np.allclose(np.array([result[1][col + i] for i in range(3)]), expected_end[i])
    M, c = mass_center(
        (
            (10000, -1, 0, 1.5),
            (1000, 4 / sqrt(2), 0, 3 + 4 / sqrt(2)),
            (50, 8 / sqrt(2), 0, 3 + 8 / sqrt(2)),
        )
    )


def test_run_mobilecrane_move(mobile_crane_fmu: Path, show: bool = False):
    """Run the FMU using fmpy. Initial position is boom1 45 deg up. Pedestal is moved 1deg/s unit for 10s."""

    def check_boom_end(pos: Sequence[float], angle: float):
        expected = np.array((0, 0, 3)) + 8 / sqrt(2) * np.array((cos(radians(angle)), sin(radians(angle)), 1))
        assert np.allclose(expected, pos), f"@{angle}: {pos} != {expected}"

    def run_simulation(
        p_speed: float = 0.0,
        p_acc: float = 0.0,
        show: bool = False,
    ):
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
                # "der(der(pedestal.boom[2]))": 0.1,  # does not work because it fails fmpy validation
                "der(pedestal.boom[2])": p_speed,  # pedestal azimuthal angular speed in degrees/time
                "der(pedestal.boom[2],2)": p_acc,  # pedestal azimuthal angular acceleration in degrees/time**2
                "wire.boom[0]": 1e-6,
                # "d_angular[2]": 1.0,
            },
        )
        if show:
            plot_result(result)
        col = get_result_column("boom.end[0]", mobile_crane_fmu)
        assert col is not None, "Column of 'boom.end[0]' not found"
        dt = 1.0
        for i, row in enumerate(result):
            assert abs(row[0] - i) < 1e-9
            angle = 0.0 + i * dt * p_speed + i * (i - 1) / 2 * p_acc * dt * dt  # results are read before step!
            check_boom_end((row[col], row[col + 1], row[col + 2]), angle=angle)
        #    assert abs(result[10][col + 1] - 8 / sqrt(2) * sin(radians(10))) < 1e-9, f"boom_1(t=10):{result[10][col + 1]}"
        assert abs(result[10][col + 2] - 3 - 8 / sqrt(2)) < 1e-9, f"boom_2(t=10):{result[10][col + 2]}. z constant"

    logger.info("fmpy. simulate crane without movement.")
    run_simulation()
    logger.info("fmpy. simulate crane moving pedestal with constant speed 1.0 deg/s")
    run_simulation(p_speed=1.0, show=show)
    logger.info("fmpy. simulate crane moving pedestal with constant acceleration 0.1 deg/s^2")
    run_simulation(p_acc=0.1, show=show)


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    crane = _mobile_crane_fmu()
    # test_fmu()
    # test_mass_center()
    # test_mobilecrane_fmu( crane, show=True)
    # test_run_mobilecrane_static( crane, show=True)
    test_run_mobilecrane_move(crane, show=True)
