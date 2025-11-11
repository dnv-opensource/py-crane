import logging
import os
from math import cos, degrees, radians, sin, sqrt
from pathlib import Path

import pytest
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)


def var_by_name(simulator: CosimExecution, comp: str | int, name: str) -> dict:
    """Get the variable info from variable provided as name.

    Args:
        simulator (CosimExecution): the simulator (CosimExecution object)
        name (str): The variable name
        comp (str, int): the component name or its index within the system model

    Returns
    -------
        A dictionary of variable info: reference, type, causality and variability
    """
    if isinstance(comp, str):
        component = simulator.slave_index_from_instance_name(comp)
        assert component is not None, f"Component {comp} not found"
    elif isinstance(comp, int):
        assert comp >= 0 and comp <= simulator.num_slaves(), f"Invalid comp ID {comp}"
        component = comp
    else:
        raise AssertionError(f"Unallowed argument {comp} in 'var_by_name'")
    for idx in range(simulator.num_slave_variables(component)):
        struct = simulator.slave_variables(component)[idx]
        if struct.name.decode() == name:
            return {
                "reference": struct.reference,
                "type": struct.type,
                "causality": struct.causality,
                "variability": struct.variability,
            }
    raise AssertionError(f"Variable {name} was not found within component {comp}") from None


@pytest.fixture(scope="session")
def mobile_crane_fmu():
    return _mobile_crane_fmu()


def _mobile_crane_fmu():
    from component_model.model import Model

    build_path = Path(__file__).parent.parent / "examples"  # together with other crane files
    build_path.mkdir(exist_ok=True)
    fmu_path = Model.build(  # MobileCrane.build(
        str(Path(__file__).parent.parent / "examples" / "mobile_crane.py"),
        project_files=[Path(__file__).parent.parent / "src" / "crane_fmu"],
        dest=build_path,
    )
    return fmu_path


@pytest.fixture(scope="session")
def mobile_crane_system_structure(mobile_crane_fmu):
    return _mobile_crane_system_structure(_mobile_crane_fmu())


def _mobile_crane_system_structure(_mobile_crane__fmu):
    return Path(__file__).parent.parent / "examples" / "MobileCraneSystemStructure.xml"


# def test_visual_simulation_1():
#     simulator = VisualSimulator()
#     simulator.start(
#         points_3d=[
#             (
#                 ("crane", "pedestal_end[0]"),
#                 ("crane", "pedestal_end[1]"),
#                 ("crane", "pedestal_end[2]"),
#             ),
#             (
#                 ("crane", "boom_end[0]"),
#                 ("crane", "boom_end[1]"),
#                 ("crane", "boom_end[2]"),
#             ),
#             (
#                 ("crane", "wire_end[0]"),
#                 ("crane", "wire_end[1]"),
#                 ("crane", "wire_end[2]"),
#             ),
#         ],
#         osp_system_structure="MobileCraneSystemStructure.xml",
#     )


#    def test_visual_simulation_2():
#     simulator = VisualSimulator()
#     simulator.start(
#         points_3d=[
#             (
#                 ("crane", "pedestal_end[0]"),
#                 ("crane", "pedestal_end[1]"),
#                 ("crane", "pedestal_end[2]"),
#             ),
#             (
#                 ("crane", "wire_end[0]"),
#                 ("crane", "wire_end[1]"),
#                 ("crane", "wire_end[2]"),
#             ),
#         ],
#         osp_system_structure="MobileCraneSystemStructure.xml",
#     )


def test_from_osp(mobile_crane_fmu):
    def get_status(sim):
        status = sim.status()
        return {
            "currentTime": status.current_time,
            "state": CosimExecutionState(status.state).name,
            "error_code": CosimErrorCode(status.error_code).name,
            "real_time_factor": status.real_time_factor,
            "rolling_average_real_time_factor": status.rolling_average_real_time_factor,
            "real_time_factor_target": status.real_time_factor_target,
            "is_real_time_simulation": status.is_real_time_simulation,
            "steps_to_monitor": status.steps_to_monitor,
        }

    sim = CosimExecution.from_step_size(step_size=1e7)  # empty execution object with fixed time step in nanos
    crane = CosimLocalSlave(fmu_path=str(mobile_crane_fmu.absolute()), instance_name="crane")

    icrane = sim.add_local_slave(crane)
    assert icrane == 0, f"local slave number {icrane}"
    info = sim.slave_infos()
    assert info[0].name.decode() == "crane", "The name of the component instance"
    assert info[0].index == 0, "The index of the component instance"
    assert sim.slave_index_from_instance_name("crane") == 0
    assert sim.num_slaves() == 1
    assert sim.num_slave_variables(0) == 65, f"Found #variables:{sim.num_slave_variables(0)}"
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(icrane)}
    assert variables["fixation.mass"] == 0


def test_mobilecrane(mobile_crane_fmu: Path):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def set_initial(sim: CosimExecution, comp: int, name: str, value: float):
        assert isinstance(comp, int)
        var = var_by_name(sim, comp, name)
        sim.real_initial_value(slave_index=comp, variable_reference=var["reference"], value=value)

    mobile_crane_system_structure = _mobile_crane_system_structure(mobile_crane_fmu)
    os.chdir(mobile_crane_system_structure.parent)
    logger.info(f"STRUCTURE {mobile_crane_system_structure} @ {os.path.abspath(os.path.curdir)}")
    sim = CosimExecution.from_osp_config_file(str(mobile_crane_system_structure))
    assert isinstance(sim, CosimExecution)
    _crane = sim.slave_index_from_instance_name("crane")
    if not isinstance(_crane, int):
        raise KeyError("Crane not found in system") from None
    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    _ = sim.slave_infos()
    logger.info(f"Status: {sim.execution_status}")

    observer = CosimObserver.create_last_value()
    sim.add_observer(observer=observer)
    # boom.end components
    manipulator = CosimManipulator.create_override()
    sim.add_manipulator(manipulator=manipulator)
    # Variable references and settings (as in test_mobile_crane_fmi):
    set_initial(sim, _crane, "pedestal.mass", 10000)
    set_initial(sim, _crane, "pedestal.boom[0]", 3.0)
    set_initial(sim, _crane, "boom.mass", 1000.0)
    set_initial(sim, _crane, "boom.boom[0]", 8)
    set_initial(sim, _crane, "boom.boom[1]", 45)
    set_initial(sim, _crane, "der(pedestal.boom[2])", 1.0)
    refs = [var_by_name(sim, _crane, f"boom.end[{i}]")["reference"] for i in range(3)]
    time = []
    boom_end = []

    t = 0.0
    dt = 0.01
    assert var_by_name(sim, 0, "boom.end[0]")["reference"], f"boom.end[0]: {var_by_name(sim, 0, 'boom.end[0]')}"
    while t < 10.0:
        t += dt
        time.append(t)
        assert sim.simulate_until(
            target_time=t * 1e7
        )  # automatic stepping with stopTime in nanos (alternative: .step())
        boom_end.append(observer.last_real_values(slave_index=_crane, variable_references=refs))
        # print(f"@{t}: {boom_end[-1]}")
    for t, [b_x, b_y, b_z] in zip(time, boom_end, strict=True):
        angle = radians(t)
        assert abs(b_x - 8 / sqrt(2) * cos(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * cos(angle)}"
        assert abs(b_y - 8 / sqrt(2) * sin(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * sin(angle)}"
        assert abs(b_z - 3 - 8 / sqrt(2)) < 1e-9
    logger.info("Simulation finalized")


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_from_osp( _mobile_crane_fmu())
    # test_mobilecrane(_mobile_crane_fmu())
