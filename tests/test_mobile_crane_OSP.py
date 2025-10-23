import logging
import os
from math import cos, degrees, radians, sin, sqrt
from pathlib import Path

import pytest
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState, CosimVariableType
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)


def var_by_name(simulator: CosimExecution, name: str, comp: str | int) -> dict:
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
#                 ("mobileCrane", "pedestal_end[0]"),
#                 ("mobileCrane", "pedestal_end[1]"),
#                 ("mobileCrane", "pedestal_end[2]"),
#             ),
#             (
#                 ("mobileCrane", "boom_end[0]"),
#                 ("mobileCrane", "boom_end[1]"),
#                 ("mobileCrane", "boom_end[2]"),
#             ),
#             (
#                 ("mobileCrane", "wire_end[0]"),
#                 ("mobileCrane", "wire_end[1]"),
#                 ("mobileCrane", "wire_end[2]"),
#             ),
#         ],
#         osp_system_structure="MobileCraneSystemStructure.xml",
#     )


#    def test_visual_simulation_2():
#     simulator = VisualSimulator()
#     simulator.start(
#         points_3d=[
#             (
#                 ("mobileCrane", "pedestal_end[0]"),
#                 ("mobileCrane", "pedestal_end[1]"),
#                 ("mobileCrane", "pedestal_end[2]"),
#             ),
#             (
#                 ("mobileCrane", "wire_end[0]"),
#                 ("mobileCrane", "wire_end[1]"),
#                 ("mobileCrane", "wire_end[2]"),
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
    assert sim.num_slave_variables(0) == 81
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(icrane)}
    assert variables["fixation.mass"] == 0


def test_mobilecrane(mobile_crane_fmu: Path):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def set_initial(name: str, value: float):
        var = var_by_name(simulator, name, _crane)
        simulator.real_initial_value(slave_index=_crane, variable_reference=var["reference"], value=value)

    mobile_crane_system_structure = _mobile_crane_system_structure(mobile_crane_fmu)
    os.chdir(mobile_crane_system_structure.parent)
    logger.info(f"STRUCTURE {mobile_crane_system_structure} @ {os.path.abspath(os.path.curdir)}")
    simulator = CosimExecution.from_osp_config_file(str(mobile_crane_system_structure))
    assert isinstance(simulator, CosimExecution)
    crane = CosimLocalSlave(fmu_path=str(mobile_crane_fmu), instance_name="mobileCrane")
    assert isinstance(crane, CosimLocalSlave)
    _crane = simulator.add_local_slave(crane)
    assert isinstance(_crane, int)
    sim_status = simulator.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    _ = simulator.slave_infos()
    logger.info(f"Status: {simulator.execution_status}")
    _crane = simulator.slave_index_from_instance_name("mobileCrane") or -1
    assert _crane >= 0, "Slave index of 'mobileCrane' not found"
    vars = simulator.slave_variables(_crane)
    for var in vars:
        logger.info(f"Slave variable (crane) ({var.reference}): {var.name}")
    assert vars[0].name.decode() == "fixation.mass"
    assert vars[7].name.decode() == "fixation.torque[0]", f"Found {vars[7].name.decode()}"
    assert vars[46].name.decode() == "boom.end[0]"
    f_t = CosimObserver.create_time_series()
    simulator.add_observer(observer=f_t)
    # boom.end components
    assert f_t.start_time_series(slave_index=_crane, value_reference=46, variable_type=CosimVariableType.REAL)
    assert f_t.start_time_series(slave_index=_crane, value_reference=47, variable_type=CosimVariableType.REAL)
    assert f_t.start_time_series(slave_index=_crane, value_reference=48, variable_type=CosimVariableType.REAL)
    manipulator = CosimManipulator.create_override()
    simulator.add_manipulator(manipulator=manipulator)
    # Variable references and settings (as in test_mobile_crane_fmi):
    set_initial("pedestal.mass", 10000)
    set_initial("pedestal.boom[0]", 3.0)
    set_initial("boom.mass", 1000.0)
    set_initial("boom.boom[0]", 8)
    set_initial("boom.boom[1]", 45)
    set_initial("der(pedestal.boom[2])", 1.0)
    # set_initial("der(fixation.boom[2])", 0.1)

    res = simulator.simulate_until(target_time=10e9)  # automatic stepping with stopTime in nanos (alternative: .step())
    logger.info(f"Ran OSP simulation. Success: {res}")
    time, _, boom_x = f_t.time_series_real_samples(_crane, value_reference=46, from_step=1, sample_count=100)
    time, _, boom_y = f_t.time_series_real_samples(_crane, value_reference=47, from_step=1, sample_count=100)
    time, _, boom_z = f_t.time_series_real_samples(_crane, value_reference=48, from_step=1, sample_count=100)
    for t, b_x, b_y, b_z in zip(time, boom_x, boom_y, boom_z, strict=True):
        angle = radians(t / 1e9)
        assert abs(b_x - 8 / sqrt(2) * cos(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * cos(angle)}"
        assert abs(b_y - 8 / sqrt(2) * sin(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * sin(angle)}"
        assert abs(b_z - 3 - 8 / sqrt(2)) < 1e-9
    logger.info("Simulation finalized")

def test_mobilecrane2(mobile_crane_fmu: Path):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def set_initial(sim:CosimExecution, name: str, value: float):
        var = var_by_name(sim, name, _crane)
        sim.real_initial_value(slave_index=_crane, variable_reference=var["reference"], value=value)

    mobile_crane_system_structure = _mobile_crane_system_structure(mobile_crane_fmu)
    os.chdir(mobile_crane_system_structure.parent)
    logger.info(f"STRUCTURE {mobile_crane_system_structure} @ {os.path.abspath(os.path.curdir)}")
    sim = CosimExecution.from_osp_config_file(str(mobile_crane_system_structure))
    assert isinstance(sim, CosimExecution)
    _crane = sim.slave_index_from_instance_name("mobileCrane")
    assert isinstance(_crane, int)
    sim_status = sim.status()
    assert sim_status.current_time == 0
    assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
    _ = sim.slave_infos()
    logger.info(f"Status: {sim.execution_status}")

    f_t = CosimObserver.create_last_value()
    sim.add_observer(observer=f_t)
    # boom.end components
    manipulator = CosimManipulator.create_override()
    sim.add_manipulator(manipulator=manipulator)
    # Variable references and settings (as in test_mobile_crane_fmi):
    set_initial(sim, "pedestal.mass", 10000)
    set_initial(sim, "pedestal.boom[0]", 3.0)
    set_initial(sim, "boom.mass", 1000.0)
    set_initial(sim, "boom.boom[0]", 8)
    set_initial(sim, "boom.boom[1]", 45)
    set_initial(sim, "der(pedestal.boom[2])", 1.0)
    # set_initial(sim, "der(fixation.boom[2])", 0.1)
    time = []
    boom_x = []
    boom_y = []
    boom_z = []
    
    t = 0
    dt = 1.0
    while t < 10.0:
        t += dt
        assert sim.simulate_until(target_time=t)  # automatic stepping with stopTime in nanos (alternative: .step())
#     time, _, boom_x = f_t.time_series_real_samples(_crane, value_reference=46, from_step=1, sample_count=100)
#     time, _, boom_y = f_t.time_series_real_samples(_crane, value_reference=47, from_step=1, sample_count=100)
#     time, _, boom_z = f_t.time_series_real_samples(_crane, value_reference=48, from_step=1, sample_count=100)
#     for t, b_x, b_y, b_z in zip(time, boom_x, boom_y, boom_z, strict=True):
#         angle = radians(t / 1e9)
#         assert abs(b_x - 8 / sqrt(2) * cos(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * cos(angle)}"
#         assert abs(b_y - 8 / sqrt(2) * sin(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * sin(angle)}"
#         assert abs(b_z - 3 - 8 / sqrt(2)) < 1e-9
    logger.info("Simulation finalized")


if __name__ == "__main__":
    retcode = 0#pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_from_osp( _mobile_crane_fmu())
    # test_mobilecrane(_mobile_crane_fmu())
    test_mobilecrane2(_mobile_crane_fmu())
