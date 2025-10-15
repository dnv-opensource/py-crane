import os
from math import radians
from pathlib import Path

import pytest
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState, CosimVariableType
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave


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
        assert comp > 0 and comp <= simulator.num_slaves(), f"Invalid comp ID {comp}"
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
def _mobile_crane_fmu():
    return _get_fmu("MobileCrane.fmu")


@pytest.fixture(scope="session")
def oscillator_fmu():
    return _get_fmu("HarmonicOscillator6D.fmu")


def _get_fmu(fmu_file: str) -> Path:
    fmu = Path(__file__).parent.parent / "examples" / fmu_file
    assert fmu.exists(), f"{fmu_file} file expected at {fmu}. Not found."
    return fmu


@pytest.fixture(scope="session")
def mobile_crane_system_structure(mobile_crane_fmu):
    return _mobile_crane_system_structure(_mobile_crane_fmu())


def _mobile_crane_system_structure(_mobile_crane__fmu):
    return Path(__file__).parent.parent / "examples" / "OspSystemStructure.xml"


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
#         osp_system_structure="OspSystemStructure.xml",
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
#         osp_system_structure="OspSystemStructure.xml",
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
    assert sim.num_slave_variables(0) == 78
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(icrane)}
    assert variables["fixation.mass"] == 0


def test_mobilecrane(mobile_crane_system_structure: Path, mobile_crane_fmu: Path):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def set_initial(name: str, value: float):
        var = var_by_name(simulator, name, _crane)
        simulator.real_initial_value(slave_index=_crane, variable_reference=var["reference"], value=value)

    os.chdir(mobile_crane_system_structure.parent)
    print("STRUCTURE", mobile_crane_system_structure, os.path.abspath(os.path.curdir))
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
    #    print("DIR", dir(simulator))
    #    print( simulator.execution_status)
    _crane = simulator.slave_index_from_instance_name("mobileCrane") or -1
    assert _crane >= 0, "Slave index of 'mobileCrane' not found"
    vars = simulator.slave_variables(_crane)
    for var in vars:
        print(f"Slave variable (crane) ({var.reference}): {var.name}")
    assert vars[0].name.decode() == "fixation.mass"
    assert vars[8].name.decode() == "fixation.torque[1]", f"Found {vars[8].name.decode()}"

    f_t = CosimObserver.create_time_series()
    simulator.add_observer(observer=f_t)
    # 9,10,11 are the torque components, 34 is the boom_angularVelocity
    assert f_t.start_time_series(slave_index=_crane, value_reference=9, variable_type=CosimVariableType.REAL)
    assert f_t.start_time_series(slave_index=_crane, value_reference=10, variable_type=CosimVariableType.REAL)
    assert f_t.start_time_series(slave_index=_crane, value_reference=11, variable_type=CosimVariableType.REAL)
    manipulator = CosimManipulator.create_override()
    simulator.add_manipulator(manipulator=manipulator)
    # Variable references and settings (as in test_mobile_crane_fmi):
    set_initial("pedestal.mass", 10000)
    set_initial("pedestal.boom[0]", 3.0)
    set_initial("boom.mass", 1000.0)
    set_initial("boom.boom[0]", 8)
    set_initial("boom.boom[1]", radians(50))
    set_initial("der(fixation.boom[1])", 0.0)
    set_initial("der(fixation.boom[2])", 0.1)

    res = simulator.simulate_until(target_time=1e9)  # automatic stepping with stopTime in nanos (alternative: .step())
    print(f"Ran OSP simulation. Success: {res}")
    t, s, torque0 = f_t.time_series_real_samples(_crane, value_reference=9, from_step=1, sample_count=11)
    t, s, torque1 = f_t.time_series_real_samples(_crane, value_reference=10, from_step=1, sample_count=11)
    t, s, torque2 = f_t.time_series_real_samples(_crane, value_reference=11, from_step=1, sample_count=11)
    for i in range(len(t)):
        print(f"{t[i] / 1e9}, {torque0[i]}, {torque1[i]}, {torque2[i]}")
    print("Simulation finalized")


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_from_osp( fmu)
    # crane_fmu = _mobile_crane_fmu()
    # test_mobilecrane(_mobile_crane_system_structure(crane_fmu), crane_fmu)
