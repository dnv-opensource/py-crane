import os
import xml.etree.ElementTree as ET  # noqa: N817
from math import radians
from pathlib import Path

import pytest
from libcosimpy.CosimEnums import (
    CosimExecutionState,
    CosimVariableType,
)
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
def mobile_crane_system_structure(mobile_crane_fmu):
    ET.register_namespace("", "http://opensimulationplatform.com/MSMI/OSPSystemStructure")
    tree = ET.parse(Path(__file__).parent.parent / "examples" / "OspSystemStructure.xml")
    root = tree.getroot()

    root[0][0].attrib["source"] = f"../{os.path.basename(mobile_crane_fmu.parent)}/MobileCrane.fmu"

    build_path = Path(__file__).parent.parent / "examples"
    build_path.mkdir(exist_ok=True)
    system_structure_path = build_path / "OspSystemStructure.xml"
    tree.write(system_structure_path)
    return system_structure_path


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
#                 ("mobileCrane", "rope_end[0]"),
#                 ("mobileCrane", "rope_end[1]"),
#                 ("mobileCrane", "rope_end[2]"),
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
#                 ("mobileCrane", "rope_end[0]"),
#                 ("mobileCrane", "rope_end[1]"),
#                 ("mobileCrane", "rope_end[2]"),
#             ),
#         ],
#         osp_system_structure="OspSystemStructure.xml",
#     )


def test_mobilecrane(mobile_crane_system_structure, mobile_crane_fmu):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def set_initial(name: str, value: float):
        var = var_by_name(simulator, name, _crane)
        simulator.real_initial_value(slave_index=_crane, variable_reference=var["reference"], value=value)

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
    assert vars[0].name.decode() == "fixation_mass"
    assert vars[8].name.decode() == "fixation_angularVelocity[1]", f"Found {vars[8].name.decode()}"

    f_t = CosimObserver.create_time_series()
    simulator.add_observer(observer=f_t)
    # 9,10,11 are the torque components, 34 is the boom_angularVelocity
    assert f_t.start_time_series(slave_index=_crane, value_reference=9, variable_type=CosimVariableType.REAL)
    assert f_t.start_time_series(slave_index=_crane, value_reference=10, variable_type=CosimVariableType.REAL)
    assert f_t.start_time_series(slave_index=_crane, value_reference=11, variable_type=CosimVariableType.REAL)
    manipulator = CosimManipulator.create_override()
    simulator.add_manipulator(manipulator=manipulator)
    # Variable references and settings (as in test_mobile_crane_fmi):
    set_initial("pedestal_mass", 10000)
    set_initial("pedestal_boom[0]", 3.0)
    set_initial("boom_mass", 1000.0)
    set_initial("boom_boom[0]", 8)
    set_initial("boom_boom[1]", radians(50))
    set_initial("fixation_angularVelocity[0]", 0.0)
    set_initial("fixation_angularVelocity[1]", 0.1)

    res = simulator.simulate_until(target_time=1e9)  # automatic stepping with stopTime in nanos (alternative: .step())
    print(f"Ran OSP simulation. Success: {res}")
    t, s, torque0 = f_t.time_series_real_samples(_crane, value_reference=9, from_step=1, sample_count=11)
    t, s, torque1 = f_t.time_series_real_samples(_crane, value_reference=10, from_step=1, sample_count=11)
    t, s, torque2 = f_t.time_series_real_samples(_crane, value_reference=11, from_step=1, sample_count=11)
    for i in range(len(t)):
        print(f"{t[i] / 1e9}, {torque0[i]}, {torque1[i]}, {torque2[i]}")
    print("Simulation finalized")


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    structure = Path(__file__).parent.parent / "examples" / "OspSystemStructure.xml"
    fmu = Path(__file__).parent.parent / "examples" / "MobileCrane.fmu"
    test_mobilecrane(structure, fmu)
