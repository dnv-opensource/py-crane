import logging
import os
from math import cos, degrees, radians, sin, sqrt
from pathlib import Path

#from libcosimpy._internal import libcosimc
#libcosimc()

#import matplotlib.pyplot as plt
import pytest
from libcosimpy.CosimEnums import CosimErrorCode, CosimExecutionState, CosimVariableType
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)


def comp_idx(sim: CosimExecution, comp: str | int) -> int:
    if isinstance(comp, str):
        idx = sim.slave_index_from_instance_name(comp)
        assert idx is not None, f"Component {comp} not found"
    elif isinstance(comp, int):
        assert comp >= 0 and comp <= sim.num_slaves(), f"Invalid comp ID {comp}"
        idx = comp
    else:
        raise AssertionError(f"Unallowed argument {comp} in 'var_by_name'")
    return idx


def var_by_name(sim: CosimExecution, comp: str | int, name: str) -> dict:
    """Get the variable info from variable provided as name.

    Args:
        simulator (CosimExecution): the simulator (CosimExecution object)
        name (str): The variable name
        comp (str, int): the component name or its index within the system model

    Returns
    -------
        A dictionary of variable info: reference, type, causality and variability
    """
    component = comp_idx(sim, comp)
    for idx in range(sim.num_slave_variables(component)):
        struct = sim.slave_variables(component)[idx]
        if struct.name.decode() == name:
            return {
                "reference": struct.reference,
                "type": struct.type,
                "causality": struct.causality,
                "variability": struct.variability,
            }
    raise AssertionError(f"Variable {name} was not found within component {comp}") from None


def set_initial(sim: CosimExecution, comp: str | int, name: str, value: float):
    component = comp_idx(sim, comp)
    var = var_by_name(sim, component, name)
    sim.real_initial_value(slave_index=component, variable_reference=var["reference"], value=value)


def add_trace(sim: CosimExecution, comp_var:tuple) -> dict[str, tuple]:
    traces = {}
    for c,varname in comp_var:     
        component = comp_idx(sim, c)
        var = var_by_name(sim, component, varname)
        assert var is not None, f"Variable {varname} not found in {comp}"
        ref = var["reference"]
        traces.update({varname : (component, ref)})
    return traces


def do_plot(time: list, traces: tuple, data: dict, title: str = "CraneOnSpring"):
    fig, ax = plt.subplots()
    for k, v in data.items():
        if k in traces:
            ax.plot(time, v, label=k)
    ax.legend()
    plt.title(title)
    plt.show()


@pytest.fixture(scope="session")
def mobile_crane_fmu():
    return _get_fmu("MobileCrane.fmu")


@pytest.fixture(scope="session")
def oscillator_fmu():
    return _get_fmu("HarmonicOscillator6D.fmu")


def _get_fmu(fmu_file: str) -> Path:
    fmu = Path(__file__).parent.parent / "examples" / fmu_file
    assert fmu.exists(), f"{fmu_file} file expected at {fmu}. Not found."
    return fmu


@pytest.fixture(scope="session")
def system_structure(mobile_crane_fmu):
    return _system_structure()


def _system_structure():
    return Path(__file__).parent.parent / "examples" / "CraneOnSpringSystemStructure.xml"


# def test_visual_simulation_1():
#     sim = VisualSimulator()
#     sim.start(
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
#     sim.start(
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
    assert sim.num_slave_variables(0) == 81
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(icrane)}
    assert variables["fixation.mass"] == 0


def test_mobilecrane(mobile_crane_fmu: Path, show: bool = False):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def run_simulation(
        dt:float = 1.0,
        t_end:float = 10.0,
        show: bool = False,
        add_initial: list | None = None
        ):
        if add_initial is None:
            add_initial = []
        system_structure = _system_structure()
        os.chdir(system_structure.parent)
        logger.info(f"STRUCTURE {system_structure} @ {os.path.abspath(os.path.curdir)}")
        sim = CosimExecution.from_osp_config_file(str(system_structure))
        assert isinstance(sim, CosimExecution)
        _crane = sim.slave_index_from_instance_name("crane")
        assert isinstance(_crane, int)
        #_osc = sim.slave_index_from_instance_name("osc")
        #assert isinstance(_osc, int)
        if show:
            logger.info(f"Variables of 'crane':{_crane}")
            for var in sim.slave_variables(_crane):
                logger.info(f"Variable {var.name} : {var.reference}")
            #logger.info(f"Variables of 'osc':{_osc}")
            #for var in sim.slave_variables(_osc):
            #    logger.info(f"Variable {var.name} : {var.reference}")

        sim_status = sim.status()
        assert sim_status.current_time == 0
        assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED
        _ = sim.slave_infos()

        observer = CosimObserver.create_last_value()
        logger.info(f"Add observer: {sim.add_observer(observer=observer)}")
    
        traces = add_trace(sim, ((_crane, "boom.end[0]"), (_crane, "boom.end[1]"), (_crane, "boom.end[2]")))
        manipulator = CosimManipulator.create_override()
        sim.add_manipulator(manipulator=manipulator)
        # Variable references and settings (as in test_mobile_crane_fmi):
        set_initial(sim, _crane, "pedestal.mass", 10000)
        set_initial(sim, _crane, "pedestal.boom[0]", 3.0)
        set_initial(sim, _crane, "boom.mass", 1000.0)
        set_initial(sim, _crane, "boom.boom[0]", 8)
        set_initial(sim, _crane, "boom.boom[1]", 45)
        for comp, name, val in add_initial:
            set_initial(sim, comp, name, val)
        # set_initial("der(fixation.boom[2])", 0.1)
        logger.info(traces.keys())
        results = {k : [] for k in traces.keys()}
        times: list = []
        t = 0
        while t <= t_end:
            t += dt
            _res = sim.simulate_until(target_time=1)#t*1e9)  # automatic stepping with stopTime in nanos (alternative: .step())
            logger.info(f"Ran OSP simulation. Success: {_res}")
            times.append(t)
            for k, (c, idx) in traces.items():
                vals = observer.last_real_values(slave_index=c, variable_references=[idx])
                results[k].append(vals[0])
        return (times, results)

    times, results = run_simulation(
        dt = 1.0,
        t_end = 10.0,
        show=True,
        add_initial=[
            ["crane", "der(pedestal.boom[2])", 1.0],
        ],
    )
    return

    for t, b_x, b_y, b_z in zip(
        time, results["boom.end[0]"], results["boom.end[1]"], results["boom.end[2]"], strict=True
    ):
        angle = radians(t / 1e9)
        assert abs(b_x - 8 / sqrt(2) * cos(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * cos(angle)}"
        assert abs(b_y - 8 / sqrt(2) * sin(angle)) < 1e-9, f"@{degrees(angle)}: {b_x} != {8 / sqrt(2) * sin(angle)}"
        assert abs(b_z - 3 - 8 / sqrt(2)) < 1e-9
    print(time, results["boom.end[0]"])
    do_plot( time, ("boom.end[0]", ), results, "Test")
    print("Simulation finalized")


def test_run():
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def run_simulation(
        dt:float = 1.0,
        t_end:float = 10.0,
        show: bool = False,
        add_initial: list | None = None
        ):
        if add_initial is None:
            add_initial = []
        system_structure = Path(__file__).parent.parent / "examples" / "MobileCraneSystemStructure.xml"
#        system_structure = Path(__file__).parent.parent / "examples" / "HarmonicOscillatorSystemStructure.xml"
        os.chdir(system_structure.parent)
        logger.info(f"STRUCTURE {system_structure} @ {os.path.abspath(os.path.curdir)}")
        sim = CosimExecution.from_osp_config_file(str(system_structure))
        assert isinstance(sim, CosimExecution)
        _ = sim.slave_infos()
        _crane = sim.slave_index_from_instance_name("mobileCrane")
#        _crane = sim.slave_index_from_instance_name("osc")
        assert isinstance(_crane, int)
        if show:
            logger.info(f"Variables of 'crane':{_crane}")
            for var in sim.slave_variables(_crane):
                logger.info(f"Variable {var.name} : {var.reference}")

        sim_status = sim.status()
        assert sim_status.current_time == 0
        assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED

        observer = CosimObserver.create_last_value()
        assert sim.add_observer(observer=observer)
        sim.simulate_until(1)
    
        traces = add_trace(sim, ((_crane, "boom.end[0]"), (_crane, "boom.end[1]"), (_crane, "boom.end[2]")))
#        traces = add_trace(sim, ((_crane, "x[0]"), (_crane, "v[0]"), (_crane, "f[0]")))
        manipulator = CosimManipulator.create_override()
        sim.add_manipulator(manipulator=manipulator)
#         # Variable references and settings (as in test_mobile_crane_fmi):
#         set_initial(sim, _crane, "pedestal.mass", 10000)
#         set_initial(sim, _crane, "pedestal.boom[0]", 3.0)
#         set_initial(sim, _crane, "boom.mass", 1000.0)
#         set_initial(sim, _crane, "boom.boom[0]", 8)
#         set_initial(sim, _crane, "boom.boom[1]", 45)
#         for comp, name, val in add_initial:
#             set_initial(sim, comp, name, val)
        # set_initial("der(fixation.boom[2])", 0.1)
#         logger.info(traces.keys())
        results = {k : [] for k in traces.keys()}
        times: list = []
        t = 0
        while t <= t_end:
            t += dt
            _res = sim.simulate_until(target_time=1)#t*1e9)  # automatic stepping with stopTime in nanos (alternative: .step())
            logger.info(f"Ran OSP simulation. Success: {_res}")
            times.append(t)
            for k, (c, idx) in traces.items():
                vals = observer.last_real_values(slave_index=c, variable_references=[idx])
                results[k].append(vals[0])
        return (times, results)

    times, results = run_simulation(
        dt = 1.0,
        t_end = 10.0,
        show=True,
    )


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "True", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # test_from_osp( _get_fmu("MobileCrane.fmu"))
    # test_mobilecrane(_get_fmu("MobileCrane.fmu"), show=True)
    test_run()
