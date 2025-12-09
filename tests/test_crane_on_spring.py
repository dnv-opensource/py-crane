import logging
from importlib.machinery import SourceFileLoader
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Sequence, Callable, Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from component_model.model import Model
from libcosimpy.CosimEnums import (
    CosimErrorCode,
    CosimExecutionState,
)
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimSlave import CosimLocalSlave

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

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
        raise AssertionError(f"Unallowed argument {comp} in 'comp_idx'")
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


def add_trace(sim: CosimExecution, comp_var: Sequence, traces: dict | None = None) -> dict[str, tuple]:
    if traces is None:
        traces = {}
    for c, varname in comp_var:
        component = comp_idx(sim, c)
        var = var_by_name(sim, component, varname)
        assert var is not None, f"Variable {varname} not found in {c}"
        ref = var["reference"]
        traces.update({varname: (component, ref)})
    return traces


def check_equal(intro: str, val1, val2, eps=1e-10):
    assert abs(val1 - val2) < eps, f"{intro} {val1} != {val2}"


def do_plot(time: list, traces: tuple, data: dict, title: str = "CraneOnSpring"):
    """Plot selected 'traces' from the 'data' repository, with 'title'."""
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


def make_fmu(build_path: Path, source: str, resource: Path, newargs: dict | None = None):
    """Make single FMU. Non-default arguments allowed."""
    build_path.mkdir(exist_ok=True)
    fmu = Model.build(
        str(build_path / source),
        project_files=[resource],
        dest=build_path,
        newargs=newargs,
    )
    return fmu


def make_fmus():
    """Re-make the FMUs. Only default arguments used here."""
    make_fmu(
        Path(__file__).parent.parent / "examples", "mobile_crane.py", Path(__file__).parent.parent / "src" / "crane_fmu"
    )
    make_fmu(
        Path(__file__).parent.parent.parent / "component-model" / "examples",
        "oscillator_6dof_fmu.py",
        Path(__file__).parent.parent.parent / "component-model" / "src" / "component_model",
    )
    shutil.copy(
        Path(__file__).parent.parent.parent / "component-model" / "examples" / "HarmonicOscillator6D.fmu",
        Path(__file__).parent.parent / "src" / "crane_fmu" / "examples",
    )


def make_mobile_crane_straight():
    fmu = make_fmu(
        Path(__file__).parent.parent / "examples",
        "mobile_crane.py",
        Path(__file__).parent.parent / "src" / "crane_fmu",
        newargs={"pedestalCoM": (0.5, 0, 0), "boomAngle": "0 deg", "wire_length": "5 m"},
    )
    shutil.copy(fmu, fmu.parent / "MobileCraneStraight.fmu")


@pytest.fixture(scope="session")
def oscillator_fmu():
    return _get_fmu("HarmonicOscillator6D.fmu")


def _get_fmu(fmu_file: str) -> Path:
    fmu = Path(__file__).parent.parent / "examples" / fmu_file
    assert fmu.exists(), f"{fmu_file} file expected at {fmu}. Not found."
    return fmu


def _system_structure(mobile_crane_fmu: Path | str):
    return Path(__file__).parent.parent / "examples" / mobile_crane_fmu


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

def test_crane_on_spring(show:bool=False):
    """Test the crane object as force on a 6D oscillator. No FMUs involved."""
#     cm_path = None
#     for p in sys.path:
#         if p.endswith('component-model'):
#             cm_path = p
#             break
#     assert cm_path is not None, "Path to component-model package not found. Needed for OscillatorXD."
#     sys.path.insert(0, str(Path(cm_path) / "examples"))
    from examples.oscillator_xd import OscillatorXD, Force
    sys.path.insert(0, str(Path(__file__).parent.parent / 'examples' / 'mobile_crane.py')) # need to overwrite. 'examples' twice
    assert Path(sys.path[0]).exists()
    from mobile_crane import MobileCrane

    def _force( t:float, x:np.ndarray, v:np.ndarray, crane:Any, dt:float|None = None):
        """Calculate the force that the crane exhibits on the oscillator,
        given the generalized position and velocities.
        """
        crane.position = x[:3]
        crane.angular = x[3:]
        crane.velocity = v[:3]
        crane.d_angular = v[3:]
        crane.do_step( t, dt) #calc_statics_dynamics( dt)
        #print(f"Crane z:{x[2]}, v:{v[2]}, a:{crane.boom0.acceleration[2]} => f:{crane.force[2]}")
        return np.append( crane.force, crane.torque)


    def do_experiment(
        pM: str = "10000.0 kg",
        pCoM: tuple = (0.5, -1.0, 0.8),
        pH: str = "3.0 m",
        bM: str = "1000.0 kg",
        bL: str = "8 m",
        bA: str = "90deg",
        wM: float = "50kg",
        wL: float = 1e-6,
        k:tuple = (1e4,)*6,
        c:tuple = (0,)*6,
        m:float = 1e4,
        x0: tuple = (0.0,)*6,
        v0: tuple = (0.0,)*6,
        title:str = "Experiment",
        show:bool = show,
        ):

        crane = MobileCrane( pedestalMass=pM, pedestalCoM=pCoM, pedestalHeight=pH, boomMass=bM,
                             boomLength0=bL, boomAngle=bA,
                             wire_mass_range=(wM, "2000 kg"), wire_length=wL)
           
        force = Force( dim=6, func=partial(_force, crane=crane))
        osc = OscillatorXD( dim=6, k=k, c=c, m=m, force=force)
        for i in range(len(x0)):
            osc.x[i] = x0[i]
        for i in range( len( v0)):
            osc.v[i] = v0[i]

        results : dict = {}
        for i in range(3):
            results.update( { f"boom.end[{i}]":[], f"f[{i}]":[], f"v[{i}]":[] })
        times: list = []
        t = 0.0
        dt = 0.01
        while t <= 10.0:
            t += dt
            times.append(t)
            osc.do_step( t, dt)
            for i in range(3):
                results[f"boom.end[{i}]"].append( crane.boom_by_name('boom').end[i])
                results[f"f[{i}]"].append( osc.force.out[i])
                results[f"v[{i}]"].append( osc.v[i])
            
        if show:
            do_plot(times, ("boom.end[0]", "boom.end[1]", "boom.end[2]", "f[2]", "v[2]"), results, title)

    do_experiment( m=1e4, c=(0.5,)*6, pCoM = (0.5,0,0), bA = "180deg", title = "Straight crane", show=True)
    # do_experiment( m=1e4, c=(0.5,)*6, pCoM = (0.5,0,0), bA = "90deg", title = "90deg boom crane", show=True)
    
    
    
    
        
        

def test_from_osp():
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
    crane = CosimLocalSlave(fmu_path=str(_get_fmu("MobileCrane.fmu")), instance_name="crane")

    icrane = sim.add_local_slave(crane)
    assert icrane == 0, f"local slave number {icrane}"
    info = sim.slave_infos()
    assert info[0].name.decode() == "crane", "The name of the component instance"
    assert info[0].index == 0, "The index of the component instance"
    assert sim.slave_index_from_instance_name("crane") == 0
    assert sim.num_slaves() == 1
    assert sim.num_slave_variables(0) == 65, f"Number of variables: {sim.num_slave_variables(0)}"
    variables = {var_ref.name.decode(): var_ref.reference for var_ref in sim.slave_variables(icrane)}
    assert variables["fixation.mass"] == 0


#@pytest.mark.skip(reason="libcosimpy creates a failure together with pytest. Not yet resolved")
def test_mobilecrane(show: bool = False):
    """Stand-alone test of MobileCrane.fmu using OSP"""

    def run(
        dt: float = 0.01,
        t_end: float = 10.0,
        osp_config: str = "CraneOnOscillatorSystemStructure.xml",
        show: bool = False,
        add_initial: list | None = None,
        title: str | None = None,
    ):
        if title is None:
            title = f"Test with {osp_config}"
        if add_initial is None:
            add_initial = []
        system_structure = _system_structure(osp_config)
        os.chdir(system_structure.parent)
        logger.info(f"STRUCTURE {system_structure} @ {os.path.abspath(os.path.curdir)}")
        sim = CosimExecution.from_osp_config_file(str(system_structure))
        assert isinstance(sim, CosimExecution)
        _crane = sim.slave_index_from_instance_name("crane")
        _osc = sim.slave_index_from_instance_name("osc")
        _comps = (_crane, _osc)
        assert _comps[0] is None or "Crane" in osp_config, "Component model 'crane' not found"
        assert _comps[1] is None or "Oscillator" in osp_config, "Component model 'osc' not found"
        #         if show:
        #             if _comps[0] is not None:
        #                 logger.info(f"Variables of 'crane':{_crane}")
        #                 for var in sim.slave_variables(_crane):
        #                     logger.info(f"Variable {var.name} : {var.reference}")
        #             if _comps[1] is not None:
        #                 logger.info(f"Variables of 'osc':{_osc}")
        #                 for var in sim.slave_variables(_osc):
        #                    logger.info(f"Variable {var.name} : {var.reference}")

        sim_status = sim.status()
        assert sim_status.current_time == 0
        assert CosimExecutionState(sim_status.state) == CosimExecutionState.STOPPED

        observer = CosimObserver.create_last_value()
        sim.add_observer(observer=observer)

        # Initial settings (similar to test_mobile_crane_fmi):
        _initial: dict = {}
        if _crane is not None:  # crane is part of run
            _initial.update(
                {
                    "pedestal.mass": (_crane, 10000),
                    "pedestal.boom[0]": (_crane, 3.0),
                    "boom.mass": (_crane, 1000.0),
                    "boom.boom[0]": (_crane, 8),
                    "boom.boom[1]": (_crane, 45),
                }
            )
        if _osc is not None:  # oscillator is part of run
            _initial.update({f"m[{i}]": (_osc, 10000) for i in range(6)})
            _initial.update({f"k[{i}]": (_osc, 10000) for i in range(6)})
        _initial.update({name: (c, val) for c, name, val in add_initial})  # can overwrite above before set

        for k, (c, v) in _initial.items():
            set_initial(sim, c, k, v)

        traces: dict[str, tuple] = {}
        if _comps[0] is not None:
            traces = add_trace(sim, [(_crane, f"boom.end[{i}]") for i in range(3)], traces)
            traces = add_trace(sim, [(_crane, f"angular[{i}]") for i in range(3)], traces)
            traces = add_trace(sim, [(_crane, f"pedestal.boom[{i}]") for i in range(3)], traces)
            traces = add_trace(sim, [(_crane, f"torque[{i}]") for i in range(3)], traces)
        if _comps[1] is not None:
            traces = add_trace(sim, [(_osc, f"f[{i}]") for i in range(6)], traces)
            traces = add_trace(sim, [(_osc, f"v[{i}]") for i in range(6)], traces)
        manipulator = CosimManipulator.create_override()
        sim.add_manipulator(manipulator=manipulator)
        results: dict[str, list] = {k: [] for k in traces.keys()}
        times: list = []
        t = 0.0
        while t <= t_end:
            t += dt
            _res = sim.simulate_until(target_time=t * 1e7)  # automatic stepping with stopTime in nanos
            times.append(t)
            for k, (c, idx) in traces.items():
                assert var_by_name(sim, c, k)["reference"] == idx, (
                    f"Something wrong with {k}. {idx} != {var_by_name(sim, c, k)}"
                )
                vals = observer.last_real_values(slave_index=c, variable_references=[idx])
                results[k].append(vals[0])
        if show:
            do_plot(times, ("boom.end[0]", "boom.end[1]", "boom.end[2]", "f[3]", "v[3]"), results, "Test")
        return (times, results)

        times, results = run(
            dt=0.01,
            t_end=10.0,
            osp_config="HarmonicOscillatorSystemStructure.xml",
            show=False,
            add_initial=[("osc", "v[0]", 1.0)],
        )
        for t, v in zip(times, results["v[0]"], strict=True):
            check_equal(f"@{t}: ", v, np.cos(t), 1e-4)

    times, res = run(
        t_end=10,
        dt=0.01,
        osp_config="MobileCraneSystemStructure.xml",
        show=False,
        add_initial=[
            ["crane", "der(pedestal.boom[2])", -5.0],
            ["crane", "d_angular[2]", 5.0],
        ],
    )
    for t, x, y, z, a, p in zip(
        times,
        res["boom.end[0]"],
        res["boom.end[1]"],
        res["boom.end[2]"],
        res["angular[2]"],
        res["pedestal.boom[2]"],
        strict=True,
    ):
        angle = 10 * np.radians(t)
        check_equal(f"@{t}, angles {a}, {p}: ", a - p, np.degrees(angle))
        check_equal(f"@{t}, x: ", x, 8 / np.sqrt(2) * np.cos(angle), 1e-2)
        check_equal(f"@{t}, y: ", y, -8 / np.sqrt(2) * np.sin(angle), 1e-2)
        check_equal(f"@{t}, z: ", z, 3 + 8 / np.sqrt(2), 1e-4)

    if not (Path(__file__).parent.parent / "examples" / "MobileCraneStraight.fmu").exists():
        make_mobile_crane_straight()

    times, res = run(
        t_end=25.0,
        dt=0.01,
        osp_config="CraneOnOscillatorSystemStructure.xml",
        show=show,
        add_initial=[
            ["osc", "v[3]", 1.0],  # linked to rolling angular velocity
            ["osc", "k[3]", 100000.0],  # 10x spring
            #   ["crane", "der(d_angular[0])", 0.1],
        ],
        title="Crane on spring",
    )

    # for t, x, y, z in zip(times, res["boom.end[0]"], res["boom.end[1]"], res["boom.end[2]"], strict=True):
    #    pass

    print("Simulation finalized")

    return


if __name__ == "__main__":
    """Run the tests defined here.

    Note: The FMUs are not produced here. Only loaded. To change the FMUs use
    'test_mobile_crane_*.py' for the crane
    'test_oscillator_6dof_fmu.py' from the component-model package for the oscillator
    """
    retcode = 0#pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # make_fmus()
    # make_mobile_crane_straight()
    # test_from_osp()
    # test_mobilecrane(show=True)
    test_crane_on_spring()
