import logging
import shutil
import sys, os
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from component_model.model import Model
from component_model.utils.controls import Controls
from crane_fmu.crane import Crane
from crane_fmu.animation import AnimateCrane

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # DEBUG)


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

def ensure_subpath( pkg:str, folder:str) -> bool:
    """Ensure that the path 'pkg/folder' is in sys.path.
    It is expected that pkg is loaded, such that its path is known
    and that 'folder' is a sub-path below pkg, like 'examples'.""" 
    pkg_path = ""
    for p in sys.path:
        if p.endswith( os.path.join( pkg, folder)):
            return True
        elif p.endswith( pkg):
            pkg_path = p
    if pkg_path == "":
        raise ValueError(f"Package {pkg} is not loaded and we cannot include a sub-folder in sys.path") from None
        return False
    sys.path.insert( 0, os.path.join( pkg_path, folder))
    return True


def make_crane_on_spring(
    pM: float = 10000.0, # in kg
    pCoM: tuple = (0.5, -1.0, 0.8), # in m
    pH: float = 3.0, # in m
    bM: float = 1000.0, # in kg
    bL: float = 8.0, # in m
    bA: float = np.radians(90.0), #in radians
    wM: str = 50.0, # in kg
    wL: float = 1e-6,
    wQ: float = 50.0, # dimensionless quality factor
    k: tuple = (1e4,) * 6,
    c: tuple = (0,) * 6,
    m: float = 1e4,
    x0: tuple = (0.0,) * 6,
    v0: tuple = (0.0,) * 6,
):
    """Initialize and return crane, force and oscillator."""
    def calc_force(t: float, x: np.ndarray, v: np.ndarray, crane: Any, dt: float | None = None):
        """Connect the crane to the oscillator.
        Transfer 6D position and 6D speed of oscillator to crane, step crane and return updated force and torque.
        """
        crane.position, crane.angular = np.split(x, 2) #x[:3], x[3:]
        crane.velocity, crane.d_angular = np.split(v, 2) #v[:3], v[3:]
        crane.do_step(t, dt)  # calc_statics_dynamics( dt)
        # print(f"Crane z:{x[2]}, v:{v[2]}, a:{crane.boom0.acceleration[2]} => f:{crane.force[2]}")
        return np.append(crane.force, crane.torque)

    crane = Crane()
    _ = crane.add_boom(
        name="pedestal",
        description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
        mass=pM,
        mass_center=pCoM,
        boom=(pH, 0, 0),
    )
    _ = crane.add_boom(
        name="boom",
        description="The first boom. Can be lifted",
        mass=bM,
        mass_center=0.5,
        boom=(bL, bA, 0),
    )
    _ = crane.add_boom(
        name="wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=wM,  # so far basically the hook
        mass_rng=(50, 2000),
        mass_center=1.0,
        boom=(wL, np.radians(90), 0),
        q_factor=wQ,
    )
    
    assert ensure_subpath("component-model", "examples")
    from oscillator_xd import Force, OscillatorXD  # type: ignore  ## it works for pytest!

    force = Force(dim=6, func=partial(calc_force, crane=crane))
    osc = OscillatorXD(dim=6, k=k, c=c, m=m, force=force)
    assert osc.force is not None
    for i in range(len(x0)):
        osc.x[i] = x0[i]
    for i in range(len(v0)):
        osc.v[i] = v0[i]
        
    return (crane, force, osc)

def test_crane_on_spring(show: bool = False):
    """Test the crane object as force on a 6D oscillator. No FMUs involved."""

    def do_experiment(
        pM: float = 10000.0,
        pCoM: tuple = (0.5, -1.0, 0.8),
        pH: float = 3.0,
        bM: float = 1000.0,
        bL: float = 8.0,
        bA: float = np.radians(90.0),
        wM: float = 50.0,
        wL: float = 1e-6,
        wQ: float = 50.0,
        k: tuple = (1e4,) * 6,
        c: tuple = (0,) * 6,
        m: float = 1e4,
        x0: tuple = (0.0,) * 6,
        v0: tuple = (0.0,) * 6,
        title: str = "Experiment",
        show: bool = show,
    ):
        crane, force, osc = make_crane_on_spring( pM, pCoM, pH, bM, bL,bA, wM, wL, wQ, k, c, m, x0, v0)

        results: dict = {}
        for i in range(3):
            results.update({f"boom.end[{i}]": [], f"f[{i}]": [], f"v[{i}]": []})
        times: list = []
        t = 0.0
        dt = 0.01
        while t <= 3.145:#10.0:
            t += dt
            times.append(t)
            osc.do_step(t, dt) # takes updated force, x and v and calculates updated x and v
            for i in range(3):
                results[f"boom.end[{i}]"].append(crane.boom_by_name("boom").end[i])
                results[f"f[{i}]"].append(osc.force.out[i])
                results[f"v[{i}]"].append(osc.v[i])
                
        for b in crane.booms():
            print(f"{b.name}: {b.direction}")
        print(f"Force: {osc.force.out}")
        print(f"Speed: {osc.v}")

        if show:
            do_plot(times, ("boom.end[0]", "boom.end[1]", "boom.end[2]", "f[1]", "v[1]"), results, title)

    #do_experiment(m=1e4, c=(0.5,) * 6, pCoM=(0.5, 0, 0), bA=np.radians(180), title="Straight crane", show=show)
    do_experiment(m=1e4, c=(0.5,)*6, pCoM = (0.5,0,0), bA = np.radians(90), title = "90deg boom crane", show=show)

def test_controlled_crane_on_spring( show:bool = False):
    """Move the crane-on-spring, which should lead to wobling."""
    def movement(crane, dt: float, t_end: float, osc):
        """Create movement of the crane through definition and usage of Controls.
        Generaor function which yields updated crane objects.
        time is defined global as a simple way to draw the current time together with the title.
        """
        # initial definition of controls and start values
        controls = Controls(limit_err=logging.WARNING)  # CRITICAL)
        f, p, b1, r = list(crane.booms())
        controls.append("turn", (None, (-0.31, 0.31), (-1,1)))#(-0.16, 0.16)))  # free rotation, max 1 turn/20sec, 2sec to max
        controls.append("luff", ((0, 1.58), (-0.18, 0.09), (-0.09, 0.05)))  # 90 deg, 5/-2.5 deg/sec, 2sec to max
        controls.append("boom", ((8, 50), (-0.2, 0.1), (-0.1, 0.05)))  # 8m..50m, 0.1/-0.2 m/sec, 2sec to max
        controls.append("wire", ((0.5, 50), (-0.1, 1.0), (-0.05, 0.1)))  # 0.5m..50m, -0.1/1 m/sec, 2sec to max
        f, p, b1, r = list(crane.booms())
        controls.current[2][0] = 8.0  # b1 starts with 8m
        controls.current[1][0] = np.radians(90)  # b1 starts at 90 deg
        controls.current[3][0] = 0.5  # wire length starts 0.5m

        # From time 0 we set three goals
        controls.setgoal("turn", 0, np.radians(90), 0.0)  # turn pedestal 90 deg
        #controls.setgoal("luff", 0, np.radians(45), 0.0)  # luff boom to 45 deg
        #controls.setgoal("boom", 1, 0.1, 0.0)  # increase length 0.1m/s
        for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
            #if time > 10 and controls.goals[3] is None:  # Start to increase wire length with 1m/s
            #    controls.setgoal("wire", 1, 1.0, 10.0)
            controls.step(time, dt)
            if controls.goals[3] is not None:
                r.boom_setter((controls.current[3][0], None, None))
            if controls.goals[1] is not None or controls.goals[2] is not None:
                b1.boom_setter((controls.current[2][0], controls.current[1][0], None))
            if controls.goals[0] is not None:
                p.boom_setter((None, None, controls.current[0][0]))
            crane.do_step(time, dt) # takes updated force, x and v and calculates updated x and v
            #crane.boom0.translate(-crane.boom0.origin)
            yield (time + dt, crane)

    if not show: # nothing to do in this case
        return

    crane, force, osc = make_crane_on_spring( m=1e7, bL=8.0, wL=1.0, wM=1000.0, wQ=10000) # define a default crane on spring

    ani = AnimateCrane(crane, frame_gen=movement, dt=0.01, t_end=1.2, interval=10, osc=osc) # type: ignore  ## It is a Generator!
    ani.do_animation()


if __name__ == "__main__":
    """Run the tests defined here.

    Note: The FMUs are not produced here. Only loaded. To change the FMUs use
    'test_mobile_crane_*.py' for the crane
    'test_oscillator_6dof_fmu.py' from the component-model package for the oscillator
    """
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    # make_fmus()
    # make_mobile_crane_straight()
    # test_mobilecrane(show=True)
    # test_crane_on_spring( show=True)
    test_controlled_crane_on_spring( show=True)
