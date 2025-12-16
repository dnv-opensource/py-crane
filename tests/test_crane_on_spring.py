import logging
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from component_model.model import Model

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


def test_crane_on_spring(show: bool = False):
    """Test the crane object as force on a 6D oscillator. No FMUs involved."""
    cm_path = None
    for p in sys.path:
        if p.endswith("component-model"):
            cm_path = p
            break
    assert cm_path is not None, "Path to component-model package not found. Needed for OscillatorXD."
    sys.path.insert(0, str(Path(cm_path) / "examples"))
    from oscillator_xd import Force, OscillatorXD  # type: ignore  ## it works for pytest!

    sys.path.insert(
        0, str(Path(__file__).parent.parent / "examples")
    )  # need to overwrite. 'examples' twice
    assert Path(sys.path[0]).exists()
    from mobile_crane import MobileCrane  # type: ignore  ## it works for pytest!

    def _force(t: float, x: np.ndarray, v: np.ndarray, crane: Any, dt: float | None = None):
        """Calculate the force that the crane exhibits on the oscillator,
        given the generalized position and velocities.
        """
        crane.position = x[:3]
        crane.angular = x[3:]
        crane.velocity = v[:3]
        crane.d_angular = v[3:]
        crane.do_step(t, dt)  # calc_statics_dynamics( dt)
        # print(f"Crane z:{x[2]}, v:{v[2]}, a:{crane.boom0.acceleration[2]} => f:{crane.force[2]}")
        return np.append(crane.force, crane.torque)

    def do_experiment(
        pM: str = "10000.0 kg",
        pCoM: tuple = (0.5, -1.0, 0.8),
        pH: str = "3.0 m",
        bM: str = "1000.0 kg",
        bL: str = "8 m",
        bA: str = "90deg",
        wM: str = "50kg",
        wL: float = 1e-6,
        k: tuple = (1e4,) * 6,
        c: tuple = (0,) * 6,
        m: float = 1e4,
        x0: tuple = (0.0,) * 6,
        v0: tuple = (0.0,) * 6,
        title: str = "Experiment",
        show: bool = show,
    ):
        crane = MobileCrane(
            pedestalMass=pM,
            pedestalCoM=pCoM,
            pedestalHeight=pH,
            boomMass=bM,
            boomLength0=bL,
            boomAngle=bA,
            wire_mass_range=(wM, "2000 kg"),
            wire_length=wL,
        )

        force = Force(dim=6, func=partial(_force, crane=crane))
        osc = OscillatorXD(dim=6, k=k, c=c, m=m, force=force)
        assert osc.force is not None
        for i in range(len(x0)):
            osc.x[i] = x0[i]
        for i in range(len(v0)):
            osc.v[i] = v0[i]

        results: dict = {}
        for i in range(3):
            results.update({f"boom.end[{i}]": [], f"f[{i}]": [], f"v[{i}]": []})
        times: list = []
        t = 0.0
        dt = 0.01
        while t <= 10.0:
            t += dt
            times.append(t)
            osc.do_step(t, dt)
            for i in range(3):
                results[f"boom.end[{i}]"].append(crane.boom_by_name("boom").end[i])
                results[f"f[{i}]"].append(osc.force.out[i])
                results[f"v[{i}]"].append(osc.v[i])

        if show:
            do_plot(times, ("boom.end[0]", "boom.end[1]", "boom.end[2]", "f[2]", "v[2]"), results, title)

    do_experiment(m=1e4, c=(0.5,) * 6, pCoM=(0.5, 0, 0), bA="180deg", title="Straight crane", show=show)
    # do_experiment( m=1e4, c=(0.5,)*6, pCoM = (0.5,0,0), bA = "90deg", title = "90deg boom crane", show=True)


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
