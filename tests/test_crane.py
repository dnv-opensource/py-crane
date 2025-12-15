import logging
from math import cos, radians, sin, sqrt
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
from component_model.utils.analysis import extremum_series
from component_model.utils.controls import Controls
from component_model.utils.transform import rot_from_spherical, rot_from_vectors
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot

from crane_fmu.boom import Boom
from crane_fmu.crane import Crane

# from mpl_toolkits.mplot3d.art3d import Line3D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=4, suppress=True)


def do_show(times: list[float], traces: dict[str, list[list[float]]], selection: tuple[str, ...] | None = None):
    """Plot selected traces."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for label, trace in traces.items():
        if selection is None or label in selection:
            if label not in ("force", "torque"):
                _ = ax1.plot(times, trace, label=label)
            else:
                _ = ax2.plot(times, trace, label=label)
    _ = ax1.legend()
    _ = ax2.legend()
    plt.show()


def set_wire_direction(r: Boom, angles: Sequence, degrees=False):
    """Set the angles of a wire object. Makes only sense for preparation of test cases. Not allowed in Boom class."""
    _angles = np.array(angles)
    if degrees:
        _angles = np.radians(_angles)
    r.boom[1:] = _angles
    assert r.anchor0 is not None
    r.rot(r.anchor0.rot() * rot_from_spherical(_angles))
    r.direction = r.rot().apply(np.array((0, 0, 1), float))
    r.r_v = np.array((0, 0, 0), float)  # reset also the speed


def mass_center(xs: Sequence[Sequence]):
    """Calculate the total mass center of a number of point masses provided as 4-tuple"""
    M, c = 0.0, np.array((0, 0, 0), float)
    for x in xs:
        M += x[0]
        c += x[0] * np.array(x[1], float)
    return (M, c / M)


def test_mass_center():
    def do_test(Mc, _M, _c):
        assert Mc[0] == _M, f"Mass not as expected: {Mc[0]} != {_M}"
        assert np.allclose(Mc[1], _c)

    do_test(mass_center(((1, -1, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0))), 4, (0, 0, 0))
    do_test(mass_center(((1, 1, 1, 0), (1, 1, -1, 0), (1, -1, -1, 0), (1, -1, 1, 0))), 4, (0, 0, 0))
    do_test(mass_center(((1, 1, 1, 0), (1, 1, -1, 0), (1, -1, -1, 0), (1, -1, 1, 0))), 4, (0, 0, 0))


def aligned(p_i):
    """Check whether all points pi are on the same straight line."""
    assert len(p_i) > 2, (
        f"Checking whether points are on the same line should include at least 3 points. Got only {len(p_i)}"
    )
    directions = [p_i[i] - p_i[0] for i in range(1, len(p_i))]
    n_dir0 = directions[0] / np.linalg.norm(directions[0])
    for i in range(1, len(directions)):
        assert np.allclose(n_dir0, directions[i] / np.linalg.norm(directions[i]))


def pendulum_relax(wire: Boom, show: bool, steps: int = 1000, dt: float = 0.01):
    x = []
    for _ in range(steps):  # let the pendulum relax
        wire.calc_statics_dynamics(dt)
        x.append(wire.end[2])
    if show:
        fig, ax = plt.subplots()
        ax.plot(x)
        plt.title("Pendulum relaxation", loc="left")
        plt.show()


@pytest.fixture
def crane(scope="session", autouse=True):
    return _crane()


def _crane():
    crane = Crane()
    _ = crane.add_boom(
        name="pedestal",
        description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
        mass=2000.0,
        mass_center=(0.5, -1, 0.8),
        boom=(3.0, 0, 0),
    )
    _ = crane.add_boom(
        name="boom1",
        description="The first boom. Can be lifted",
        mass=200.0,
        mass_center=0.5,
        boom=(10.0, radians(90), 0),
    )
    _ = crane.add_boom(
        name="wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_rng=(50, 2000),
        mass_center=1.0,
        boom=(0.5, radians(90), 0),
        q_factor=10.0,
    )
    return crane


def test_initial(crane):
    """Test the initial state of the crane."""
    # test general crane issues
    assert isinstance(crane.to_crane_angle, Callable)  # type: ignore [arg-type] # do not know about any other way
    assert np.allclose(crane.to_crane_angle(np.pi / 2 * np.array((1, 1, 1))), np.pi / 2 * np.array((1, -1, -1)))
    # test indexing of booms
    booms = [b.name for b in crane.booms()]
    assert booms == ["fixation", "pedestal", "boom1", "wire"]
    fixation, pedestal, boom1, wire = [b for b in crane.booms()]

    assert crane.boom0.name == "fixation", "Boom0 should be fixation"
    assert np.allclose(crane.boom0.origin, (0.0, 0.0, -1e-10))  # fixation somewhat below surface@ {crane.boom0.origin}
    assert np.allclose(crane.boom0.end, (0, 0, 0))  # fixation end at 0: {crane.boom0.end}
    bs = crane.booms()  # iterator generator for the booms based on crane object
    next(bs)
    assert next(bs).name == "pedestal", "Should be 'pedestal'"
    bs = crane.booms(reverse=True)
    assert next(bs).name == "wire", "First reversed should be 'wire'"
    assert next(bs).name == "boom1", "Next reversed should be 'boom1'"

    assert pedestal[0].name == "pedestal", "pedestal[0] should be 'pedestal'"
    assert pedestal[1].name == "boom1", "pedestal[1] should be 'boom1'"
    assert pedestal[-1].name == "wire", "pedestal[-1] should be 'wire'"
    assert pedestal[-2].name == "boom1", "pedestal[-2] should be 'boom1'"

    # for i,b in enumerate(crane.booms()):
    #    print( f"Boom {i}: {b.name}")
    assert list(crane.booms())[2].name == "boom1", "'boom1' from boom iteration expected"
    assert list(crane.booms(reverse=True))[1].name == "boom1", "'boom1' from reversed boom iteration expected"
    assert pedestal in crane.booms(), "pedestal expected as boom"

    assert pedestal.length == 3.0
    assert boom1.length == 10.0
    assert pedestal.anchor1.name == "boom1"
    assert boom1.anchor1.name == "wire"
    assert pedestal.name == "pedestal"
    assert pedestal.mass == 2000.0, f"Found {pedestal.mass}"
    assert np.allclose(pedestal.origin, (0, 0, 0))
    assert np.allclose(pedestal.direction, (0, 0, 1))
    assert np.allclose(pedestal.c_m, (-1, 0.8, 1.5))
    assert pedestal.length == 3
    assert np.allclose(pedestal.end, boom1.origin)
    assert np.allclose(boom1.origin, (0, 0, 3.0))
    assert np.allclose(boom1.direction, (1, 0, 0))
    assert boom1.length == 10
    assert np.allclose(boom1.end, wire.origin)
    assert np.allclose(wire.origin, (10, 0, 3)), f"Expecte wire.origin: (10,0,3). Found:{wire.origin}"
    assert np.allclose(wire.end, (10, 0, 2.5)), f"Expecte wire.end: (10,0,2.5). Found:{wire.end}"
    assert np.allclose(crane.velocity, (0, 0, 0))

    # Check center of mass calculation
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = crane.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    assert np.allclose(_c, c)

    # simplify crane and perform manual torque calculation
    pedestal.mass_center = (0.5, 0, 0)
    boom1.boom_setter((None, radians(90), None))
    wire.mass = 1e-100
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = crane.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    assert np.allclose(_c, c)
    assert np.allclose(fixation.torque, (0, M * c[0] * 9.81, 0))

    # align booms and perform manual calculation
    pedestal.mass_center = (0.5, 0, 0)
    boom1.boom_setter((None, 0, None))
    wire.mass = 1e-100
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = crane.c_m_sub
    assert abs(_M - 2200) < 1e-9, f"Masses {_M} != {M}"
    c_m = (0, 0, (2000 * 1.5 + 200 * 8) / 2200)
    assert np.allclose(_c, c_m), f"Expected CoM: {c_m}. Found: {_c}"
    assert np.allclose(crane.torque, (0, 0, 0)), "Zero torque, due to 'straight up'"
    assert np.allclose(crane.force, (0, 0, -_M * 9.81)), f"Expected force_z: {9.81 * _M}. Found: {crane.force}"


# @pytest.mark.skip()
def test_pendulum(crane, show: bool = False):
    """Test crane with 1m wire and 50kg load at end as pendulum (in various configurations)."""

    def sim_run(
        t_end: float,
        dt: float = 0.1,
        min_z: float | None = None,
        max_z: float | None = None,
        max_speed: float | None = None,
        show: bool = False,
        title: str = "test_pendulum",
        crane_position: Callable | None = None,
        crane_roll: Callable | None = None,
        crane_pitch: Callable | None = None,
        crane_yaw: Callable | None = None,
    ):
        z_pos: list[float] = []
        speed: list[float] = []
        time: list[float] = []
        _dz0_dt = -1
        z0 = r.end[2]
        crane.current_time = 0.0
        length = r.boom[0]
        while crane.current_time < t_end:
            z0 = r.end[2]
            v = float(np.linalg.norm(r.r_v))
            assert min_z is None or r.end[2] >= min_z, f"@{crane.current_time}. Min z {r.end[2]} < {min_z}"
            assert max_z is None or r.end[2] <= max_z + 1e-6, f"@{crane.current_time}. Max z {r.end[2]} > {max_z}"
            assert max_speed is None or v <= max_speed, f"@{crane.current_time}. Max speed {v} < {max_speed}"
            time.append(crane.current_time)
            z_pos.append(z0)
            speed.append(v)
            crane.current_time += dt
            if crane_position is not None:
                crane.position = np.array((crane_position(crane.current_time), 0, 0), float)
            if crane_roll is not None:
                crane.rotate((crane_roll(crane.current_time), 0, 0), absolute=True)
            if crane_pitch is not None:
                crane.rotate((0, crane_pitch(crane.current_time), 0), absolute=True)
            if crane_yaw is not None:
                crane.rotate((0, 0, crane_yaw(crane.current_time)), absolute=True)
            crane.calc_statics_dynamics(dt)
            assert length == r.boom[0], f"Pendulum length {r.boom[0]} != {length}"

        z_max = [[0.0, z0]]
        z_max.extend(extremum_series(time[10:], z_pos[10:], which="max"))

        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(time, z_pos)
            ax2.plot(time, speed)
            plt.title(title)
            plt.show()
        return (time, z_pos, speed, z_max)

    def maximum(tbl: list[list[float]], col: int = 1):
        """Find maximum in 'col' of table and return row."""
        rmax = [float("-inf")] * len(tbl[0])
        for r in tbl:
            if r[col] > rmax[col]:
                rmax = r
        return rmax

    f, p, b1, r = [b for b in crane.booms()]
    # start with a 1m pendulum through the origin along the x-axis
    b1.boom_setter((2, np.pi, 0))  # boom down
    set_wire_direction(r, (0, 0))  # wire straight down
    r.boom_setter((1.0, None, None))  # wire 1m long
    assert np.allclose(r.origin, (0, 0, 1.0))
    assert np.allclose(r.end, (0, 0, 0)), f"Load expected at (0,0,0). Found {r.end}"

    a0 = np.radians(1.0)
    set_wire_direction(r, (a0, 0))
    assert np.allclose(r.end, (-np.sin(a0), 0, 1 - np.cos(a0))), (
        f"Wire angle {np.degrees(a0)}, load: ({-np.sin(a0)},0,{1 - np.cos(a0)}). Found {r.end}"
    )
    r._damping_time = 1e300  # no damping
    if False:  # show:
        show_crane(crane, True, False, title="start")

    # Start the crane at maximum potential energy of load (1 deg) without damping
    # we should have theta(t) = theta0* cos(w*t) with theta0= 0.0175 (1deg) and w= sqrt(g/L) = 3.132 => T = 2.006
    set_wire_direction(r, (a0, 0))
    assert np.allclose(r.end, (-np.sin(a0), 0, 1 - cos(a0)))
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        min_z=0,
        max_z=1 - np.cos(a0),
        max_speed=np.sqrt(2 * 9.81 / (1 - np.cos(a0))),
        show=False,  # show,
        title=f"test_pendulum. {np.degrees(a0)}deg through origin",
    )

    for t, z in zip(time, z_pos, strict=False):
        theta = a0 * np.cos(np.sqrt(9.81) * t)
        assert abs(z - 1 + np.cos(theta)) < 1e-5, f"@{t}: z={z}. Expected {1.0 - np.cos(theta)}"
    #        print(f"@{t}, theta:{theta}, z:{z}, {1 - np.cos(theta)}")
    last = -np.pi / np.sqrt(9.81)
    for t, _ in z_max:  # Note. the pendulum has 2 max points per period!
        assert abs(t - last - np.pi / np.sqrt(9.81)) < 0.01, f"Period {t}-{last} != {np.pi / np.sqrt(9.81)}"
        last = t

    # same test with q_factor
    set_wire_direction(r, (a0, 0))
    assert np.allclose(r.end, (-np.sin(a0), 0, 1 - cos(a0)))
    r.q_factor = 100
    r._damping_time = 0.5 * sqrt(r.length * r.mass_center[0] / 9.81 * (r.q_factor**2 + 0.25))
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        min_z=0,
        max_z=1 - np.cos(a0),
        max_speed=np.sqrt(2 * 9.81 / (1 - np.cos(a0))),
        show=False,  # show,
        title="test_pendulum. 1deg through origin",
    )
    last_t, last_z = z_max[0]
    for t, z in z_max[1:]:  # Note: The pendulum has 2 max points per period!
        assert abs(t - last_t - np.pi / np.sqrt(9.81)) < 0.025, f"Period {t}-{last_t} != {np.pi / np.sqrt(9.81)}"
        check = abs(np.pi * last_z / (last_z - z) / r.q_factor)
        assert check - 1.0 < 2e-1, f"Q-factor not reproduced: {check}"
        last_t, last_z = (t, z)

    # Move the whole crane according to a sin function in x-direction, causing forced pendulum actions
    # Perform a frequency sweep, clearly exhibiting the resonant behaviour of the crane.
    set_wire_direction(r, (0, 0))
    r._damping_time = 1000  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=100,
        dt=0.01,
        min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=False,  # show,
        title="test_pendulum. Forced acceleration frequency sweep",
        crane_position=lambda t: 0.01 * np.sin((0.1 + t / 20) * t),
    )
    tmax, zmax = maximum(z_max)
    assert abs(0.1 + tmax / 20 - 2 * np.pi / np.sqrt(9.81)) < 0.2, "Close to resonance frequency"
    assert zmax > 0.15, "Amplitude at about resonance."

    # Forced roll movement at about resonance with damping
    set_wire_direction(r, (0, 0))
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=100,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=False,  # show,
        title="test_pendulum. Forced rolling motion",
        crane_roll=lambda t: 0.01 * np.sin((0.1 + t / 20) * t),
    )
    assert max(z[1] for z in z_max) > 0.028, f"Overall max: {max(z[1] for z in z_max)}"
    assert max(z[1] for z in z_max[:5]) < 1e-4, f"Low frequency max: {max(z[1] for z in z_max[:5])}"
    assert max(z[1] for z in z_max[-5:]) < 1e-3, f"High frequency max: {max(z[1] for z in z_max[-5:])}"

    # Forced yaw movement, which does not have effect on load (inertia not modelled)
    set_wire_direction(r, (0, 0))
    crane.rot((0, 0, 0))  # reset crane
    r.pendulum_relax()
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=False,  # show,
        title="test_pendulum. Forced yaw motion",
        crane_yaw=lambda t: 0.01 * np.sin((0.1 + t / 20) * t),
    )
    assert max(z[1] for z in z_max) < 1e-9, f"Overall max: {max(z[1] for z in z_max)}"

    # roll the whole crane according to a sin function in x-direction, causing forced pendulum actions
    # Perform a frequency sweep, clearly exhibiting the resonant behaviour of the crane.
    set_wire_direction(r, (0, 0))
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=100,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=False,  # show,
        title="test_pendulum. Forced rolling frequency sweep",
        crane_roll=lambda t: 0.01 * np.sin((0.1 + t / 20) * t),
    )
    assert max(z[1] for z in z_max) > 0.06, f"Overall max: {max(z[1] for z in z_max)}"
    assert max(z[1] for z in z_max[:5]) < 1e-4, f"Low frequency max: {max(z[1] for z in z_max[:5])}"
    assert max(z[1] for z in z_max[-5:]) < 1e-3, f"High frequency max: {max(z[1] for z in z_max[-5:])}"


# @pytest.mark.skip()
def test_sequence(crane, show: bool = False):
    """Test sequence of crane movements and check that position is as intended."""

    def check_rot(r: Rot, vec: Sequence):
        """Compare a rotation object with the expected result when rotating (0,0,1)."""
        rot2 = rot_from_vectors(np.array((0, 0, 1), float), np.array(vec, float))
        if not np.allclose(r.apply(np.array((0, 0, 1))), rot2.apply(np.array((0, 0, 1)))):
            print("Rotation matrices not equal:")
            print(r.as_euler("XYZ", degrees=True))
            print(rot2.as_euler("XYZ", degrees=True))
            print(f"Rotate (0,0,1): {rot2.apply(np.array((0, 0, 1)))} =? {r.apply(np.array((0, 0, 1)))}")
            raise AssertionError("Rotation object error") from None

    f, p, b1, r = [b for b in crane.booms()]
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Initial state. Crane folded")
    assert np.allclose(r.origin + r.length * r.direction, (10, 0, 2.5))
    assert np.allclose(r.end, (10, 0, 2.5))

    logger.info("boom1 up (vertical)")
    b1.boom_setter((None, 0, None))
    assert np.allclose(r.end, (0 + 0.5 / sqrt(2), 0, 3 + 10 - 0.5 / sqrt(2)), atol=0.1)  # somewhat lower due to length
    pendulum_relax(r, show=False)
    assert np.allclose(r.end, (0, 0, 3 + 10 - 0.5), atol=0.05), f"Found equilibrium position {r.end}"

    logger.info("boom1 45 deg up")
    b1.boom_setter((None, radians(45), None))
    r.pendulum_relax()
    assert np.allclose(r.end, [10 / sqrt(2), 0, 3 - 0.5 + 10 / sqrt(2)])
    check_rot(b1._rot, (1.0 / sqrt(2), 0.0, 1.0 / sqrt(2)))
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). boom 1 at 45 degrees.")

    logger.info("wire 0.5m -> 5m")
    r.boom_setter((5.0, None, None))
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Wire 0.5m -> 5m")
    r.pendulum_relax()
    assert np.allclose(r.end, [10 / sqrt(2), 0, 3 + 10 / (sqrt(2)) - 5]), f"Found load position {r.end}"

    logger.info("turn base 360 deg in steps of 5 deg")
    for i in range(73):  # turn in steps of 5 deg
        p.boom_setter((None, None, radians(i * 5)))
        assert np.allclose(
            b1.end, (10 / sqrt(2) * cos(radians(i * 5)), 10 / sqrt(2) * sin(radians(i * 5)), 3 + 10 / sqrt(2))
        )
        check_rot(b1._rot, (1.0 / sqrt(2) * cos(radians(i * 5)), 1.0 / sqrt(2) * sin(radians(i * 5)), 1.0 / sqrt(2)))

    check_rot(b1._rot, (1.0 / sqrt(2), 0.0, 1.0 / sqrt(2)))  # back to starting point
    r.pendulum_relax()
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Turned base 360 degrees")

    logger.info("roll crane 10 deg and 20 deg and back")
    crane.rot((radians(10), 0, 0))
    check_rot(b1._rot, (1.0 / sqrt(2), -1.0 / sqrt(2) * sin(radians(10)), 1.0 / sqrt(2) * cos(radians(10))))
    assert np.allclose(
        b1.end, (10 / sqrt(2), -(3 + 10 / sqrt(2)) * sin(radians(10)), (3 + 10 / sqrt(2)) * cos(radians(10)))
    ), f"Found {b1.end}"
    crane.rot((radians(20), 0, 0))  # these rotations are absolute, not relative
    check_rot(b1._rot, (1.0 / sqrt(2), -1.0 / sqrt(2) * sin(radians(20)), 1.0 / sqrt(2) * cos(radians(20))))
    assert np.allclose(
        b1.end, (10 / sqrt(2), -(3 + 10 / sqrt(2)) * sin(radians(20)), (3 + 10 / sqrt(2)) * cos(radians(20)))
    ), f"Found {b1.end}"
    crane.rot((0, 0, 0))
    check_rot(b1._rot, (1.0 / sqrt(2), 0.0, 1.0 / sqrt(2)))  # back to starting point

    len_0 = r.length
    # boom1 up. Dynamic
    for i in range(450):
        angle = 45 - i / 100
        b1.boom_setter((None, angle, None))
        crane.calc_statics_dynamics(1.0)
        # print(f"angle {angle}, rope length: {r.length}, rope origin: {r.origin}. rope velocity: {r.velocity}")
    assert len_0 == r.length, "Length of rope has changed!"


# @pytest.mark.skip()
def test_change_length(crane, show: bool = False):
    f, p, b1, r = [b for b in crane.booms()]
    if show:
        show_crane(crane, title="test_change_length(). Initial")
    assert r.anchor1 is None
    assert np.allclose(b1.end, r.origin)
    r.boom_setter((3, None, None))  # increase length
    r.pendulum_relax()
    assert np.allclose(r.end, (10.0, 0, 0), atol=0.001)
    if show:
        show_crane(crane, title="test_change_length(). rope -> 3m")


# @pytest.mark.skip()
def test_rotation(crane, show: bool = False):
    f, p, b1, r = [b for b in crane.booms()]
    b1.boom_setter((None, 0, None))  # b1 up
    assert np.allclose(b1.direction, (0, 0, 1))
    assert b1.length == 10
    assert np.allclose(b1.c_m, (0, 0, 5))  # measured relative to its origin!
    assert np.allclose(r.origin, (0, 0, 3 + 10))
    assert abs(r.length - 0.5) < 1e-10, f"Unexpected length {r.length}"
    b1.boom_setter((None, radians(90), None))  # b1 east (as initially)
    r.pendulum_relax()
    assert np.allclose(b1.direction, (1, 0, 0))
    assert b1.length == 10
    assert np.allclose(b1.c_m, (5, 0, 0))
    crane.calc_statics_dynamics()
    com = (50 * np.array((10, 0, 2.5)) + 200 * np.array((5, 0, 3)) + 2000 * np.array((-1, 0.8, 1.5))) / 2250
    assert np.allclose(crane.c_m_sub[1], com), f"{crane.c_m_sub[1]} != {com}"
    torque = 2250 * np.cross(com, (0, 0, -9.81))
    assert np.allclose(crane.torque, torque), f"{crane.torque} != {torque}"

    p.boom_setter((None, None, radians(-90)))  # turn p so that b1 south
    assert np.allclose(b1.direction, (0, -1, 0))
    assert np.allclose(b1.c_m, (0, -5, 0))
    r.pendulum_relax()
    crane.calc_statics_dynamics()
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_rotation(). Before rotation. b1 east.")
    com = (50 * np.array((0, -10, 2.5)) + 200 * np.array((0, -5, 3)) + 2000 * np.array((-1, 0.8, 1.5))) / 2250
    assert np.allclose(crane.c_m_sub[1], com), f"{crane.c_m_sub[1]} != {com}"
    torque = 2250 * np.cross(com, (0, 0, -9.81))
    assert np.allclose(crane.torque, torque), f"{crane.torque} != {torque}"


# @pytest.mark.skip()
def test_c_m(crane, show):
    # Note: Boom.c_m is a local measure, calculated from Boom.origin
    f, p, b1, r = [b for b in crane.booms()]
    r.mass -= 50
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_c_m(). Initial")
    # initial c_m location
    #        print("Initial c_m:", p.c_m, b1.c_m)
    assert np.allclose(p.c_m, (-1, 0.8, 1.5))  # 2000 kg
    assert np.allclose(b1.c_m, (5, 0, 0))  # 200 kg
    # all booms along a line in z-direction
    b1.boom_setter((None, 0, None))
    assert np.allclose(b1.c_m, (0, 0, 5))
    # update all subsystem center of mass points. Need to do that from last boom!
    crane.calc_statics_dynamics()
    r.pendulum_relax()
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_c_m(). All booms along a line in z-direction")
    assert np.allclose(b1.c_m_sub[1], (0, 0, 3 + 5)), f"b1 CoM: {b1.c_m_sub[1]}"
    assert np.allclose(p.c_m_sub[1], (p.mass * p.c_m + b1.c_m_sub[0] * b1.c_m_sub[1]) / (p.mass + b1.c_m_sub[0]))
    assert p.c_m_sub[0] == 2200
    b1.boom_setter((None, radians(45), None))
    p.boom_setter((None, None, radians(-20)))


def animate_sequence(crane, seq=(), nSteps=10):
    """Generate animation frames for a sequence of rotations. To be used as 'update' argument in FuncAnimation.
    A sequence element consists of a boom and an angle, which then is rotated in nSteps.
    """
    for b, a in seq:
        if b.name == "pedestal":  # azimuthal movement
            db = np.array((0, 0, radians(a / nSteps)), float)
        else:  # polar movement
            db = np.array((0, radians(a / nSteps), 0), float)
        for _ in range(nSteps):
            b.boom_setter(b.boom + db)
            # update all subsystem center of mass points. Need to do that from last boom!
            crane.calc_statics_dynamics(dt=None)
            yield (crane)


# @pytest.mark.skip("Animate crane movement")
def test_animation(crane, show):
    if not show:  # if nothing can be shown, we do not need to run it
        return

    def init():
        """Perform the needed initializations."""
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)  # type: ignore [attr-defined] ## according to matplotlib recommendations
        for b in crane.booms():
            lw = {"pedestal": 15, "rope": 2}.get(b.name, 7)
            lines.append(
                ax.plot(
                    [b.origin[0], b.end[0]],
                    [b.origin[1], b.end[1]],
                    [b.origin[2], b.end[2]],
                    linewidth=lw,
                )
            )

    def update(crane):
        """Based on the updated first boom (i.e. the fixation), draw any desired data"""
        for i, b in enumerate(crane.booms()):
            lines[i][0].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )

    f, p, b1, r = list(crane.booms())
    fig = plt.figure(figsize=(15, 15), layout=None)  # "constrained")
    ax = plt.axes(projection="3d")  # , data=line)
    lines = []

    _ = FuncAnimation(
        fig,
        update,  # type: ignore  ## this is a function!
        frames=animate_sequence(crane, seq=((p, -90), (b1, -45))),  # yields crane object as frame
        init_func=init,  # type: ignore  ## this is a function!
        interval=1000,
        blit=False,
        cache_frame_data=False,
    )
    plt.title("Crane animation", loc="left")
    plt.show()
    # assert np.allclose(r.origin, (0, -15 / sqrt(2), 3 + 15 / sqrt(2)))


def test_animation_control(crane, show):
    if not show:  # if nothing can be shown, we do not need to run it
        return

    def movement(crane, dt: float = 0.01, t_end: float = 10.0):
        """Create movement of the crane through definition and usage of Controls.
        Generaor function which yields updated crane objects.
        time is defined global as a simple way to draw the current time together with the title.
        """
        global time
        # initial definition of controls and start values
        controls = Controls(limit_err=logging.WARNING)  # CRITICAL)
        f, p, b1, r = list(crane.booms())
        controls.append("turn", (None, (-0.31, 0.31), (-0.16, 0.16)))  # free rotation, max 1 turn/20sec, 2sec to max
        controls.append("luff", ((0, 1.58), (-0.18, 0.09), (-0.09, 0.05)))  # 90 deg, 5/-2.5 deg/sec, 2sec to max
        controls.append("boom", ((8, 50), (-0.2, 0.1), (-0.1, 0.05)))  # 8m..50m, 0.1/-0.2 m/sec, 2sec to max
        controls.append("wire", ((0.5, 50), (-0.1, 1.0), (-0.05, 0.1)))  # 0.5m..50m, -0.1/1 m/sec, 2sec to max
        f, p, b1, r = list(crane.booms())
        time = 0.0
        controls.current[2][0] = 8.0  # b1 starts with 8m
        controls.current[1][0] = np.radians(90)  # b1 starts at 90 deg
        controls.current[3][0] = 0.5  # wire length starts 0.5m

        # From time 0 we set three goals
        controls.setgoal("turn", 0, np.radians(90), 0.0)  # turn pedestal 90 deg
        controls.setgoal("luff", 0, radians(45), 0.0)  # luff boom to 45 deg
        controls.setgoal("boom", 1, 0.1, 0.0)  # increase length 0.1m/s
        while True:
            if time < t_end:
                if time > 10 and controls.goals[3] is None:  # Start to increase wire length with 1m/s
                    controls.setgoal("wire", 1, 1.0, 10.0)
                controls.step(time, dt)
                if controls.goals[3] is not None:
                    r.boom_setter((controls.current[3][0], None, None))
                if controls.goals[1] is not None or controls.goals[2] is not None:
                    b1.boom_setter((controls.current[2][0], controls.current[1][0], None))
                if controls.goals[0] is not None:
                    p.boom_setter((None, None, controls.current[0][0]))
                crane.do_step(time, dt)
                yield crane
                time += dt
            else:
                break

    def init():
        """Perform the needed initializations."""
        ax = plt.axes(projection="3d")  # , data=line)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)  # type: ignore [attr-defined] ## according to matplotlib recommendations
        for b in crane.booms():
            lw = {"pedestal": 10, "rope": 1}.get(b.name, 4)
            lines.append(
                ax.plot(
                    [b.origin[0], b.end[0]],
                    [b.origin[1], b.end[1]],
                    [b.origin[2], b.end[2]],
                    linewidth=lw,
                )
            )

        return

    def update(crane):
        """Receives the frames from FuncAnimation with updated crane objects. Draw the booms as lines."""
        global time
        ends = []
        for i, b in enumerate(crane.booms()):
            lines[i][0].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )
            ends.append(b.end)
        plt.title(f"Crane animation ({time:.1f})", loc="left")

    lines = []
    time = 0.0  # noqa: F841  # used in sub-functions!
    fig = plt.figure(figsize=(8, 8), layout=None)  # "constrained")
    _ = FuncAnimation(
        fig,
        update,  # type: ignore  ## this is a function!
        frames=movement(crane, dt=0.1, t_end=20.0),  # yields crane object as frame
        init_func=init,  # type: ignore  ## this is a function!
        interval=200,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    # assert np.allclose(r.origin, (0, -15 / sqrt(2), 3 + 15 / sqrt(2)))


def show_crane(_crane, markCOM=True, markSubCOM=True, title: str | None = None):
    # update all subsystem center of mass points. Need to do that from last boom!
    _crane.calc_statics_dynamics(dt=None)
    fig = plt.figure(figsize=(9, 9), layout="constrained")
    ax = fig.add_subplot(projection="3d")  # Note: this loads Axes3D implicitly
    ax.set_xlim(-10, 10)
    ax.set_xlabel("-> x-axis")
    ax.set_ylim(-11, 11)
    ax.set_ylabel("-> y-axis")
    ax.set_zlim(0, 12)  # type: ignore [attr-defined] ## according to matplotlib recommendations
    ax.set_zlabel("-> z-axis")  # type: ignore [attr-defined] ## according to matplotlib recommendations

    lines = []
    c_ms = []
    subCOMs = []
    for i, b in enumerate(_crane.booms()):
        lw = 10 if i == 0 else 5
        lines.append(
            ax.plot(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
                linewidth=lw,
            )
        )
        if markCOM:
            #                print("SHOW_COM", b.name, b.c_m)
            c_ms.append(
                ax.text(
                    x=b.c_m[0],
                    y=b.c_m[1],
                    z=b.c_m[2],
                    s=str(b.mass),
                    color="black",
                )
            )
        if markSubCOM:
            [_, x] = b.c_m_sub
            #                print("SHOW_SUB_COM", m, x)
            subCOMs.append(ax.plot(x[0], x[1], x[2], marker="*", color="red"))
    if title is not None:
        plt.title(title, loc="left")
    plt.show()


# @pytest.mark.skip()
def test_orientation(crane: Crane, show: int | bool = False):
    """Test orientation settings, where argument is (roll, pitch, yaw).
    Note: roll, pitch, yaw is measured in vessel coordinate system (z down), while result is in crane system!
    """

    def wire_end(x1: Sequence | np.ndarray, y0: Sequence | np.ndarray, length: float = 1.0, relaxed: bool = False):
        if relaxed:
            return y0
        diff = np.array(y0) - np.array(x1)
        return np.array(x1) + length * (diff) / np.linalg.norm(diff)

    def crane_check(
        _p: Sequence, _b1: Sequence, _r: Sequence, show_it: int = 0, title: str = "", relaxed: bool = False
    ):
        """Check the end positions of all booms."""
        logger.info(f"test_orientation {title}")
        assert np.allclose(p.end, _p), f"Expected {_p}, found {p.end}"
        assert np.allclose(b1.end, _b1), f"Expected {_b1}, found {b1.end}"
        assert np.allclose(r.end, wire_end(b1.end, _r, 1.0, relaxed), atol=1e-4), (
            f"Expected {wire_end(b1.end, _r, 1.0, relaxed)}. Found {r.end}"
        )
        if show & show_it >= show_it:
            show_crane(crane, True, False, title=title)

    f, p, b1, r = [b for b in crane.booms()]
    r.boom_setter((1.0, None, None))  # wire 1m
    crane.rotate((0, 0, 0), absolute=False)  # zero turn
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="test_orientation(initial).")
    angle = radians(90)
    # only roll
    crane.rotate((angle, 0, 0), absolute=True)  # roll 90 deg
    crane_check((0, -3, 0), (10, -3, 0), (10, 0, 2), show_it=2, title=f"test_orientation(roll({angle})).")
    # roll back
    crane.rotate((-angle, 0, 0), absolute=False)
    r.pendulum_relax()
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="Crane rotated back")
    # only pitch
    crane.rotate((0, angle, 0), absolute=True)
    crane_check((-3, 0, 0), (-3, 0, 10), (10, 0, 2), show_it=4, title=f"test_orientation(pitch({angle})).")
    # pitch back
    crane.rotate((0, -angle, 0), absolute=False)
    r.pendulum_relax()
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="Crane rotated back")
    # only yaw
    crane.rotate((0, 0, angle), absolute=True)
    crane_check((0, 0, 3), (0, -10, 3), (10, 0, 2), show_it=8, title=f"test_orientation(yaw({angle})).")
    # yaw back
    crane.rotate((0, 0, -angle), absolute=False)
    r.pendulum_relax()
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="Crane rotated back")
    # roll, pitch, yaw successively
    crane.rotate((angle, 0, 0), absolute=True)
    crane.rotate((0, angle, 0), absolute=False)
    crane.rotate((0, 0, angle), absolute=False)
    if show & 16 >= 16:
        show_crane(crane, True, False, title="roll, pitch, yaw successively")
    logger.info("Stepping roll 90x1deg...")
    crane.d_angular = np.array((np.radians(1), 0, 0), float)
    for i in range(90):
        crane.do_step(i, 1.0)
    if show & 32 >= 32:
        show_crane(crane, True, False, title="roll, pitch, yaw successively")
    crane_check((-3, 0, 0), (-3, -10, 0), (-3, -10, -1), show_it=1, title="Crane stepped 90deg roll", relaxed=True)


def test_force_torque(crane, show: bool = False):
    """Check that results for force and torque are correct with respect to crane states and movements."""
    f, p, b1, r = [b for b in crane.booms()]
    p.mass = 100
    p.mass_center = (0.5, 0, 0)
    p.boom_setter((10, None, None))
    b1.mass = 100
    r.mass = 0.0
    # -----------------------------------------------------------------
    logger.info("all booms upwards. Calculate static force and torque")
    b1.boom_setter((10, 0, None))
    crane.calc_statics_dynamics()
    assert np.allclose(b1.end, (0, 0, 20))
    assert p.c_m_sub[0] == 200
    assert np.allclose(p.c_m_sub[1], (0, 0, 10))
    M = p.mass + b1.mass  # the new total mass
    assert np.allclose(crane.force, (0, 0, -9.81 * M)), f"Only gravitational force {-9.81 * M}"
    assert np.allclose(crane.torque, (0, 0, 0)), "Zero torque"
    # ----------------------------------------------------------------------
    logger.info("Constant velocity on crane. No change on force and torque")
    crane.velocity = np.array((9.9, 8.8, 7.7), float)
    crane.do_step(0, 1.0)
    crane.do_step(1, 1.0)
    assert np.allclose(crane.force, (0, 0, -9.81 * M)), f"Force {crane.force} should not be influenced by uniform speed"
    assert np.allclose(crane.torque, (0, 0, 0)), f"Still zero torque {crane.torque}"
    # ----------------------------------------------------------------------
    logger.info("Acceleration on crane. Creates reactive force and torque.")
    crane.d_velocity = np.array((1.1, 2.2, 0), float)
    crane.do_step(2.0, 1.0)
    assert np.allclose(crane.force, (-1.1 * M, -2.2 * M, -9.81 * M)), (
        "Acceleration creates a reactive force {crane.force}"
    )
    assert np.allclose(crane.torque, np.array((-200 * 2.2 * 10, 200 * 1.1 * 10, 0), float))
    # ------------------------------------------------------------------------------------
    logger.info("Crane with horizontal boom and wire down to bottom. Static calculation.")
    crane.position = np.array((0, 0, 0), float)
    crane.velocity = np.array((0, 0, 0), float)
    crane.d_velocity = np.array((0, 0, 0), float)
    b1.boom_setter((None, np.radians(90), 0))
    r.boom_setter((10, None, None))
    set_wire_direction(r, (90, 0), degrees=True)
    r.mass = 50
    r.pendulum_relax()
    crane.do_step(3.0, 1.0)
    assert np.allclose(r.end, (10, 0, 0))
    assert abs(crane.c_m_sub[0] - 250) < 1e-10, f"Mass {crane.c_m_sub[0]}"
    assert np.allclose(crane.force, (0, 0, -9.81 * 250))
    c_m = np.array(((5 * 100 + 10 * 50) / 250, 0, (5 * 100 + 10 * 100) / 250), float)
    assert np.allclose(crane.c_m_sub[1], c_m), f"CoM:{crane.c_m_sub[1]}"
    assert np.allclose(crane.torque, 250 * np.cross(c_m, np.array((0, 0, -9.81), float)))
    # ---------------------------------------------------------------
    logger.info("Sinusoidal acceleration in x-direction: a*sin(w*t)")
    dt = 0.01
    time = 0.0
    _time: list[float] = []
    traces: dict[str, list] = {"xe": [], "load": [], "force": [], "torque": []}
    while time < 20.0:
        crane.d_velocity = np.array((0.1 * np.sin(1.0 * time), 0, 0), float)
        _time.append(time)
        crane.do_step(time, dt)
        traces["xe"].append(r.origin[0])
        traces["load"].append(r.end[0])
        traces["force"].append(crane.force[0])
        traces["torque"].append(crane.torque[1])
        time += dt
    if False:  # show:
        do_show(_time, traces, ("xe", "load", "force", "torque"))
    # ---------------------------------------------------------------
    logger.info("Sinusoidal pitch acceleration: a*sin(w*t)")
    dt = 0.01
    time = 0
    _time = []
    traces = {"xe": [], "load": [], "force": [], "torque": []}
    while time < 20.0:
        crane.d_angular = np.array((0, 0.1 * np.sin(1.0 * time), 0), float)
        _time.append(time)
        crane.do_step(time, dt)
        traces["xe"].append(r.origin[0])
        traces["load"].append(r.end[0])
        traces["force"].append(crane.force[0])
        traces["torque"].append(crane.torque[1])
        time += dt
    if show:
        do_show(_time, traces, ("xe", "load", "force", "torque"))


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel(level="warning")
    parsolog = logging.getLogger("parso")
    parsolog.setLevel(logging.WARNING)
    pillog = logging.getLogger("PIL")
    pillog.setLevel(logging.WARNING)
    # test_mass_center()
    # test_initial(_crane())
    # test_orientation(_crane(), 32)
    # test_pendulum(_crane(), show=True)
    # test_sequence(_crane(), show=True)
    # test_change_length(_crane(), show=True)
    # test_rotation(_crane(), show=True)
    # test_c_m(_crane(), show=True)
    # test_animation(_crane(), show=True)
    test_animation_control(_crane(), show=True)
    # test_force_torque(_crane(),show=True)
