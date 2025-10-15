import logging
from math import radians, sqrt
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
from component_model.variable import (
    rot_from_spherical,
)
from matplotlib.animation import FuncAnimation

from crane_fmu.boom import Boom
from crane_fmu.crane import Crane

# from mpl_toolkits.mplot3d.art3d import Line3D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=4, suppress=True)


def set_wire_direction(r: Boom, angles: Sequence, degrees: bool = False):
    """Set the angles of a wire object. Makes only sense for preparation of test cases. Not allowed in Boom class."""
    _angles = np.radians(np.array(angles)) if degrees else np.array(angles)
    r.boom[1:] = _angles
    assert r.anchor0 is not None
    r.rot = r.anchor0.rot * rot_from_spherical(_angles)
    r.direction = r.rot.apply(np.array((0, 0, 1), float))
    r.velocity = np.array((0, 0, 0), float)  # reset also the speed


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
        np.allclose(Mc[1], _c)

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
        np.allclose(n_dir0, directions[i] / np.linalg.norm(directions[i]))


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


# @pytest.fixture
# def crane(scope="module", autouse=True):
#     return _crane()
#
# def _crane():
#     crane = Crane()
#     _ = crane.add_boom(
#         name="pedestal",
#         description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
#         mass=2000.0,
#         mass_center=(0.5, -1, 0.8),
#         boom=(3.0, 0, 0),
#     )
#     _ = crane.add_boom(
#         name="boom1",
#         description="The first boom. Can be lifted",
#         mass=200.0,
#         mass_center=0.5,
#         boom=(10.0, radians(90), 0),
#     )
#     _ = crane.add_boom(
#         name="boom2",
#         description="The second boom. Can be lifted whole range",
#         mass=100.0,
#         mass_center=0.5,
#         boom=(5.0, radians(-180), 0),
#     )
#     _ = crane.add_boom(
#         name="wire",
#         description="The wire fixed to the last boom. Flexible connection",
#         mass=50.0,  # so far basically the hook
#         mass_rng=(50, 2000),
#         mass_center=1.0,
#         boom=(0.5, radians(180), 0),
#         q_factor=100.0,
#     )
#     return crane
@pytest.fixture
def crane(scope="module", autouse=True):
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
    assert np.allclose(crane.to_crane_angle(np.array((90, 90, 90)), degrees=True), np.pi / 2 * np.array((1, -1, -1)))
    # test indexing of booms
    booms = [b.name for b in crane.booms()]
    assert booms == ["fixation", "pedestal", "boom1", "wire"]
    fixation, pedestal, boom1, wire = [b for b in crane.booms()]

    assert crane.boom0.name == "fixation", "Boom0 should be fixation"
    np.allclose(crane.boom0.origin, (0.0, 0.0, -1e-10))  # fixation somewhat below surface@ {crane.boom0.origin}
    np.allclose(crane.boom0.end, (0, 0, 0))  # fixation end at 0: {crane.boom0.end}
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
    np.allclose(pedestal.origin, (0, 0, 0))
    np.allclose(pedestal.direction, (0, 0, 1))
    np.allclose(pedestal.c_m, (-1, 0.8, 1.5))
    assert pedestal.length == 3
    np.allclose(pedestal.end, boom1.origin)
    np.allclose(boom1.origin, (0, 0, 3.0))
    np.allclose(boom1.direction, (1, 0, 0))
    assert boom1.length == 10
    np.allclose(boom1.end, wire.origin)
    np.allclose(wire.origin, (5, 0, 3))
    np.allclose(wire.end, (5, 0, 2.5))
    for b in crane.booms():
        np.allclose(b.velocity, (0, 0, 0))

    # Check center of mass calculation
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = pedestal.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    np.allclose(_c, c)

    # simplify crane and perform manual torque calculation
    pedestal.mass_center = (0.5, 0, 0)
    boom1.boom_setter((None, radians(90), None))
    wire.mass = 1e-100
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = fixation.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    np.allclose(_c, c)
    np.allclose(fixation.torque, (0, M * c[0] * 9.81, 0))

    # align booms and perform manual calculation
    pedestal.mass_center = (0.5, 0, 0)
    boom1.boom_setter((None, 0, None))
    wire.mass = 1e-100
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = pedestal.c_m_sub
    assert abs(_M - 2200) < 1e-9, f"Masses {_M} != {M}"
    np.allclose(_c, (0, 0, (2000 * 1.5 + 200 * 5) / 2200))
    np.allclose(pedestal.torque, (0, 0, 0))


# @pytest.mark.skip()
def test_pendulum(crane, show):
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
        z0 = r.end[2]
        dz0_dt = -1
        z0 = r.end[2]
        z_max: list[list[float]] = [[0, z0]]
        crane.current_time = 0.0
        length = r.boom[0]
        while crane.current_time < t_end:
            v = float(np.linalg.norm(r.velocity))
            assert min_z is None or r.end[2] >= min_z, f"Min z {r.end[2]} < {min_z}"
            assert max_z is None or r.end[2] <= max_z, f"Max z {r.end[2]} > {max_z}"
            assert max_speed is None or v <= max_speed, f"Max speed {v} < {max_speed}"
            time.append(crane.current_time)
            z_pos.append(z0)
            speed.append(v)
            crane.current_time += dt
            if crane_position is not None:
                crane.position = np.array((crane_position(crane.current_time), 0, 0), float)
            if crane_roll is not None:
                crane.rotate((crane_roll(crane.current_time), 0, 0), degrees=True)
            if crane_pitch is not None:
                crane.rotate((0, crane_pitch(crane.current_time), 0), degrees=True)
            if crane_yaw is not None:
                crane.rotate((0, 0, crane_yaw(crane.current_time)), degrees=True)
            crane.calc_statics_dynamics(dt)
            assert length == r.boom[0], f"Pendulum length {r.boom[0]} != {length}"
            # print(np.degrees(r.boom[1]), 5.0*np.cos( np.sqrt(9.81)*crane.current_time))
            if dz0_dt > 0 and r.end[2] < z0:  # sign change + -> -
                z_max.append([crane.current_time, max(z0, r.end[2])])
            dz0_dt = (r.end[2] - z0) / dt
            z0 = r.end[2]
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
    b1.boom_setter((2, np.pi, 0))
    set_wire_direction(r, (0, 0))
    r.boom_setter((1.0, None, None))
    assert np.allclose(r.origin, (0, 0, 1.0))
    assert np.allclose(r.end, (0, 0, 0))

    set_wire_direction(r, (1.0, 0), degrees=True)
    r._damping_time = 1e300  # no damping
    if False:  # show:
        show_crane(crane, True, False, title="start")

    # Start the crane at maximum potential energy of load (1 deg) without damping
    # we should have theta(t) = theta0* cos(w*t) with theta0= 0.0175 (1deg) and w= sqrt(g/L) = 3.132 => T = 2.006
    set_wire_direction(r, (1.0, 0), degrees=True)
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        min_z=0,
        max_z=1 - np.cos(np.radians(1)),
        max_speed=np.sqrt(2 * 9.81 / (1 - np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. 5deg through origin",
    )
    for t, z in zip(time, z_pos, strict=False):
        theta = np.radians(1.0) * np.cos(np.sqrt(9.81) * t)
        assert abs(z - 1 + np.cos(theta)) < 1e-5, f"@{t}: z={z}. Expected {1.0 - np.cos(theta)}"
    #        print(f"@{t}, theta:{theta}, z:{z}, {1 - np.cos(theta)}")
    last = -np.pi / np.sqrt(9.81)
    for t, _ in z_max:  # Note. the pendulum has 2 max points per period!
        assert abs(t - last - np.pi / np.sqrt(9.81)) < 0.01, f"Period {t}-{last} != {np.pi / np.sqrt(9.81)}"
        last = t

    # same test with q_factor
    set_wire_direction(r, (1.0, 0), degrees=True)
    r.q_factor = 100
    r._damping_time = 0.5 * sqrt(r.length * r.mass_center[0] / 9.81 * (r.q_factor**2 + 0.25))
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        min_z=0,
        max_z=1 - np.cos(np.radians(1)),
        max_speed=np.sqrt(2 * 9.81 / (1 - np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. 5deg through origin",
    )
    for i, (t, z) in enumerate(z_max[1:]):  # Note. the pendulum has 2 max points per period!
        last_t, last_z = z_max[i]
        assert abs(t - last_t - np.pi / np.sqrt(9.81)) < 0.01, f"Period {t}-{last} != {np.pi / np.sqrt(9.81)}"
        assert abs(np.pi * last_z / (last_z - z) - r.q_factor) / r.q_factor < 1e-1, "Q-factor not reproduced"

    # Move the whole crane according to a sin function in x-direction, causing forced pendulum actions
    # Perform a frequency sweep, clearly exhibiting the resonant behaviour of the crane.
    set_wire_direction(r, (0, 0), degrees=True)
    r._damping_time = 1000  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=100,
        dt=0.01,
        min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. Forced acceleration frequency sweep",
        crane_position=lambda t: 0.01 * np.sin((0.1 + t / 20) * t),
    )
    tmax, zmax = maximum(z_max)
    assert abs(0.1 + tmax / 20 - 2 * np.pi / np.sqrt(9.81)) < 0.2, "Close to resonance frequency"
    assert zmax > 0.15, "Amplitude at about resonance."

    # Forced roll movement at about resonance with damping
    set_wire_direction(r, (0, 0), degrees=True)
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. Forced rolling motion",
        crane_roll=lambda t: 0.01 * np.sin(2.132 * t),
    )
    _max = maximum(z_max)
    assert _max[0] > 9.3, "Maximum at end of time series"
    assert _max[1] > 0.0006

    # Forced yaw movement, which does not have effect on load (inertia not modelled)
    set_wire_direction(r, (0, 0), degrees=True)
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=10,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. Forced yaw motion",
        crane_yaw=lambda t: 0.01 * np.sin(2.132 * t),
    )
    _max = maximum(z_max)
    assert abs(_max[1] - 0.0) < 1e-10, f"Yaw movement cannot be forced if load starts vertically: {_max}"

    # roll the whole crane according to a sin function in x-direction, causing forced pendulum actions
    # Perform a frequency sweep, clearly exhibiting the resonant behaviour of the crane.
    set_wire_direction(r, (0, 0), degrees=True)
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=100,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. Forced rolling frequency sweep",
        crane_roll=lambda t: 0.01 * np.sin((0.1 + t / 20) * t),
    )
    tmax, zmax = maximum(z_max)
    print(tmax, zmax)
    # assert abs( 0.1+tmax/20 - 2*np.pi/np.sqrt(9.81)) < 0.2, "Close to resonance frequency"

    # both roll, pitch and yaw the crane, switching on pitch at 4th period and yaw at 8th period
    set_wire_direction(r, (0, 0), degrees=True)
    r._damping_time = 100  # damping
    time, z_pos, speed, z_max = sim_run(
        t_end=20,
        dt=0.01,
        # min_z=0,
        # max_z=1-np.cos(np.radians(1)),
        # max_speed=np.sqrt(2*9.81/(1-np.cos(np.radians(1)))),
        show=show,
        title="test_pendulum. Forced roll, then +pitch, then +yaw",
        crane_roll=lambda t: 0.01 * np.sin(2.132 * t),
        crane_pitch=lambda t: 0 if t < 4 * np.pi / 2.132 else 0.01 * np.sin(3.132 * t),
        crane_yaw=lambda t: 0 if t < 8 * np.pi / 2.132 else 0.01 * np.sin(4.132 * t),
    )


# @pytest.mark.skip()
def test_sequence(crane, show):
    f, p, b1, r = [b for b in crane.booms()]
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Initial state. Crane folded")
    assert np.allclose(r.origin + r.length * r.direction, (10, 0, 2.5))
    assert np.allclose(r.end, (10, 0, 2.5))
    # boom1 up (vertical)
    b1.boom_setter((None, 0, None))
    assert np.allclose(r.end, (0 + 0.5 / sqrt(2), 0, 3 + 10 - 0.5 / sqrt(2)), atol=0.1)  # somewhat lower due to length
    pendulum_relax(r, show=False)
    assert np.allclose(r.end, (0, 0, 3 + 10 - 0.5), atol=0.05), f"Found equilibrium position {r.end}"
    # boom1 45 deg up
    b1.boom_setter((None, radians(45), None))
    r.pendulum_relax()
    assert np.allclose(r.end, [10 / sqrt(2), 0, 3 - 0.5 + 10 / sqrt(2)])
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). boom 1 at 45 degrees.")
    # wire 0.5m -> 5m
    r.boom_setter((5.0, None, None))
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Wire 0.5m -> 5m")
    r.pendulum_relax()
    assert np.allclose(r.end, [10 / sqrt(2), 0, 3 + 10 / (sqrt(2)) - 5]), f"Found load position {r.end}"
    # turn base 45 deg
    p.boom_setter((None, None, radians(45)))
    r.pendulum_relax()
    assert np.allclose(r.end, (10 / sqrt(2) / sqrt(2), 10 / sqrt(2) / sqrt(2), 3 - 5 + 10 / sqrt(2)))
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Turn base 45 degrees")

    len_0 = r.length
    # boom1 up. Dynamic
    for i in range(450):
        angle = 45 - i / 100
        b1.boom_setter((None, angle, None))
        crane.calc_statics_dynamics(1.0)
        # print(f"angle {angle}, rope length: {r.length}, rope origin: {r.origin}. rope velocity: {r.velocity}")
    assert len_0 == r.length, "Length of rope has changed!"


# @pytest.mark.skip()
def test_change_length(crane, show):
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
def test_rotation(crane, show):
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
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_rotation(). Before rotation. b1 east.")
    com = (50 * np.array((10, 0, 2.5)) + 200 * np.array((5, 0, 3)) + 2000 * np.array((-1, 0.8, 1.5))) / 2250
    assert np.allclose(f.c_m_sub[1], com), f"{f.c_m_sub[1]} != {com}"
    torque = 2250 * np.cross(com, (0, 0, -9.81))
    assert np.allclose(f.torque, torque), f"{f.torque} != {torque}"

    p.boom_setter((None, None, radians(-90)))  # turn p so that b1 south
    assert np.allclose(b1.direction, (0, -1, 0))
    assert np.allclose(b1.c_m, (0, -5, 0))
    crane.calc_statics_dynamics()
    com = (50 * np.array((0, -10, 2.5)) + 200 * np.array((0, -5, 3)) + 2000 * np.array((-1, 0.8, 1.5))) / 2250
    assert np.allclose(f.c_m_sub[1], com), f"{f.c_m_sub[1]} != {com}"
    torque = 2250 * np.cross(com, (0, 0, -9.81))
    assert np.allclose(f.torque, torque), f"{f.torque} != {torque}"
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_rotation(). Pedestal turned => b1 south.")


# @pytest.mark.skip()
def test_c_m(crane, show):
    # Note: Boom.c_m is a local measure, calculated from Boom.origin
    f, p, b1, r = [b for b in crane.booms()]
    r.mass -= 50
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_c_m(). Initial")
    # initial c_m location
    #        print("Initial c_m:", p.c_m, b1.c_m, b2.c_m)
    np.allclose(p.c_m, (-1, 0.8, 1.5))  # 2000 kg
    np.allclose(b1.c_m, (5, 0, 0))  # 200 kg
    # all booms along a line in z-direction
    b1.boom_setter((None, 0, None))
    np.allclose(b1.c_m, (0, 0, 5))
    # update all subsystem center of mass points. Need to do that from last boom!
    crane.calc_statics_dynamics()
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_c_m(). All booms along a line in z-direction")
    np.allclose(b1.c_m_sub[1], (0, 0, 3 + 5 + 100 / 300 * (15.5 - 8)))
    np.allclose(p.c_m_sub[1], (p.mass * p.c_m + b1.c_m_sub[0] * b1.c_m_sub[1]) / (p.mass + b1.c_m_sub[0]))
    assert p.c_m_sub[0] == 2200
    b1.boom_setter((None, radians(45), None))
    p.boom_setter((None, None, radians(-20)))


def animate_sequence(crane, seq=(), nSteps=10):
    """Generate animation frames for a sequence of rotations. To be used as 'update' argument in FuncAnimation.
    A sequence element consists of a boom and an angle, which then is rotated in nSteps.
    To do updates of statics and dynamics we need to know the last boom.
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
            lw = {"pedestal": 10, "rope": 2}.get(b.name, 5)
            lines.append(
                ax.plot(
                    [b.origin[0], b.end[0]],
                    [b.origin[1], b.end[1]],
                    [b.origin[2], b.end[2]],
                    linewidth=lw,
                )
            )

    def update(p):
        """Based on the updated first boom (i.e. the whole crane), draw any desired data"""
        for i, b in enumerate(crane.booms()):
            lines[i][0].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )

    f, p, b1, b2, r = list(crane.booms())
    fig = plt.figure(figsize=(9, 9), layout="constrained")
    ax = plt.axes(projection="3d")  # , data=line)
    lines = []

    _ = FuncAnimation(
        fig,
        update,  # type: ignore  ## this is a function!
        frames=animate_sequence(crane, seq=((p, -90), (b1, -45), (b2, 180))),
        init_func=init,  # type: ignore  ## this is a function!
        interval=1000,
        blit=False,
        cache_frame_data=False,
    )
    plt.title("Crane animation", loc="left")
    plt.show()
    # np.allclose(r.origin, (0, -15 / sqrt(2), 3 + 15 / sqrt(2)))


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

    def wire_end(x1: Sequence | np.ndarray, y0: Sequence | np.ndarray, length: float = 1.0):
        diff = np.array(y0) - np.array(x1)
        return np.array(x1) + length * (diff) / np.linalg.norm(diff)

    def crane_check(_p: Sequence, _b1: Sequence, _r: Sequence, show_it: int = 0, title: str = ""):
        """Check the end positions of all booms."""
        logger.info(f"test_orientation {title}")
        assert np.allclose(p.end, _p), f"Expected {_p}, found {p.end}"
        assert np.allclose(b1.end, _b1), f"Expected {_b1}, found {b1.end}"
        assert np.allclose(r.end, wire_end(b1.end, _r, 1.0)), f"Expected {wire_end(b1.end, _r, 1.0)}"
        if show & show_it >= show_it:
            show_crane(crane, True, False, title=title)

    f, p, b1, r = [b for b in crane.booms()]
    r.boom_setter((1.0, None, None))  # wire 1m
    crane.rotate((0, 0, 0), degrees=True, absolute=False)  # zero turn
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="test_orientation(initial).")
    angle = 90
    # only roll
    crane.rotate((angle, 0, 0), degrees=True, absolute=True)  # roll 90 deg
    crane_check((0, -3, 0), (10, -3, 0), (10, 0, 2), show_it=2, title=f"test_orientation(roll({angle})).")
    # roll back
    crane.rotate((-angle, 0, 0), degrees=True, absolute=False)
    r.pendulum_relax()
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="Crane rotated back")
    # only pitch
    crane.rotate((0, angle, 0), degrees=True, absolute=True)
    crane_check((-3, 0, 0), (-3, 0, 10), (10, 0, 2), show_it=4, title=f"test_orientation(pitch({angle})).")
    # pitch back
    crane.rotate((0, -angle, 0), degrees=True, absolute=False)
    r.pendulum_relax()
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="Crane rotated back")
    # only yaw
    crane.rotate((0, 0, angle), degrees=True, absolute=True)
    crane_check((0, 0, 3), (0, -10, 3), (10, 0, 2), show_it=8, title=f"test_orientation(yaw({angle})).")
    # yaw back
    crane.rotate((0, 0, -angle), degrees=True, absolute=False)
    r.pendulum_relax()
    crane_check((0, 0, 3), (10, 0, 3), (10, 0, 2), show_it=1, title="Crane rotated back")
    # roll, pitch, yaw successively
    crane.rotate((angle, 0, 0), degrees=True, absolute=True)
    crane.rotate((0, angle, 0), degrees=True, absolute=False)
    crane.rotate((0, 0, angle), degrees=True, absolute=False)
    if show & 16 >= 16:
        show_crane(crane, True, False, title="roll, pitch, yaw successively")


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel(level="warning")
    parsolog = logging.getLogger("parso")
    parsolog.setLevel(logging.WARNING)
    pillog = logging.getLogger("PIL")
    pillog.setLevel(logging.WARNING)
    c = _crane()
    # test_initial(c)
    # test_mass_center()
    # test_orientation(_crane(), 16)
    # test_getter_setter(c)
    # test_pendulum(_crane(), show=True)
    # test_sequence(c, True)
    # test_change_length(c, show=True)
    # test_rotation(c, show=True)
    # test_c_m(c, show=True)
    # test_animation(c, show=True)
