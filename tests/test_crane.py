import logging
from math import cos, radians, sin, sqrt
from typing import Any, Callable, Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from component_model.analytic import ForcedOscillator1D
from component_model.utils.analysis import extremum_series, sine_fit
from component_model.utils.transform import rot_from_spherical, rot_from_vectors
from scipy.spatial.transform import Rotation as Rot

from py_crane.animation import AnimatePendulum
from py_crane.boom import Boom
from py_crane.crane import Crane

# from mpl_toolkits.mplot3d.art3d import Line3D

logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.set_printoptions(precision=4, suppress=True)


def np_index(times: np.ndarray, t: float):
    """Get the closest index with respect to a value in the array."""
    return np.absolute(times - t).argmin()


def do_show(
    times: np.ndarray | list[float],
    traces: Mapping[str, list[float] | np.ndarray],
    selection: dict[str, int] | None = None,
    title: str = "",
):
    """Plot selected traces."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for label, trace in traces.items():
        if selection is None:  # all in first subplot
            _ = ax1.plot(times, trace, label=label)
        else:
            if label in selection:
                if selection[label] == 1:
                    _ = ax1.plot(times, trace, label=label)
                elif selection[label] == 2:
                    _ = ax2.plot(times, trace, label=label)
    _ = ax1.legend()
    _ = ax2.legend()
    plt.title(title)
    plt.show()


def set_wire_direction(r: Boom, angles: tuple[float, float], degrees: bool = False):
    """Set the angles of a wire object. Makes only sense for preparation of test cases. Not allowed in Boom class."""
    _angles = np.array(angles, float)
    if degrees:
        _angles = np.radians(_angles)
    r.boom[1:] = _angles
    assert r.anchor0 is not None
    r.rot(r.anchor0.rot() * rot_from_spherical(_angles))
    r.direction = r.rot().apply(np.array((0, 0, 1), float))
    r.r_v = np.array((0, 0, 0), float)  # reset also the speed


def mass_center(xs: tuple[tuple[float, np.ndarray], ...]):
    """Calculate the total mass center of a number of point masses provided as 4-tuple"""
    M, c = 0.0, np.array((0, 0, 0), float)
    for x, v in xs:
        M += x
        c += x * v
    return (M, c / M)


def test_mass_center():
    def do_test(Mc: tuple[float, np.ndarray], _M: float, _c: np.ndarray):
        print("Mc", Mc)
        assert cast(float, Mc[0]) == _M, f"Mass not as expected: {Mc[0]} != {_M}"
        assert np.allclose(cast(np.ndarray, Mc[1]), _c)

    def vec(v1: float, v2: float, v3: float) -> np.ndarray:
        return np.array((v1, v2, v3), float)

    v0 = vec(0, 0, 0)
    do_test(mass_center(((1, vec(-1, 0, 0)), (1, vec(1, 0, 0)), (2, vec(0, 0, 0)))), 4, v0)
    do_test(mass_center(((1, vec(1, 1, 0)), (1, vec(1, -1, 0)), (1, vec(-1, -1, 0)), (1, vec(-1, 1, 0)))), 4, v0)
    do_test(mass_center(((1, vec(1, 1, 0)), (1, vec(1, -1, 0)), (1, vec(-1, -1, 0)), (1, vec(-1, 1, 0)))), 4, v0)


def aligned(p_i: Sequence[float]):
    """Check whether all points pi are on the same straight line."""
    assert len(p_i) > 2, (
        f"Checking whether points are on the same line should include at least 3 points. Got only {len(p_i)}"
    )
    directions = [p_i[i] - p_i[0] for i in range(1, len(p_i))]
    n_dir0 = directions[0] / np.linalg.norm(directions[0])
    for i in range(1, len(directions)):
        assert np.allclose(n_dir0, directions[i] / np.linalg.norm(directions[i]))


def pendulum_relax(wire: Boom, show: bool = False, steps: int = 1000, dt: float = 0.01):
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
def crane(scope: str = "session", autouse: bool = True):
    return _crane()


def _crane(wire_length: float = 0.5):
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
    w = crane.add_boom(
        name="wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_center=1.0,
        boom=(wire_length, radians(90), 0),
        q_factor=10.0,
        additional_checks=True,
    )
    assert np.allclose(w.origin, (10, 0, 3.0)), f"Origin {w.origin} != (10,0,3)"
    assert np.allclose(w.end, (10, 0, 3.0 - wire_length)), f"Load position {w.end} != (10,0,{3.0 - wire_length})."

    return crane


def test_initial(crane: Crane):
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

    assert pedestal[0] is not None and pedestal[0].name == "pedestal", "pedestal[0] should be 'pedestal'"  # type: ignore
    assert pedestal[1] is not None and pedestal[1].name == "boom1", "pedestal[1] should be 'boom1'"  # type: ignore
    assert pedestal[-1] is not None and pedestal[-1].name == "wire", "pedestal[-1] should be 'wire'"  # type: ignore
    assert pedestal[-2] is not None and pedestal[-2].name == "boom1", "pedestal[-2] should be 'boom1'"  # type: ignore

    # for i,b in enumerate(crane.booms()):
    #    print( f"Boom {i}: {b.name}")
    assert list(crane.booms())[2].name == "boom1", "'boom1' from boom iteration expected"
    assert list(crane.booms(reverse=True))[1].name == "boom1", "'boom1' from reversed boom iteration expected"
    assert pedestal in crane.booms(), "pedestal expected as boom"

    assert pedestal.length == 3.0
    assert boom1.length == 10.0
    assert pedestal.anchor1 is not None and pedestal.anchor1.name == "boom1"
    assert boom1.anchor1 is not None and boom1.anchor1.name == "wire"
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
    assert np.allclose(wire.origin, (10, 0, 3)), f"Expected wire.origin: (10,0,3). Found:{wire.origin}"
    assert np.allclose(wire.end, (10, 0, 2.5)), f"Expected wire.end: (10,0,2.5). Found:{wire.end}"
    assert np.allclose(crane.velocity, (0, 0, 0))

    # Check center of mass calculation
    M, c = mass_center(tuple([(b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)]))
    crane.calc_statics_dynamics()
    _M, _c = crane.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    assert np.allclose(_c, c), f"Center point {_c} != {c}"

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


def test_animate_pendulum(show: bool = False):
    if not show:  # nothing to do
        return
    anim = AnimatePendulum(
        length=10.0,
        angles=(90, 0),
        v0=np.array((0.2, 1.0, 0), float),  # 1.0,0), float),
        t_end=30.0,
        dt=0.01,
        q_factor=50,
        buffer=100,
    )
    anim.do_animation()


# @pytest.mark.skip()
def test_pendulum(show: bool = False):
    """Test crane with 1m wire and 50kg load at end as pendulum (in various configurations)."""

    def _pendulum(q_factor: float = 20.0):
        """Defines the special crane used here: a pedestal with a wire/load hanging down."""
        crane = Crane()
        _ = crane.add_boom(
            name="pedestal",
            description="The crane base, just a pole for the pendulum, here.",
            mass=2000.0,
            mass_center=0.5,
            boom=(1.0, 0, 0),
        )
        w = crane.add_boom(
            name="wire",
            description="The wire fixed to the last boom. Flexible connection",
            mass=100.0,  # load
            mass_center=1.0,
            boom=(1.0, radians(180), 0),  # 1m down, so that load at (0,0,0) in equilibrium
            q_factor=q_factor,
            additional_checks=True,
        )
        assert np.allclose(w.origin, (0, 0, 1.0)), f"Origin {w.origin} != (0,0,1)"
        assert np.allclose(w.end, (0, 0, 0)), f"Load expected at (0,0,0). Found {w.end}"

        return crane, w

    def maximum(tbl: list[list[float]], col: int = 1):
        """Find maximum in 'col' of table and return row."""
        rmax = [float("-inf")] * len(tbl[0])
        for r in tbl:
            if r[col] > rmax[col]:
                rmax = r
        return rmax

    def check_fit(
        e_y0: float | None,
        e_a: float | None,
        e_w: float | None,
        e_phi: float | None,
        pars: tuple[Any, ...],
        eps: float = 1e-3,
    ):
        """Check the sine fit with respect to expected parameters."""
        if len(pars) == 4:
            _y0, _a, _w, _phi = pars
        elif len(pars) == 5:
            _y0, _a, _w, _phi, tm = pars
        assert e_y0 is None or abs(_y0 - e_y0) < eps, f"Translation {_y0} != {e_y0}"
        assert e_a is None or abs(_a - e_a) < eps, f"Amplitude {_a} != {e_a}"
        assert e_w is None or abs(_w - e_w) < eps, f"Angular frequency {_w} != {e_w}"
        assert e_phi is None or abs(_phi - e_phi) < eps, f"Phase {_phi} != {e_phi}"

    def sim_run(
        crane_spec: tuple[Crane, Boom],
        t_end: float,
        dt: float = 0.1,
        min_z: float | None = None,
        max_z: float | None = None,
        max_speed: float | None = None,
        show_select: dict[str, int] | None = None,
        idx: int = 1,
        title: str = "test_pendulum",
        crane_position: Callable[[float], float] | None = None,
        crane_roll: Callable[[float], float] | None = None,
        crane_pitch: Callable[[float], float] | None = None,
        crane_yaw: Callable[[float], float] | None = None,
        wire_l: Callable[[float], float] | None = None,
        _ext: Callable[[float], float] | None = None,
    ):
        """Perform the simulation as specified. Results analysis is done in calling function."""
        crane, w = crane_spec
        misc: list[float] = []
        x_pos: list[float] = []
        y_pos: list[float] = []
        z_pos: list[float] = []
        speed: list[float] = []
        time = np.arange(0, t_end, dt)
        ext = time if _ext is None else [_ext(t) for t in time]
        _dz0_dt = -1
        z0 = w.end[2]
        length = w.boom[0]
        for _time in time:
            x0 = w.end[0]
            y0 = w.end[1]
            z0 = w.end[2]
            v = float(np.linalg.norm(w.r_v))
            assert min_z is None or w.end[2] >= min_z, f"@{_time}. Min z {w.end[2]} < {min_z}"
            assert max_z is None or w.end[2] <= max_z + 1e-6, f"@{_time}. Max z {w.end[2]} > {max_z}"
            assert max_speed is None or v <= max_speed, f"@{_time}. Max speed {v} < {max_speed}"
            x_pos.append(x0)
            y_pos.append(y0)
            z_pos.append(z0)
            speed.append(v)
            misc.append(w.origin[1])
            if crane_position is not None:
                crane.position = np.array((crane_position(_time), 0, 0), float)
            if crane_roll is not None:
                crane.rotate((crane_roll(_time), 0, 0), absolute=True)
            if crane_pitch is not None:
                crane.rotate((0, crane_pitch(_time), 0), absolute=True)
            if crane_yaw is not None:
                crane.rotate((0, 0, crane_yaw(_time)), absolute=True)
            if wire_l is not None:
                w.boom_setter((wire_l(_time), None, None))
            else:
                assert abs(length - w.boom[0]) < 1e-4, f"Pendulum length {w.boom[0]} != {length}"
            crane.do_step(_time, dt)
        z_max = [[0.0, z0]]
        z_max.extend(extremum_series(time[10:], z_pos[10:], which="max"))

        if show and show_select is not None:
            title = f"{idx}. {title}"
            do_show(
                time,
                {"x_pos": x_pos, "y_pos": y_pos, "z_pos": z_pos, "speed": speed, "misc": misc, "ext": ext},
                selection=show_select,
                title=title,
            )
        return (time, x_pos, y_pos, z_pos, speed, z_max, misc)

    def _initial_angle_speed(
        a0: float = 0.0, v0: float = 0.0, damping_time: float = 1e100, show: bool = False, idx: int = 1
    ):
        crane, w = _pendulum()
        set_wire_direction(w, (a0, 0))
        assert np.allclose(w.end, (-np.sin(a0), 0, 1 - np.cos(a0))), (
            f"Wire angle {np.degrees(a0)}, load: ({-np.sin(a0)},0,{1 - np.cos(a0)}). Found {w.end}"
        )
        tau = w.damping(damping_time=damping_time)

        # Start the crane at maximum potential energy of load (1 deg) without damping
        # we should have theta(t) = theta0* cos(w*t) with theta0= 0.0175 (1deg) and w= sqrt(g/L) = 3.132 => T = 2.006
        w.r_v = np.array((v0, 0, 0), float)
        wd = np.sqrt(9.81 - 1 / 4 / tau**2)
        e_a = np.sqrt(a0**2 + (a0 / 2 / tau + v0) ** 2 / wd**2)
        phi = np.arctan2(1 / 2 / tau, wd)
        time, x_pos, y_pos, z_pos, speed, z_max, misc = sim_run(
            crane_spec=(crane, w),
            t_end=10,
            dt=0.01,
            min_z=0,
            max_z=1 - np.cos(e_a),
            max_speed=np.sqrt(2 * 9.81 / (1 - np.cos(e_a))),
            show_select=None if not show else {"x_pos": 1, "z_pos": 1, "speed": 2},
            title=f"test_pendulum. {np.degrees(a0)}deg through origin",
            idx=idx,
        )
        _y0, _a, _w, _phi, t = sine_fit(time, x_pos)
        check_fit(0.0, e_a * np.exp(-t / 2 / tau), wd, phi, (_y0, _a, _w, _phi), eps=1e-4)

        eps = 1e-5 if tau > 100 else 1e-3
        for t, z in zip(time, z_pos, strict=False):
            theta = a0 * np.cos(np.sqrt(9.81) * t)
            assert abs(z - 1 + np.cos(theta)) < eps, f"@{t}: z={z}. Expected {1.0 - np.cos(theta)}"
        e0 = 9.81 * z_pos[0] + 0.5 * speed[0] ** 2
        for t, z, v in zip(time, z_pos, speed, strict=True):
            assert abs(9.81 * z + 0.5 * v**2 - e0 * np.exp(-t / tau)) <= 1e-3, (
                f"Energy @{t} {9.81 * z + 0.5 * v**2} =! {e0 * np.exp(-t / tau)}"
            )

    def _move_crane(
        v0: float = 0.0,
        te: float = 100,
        tau: float = 5,
        c_pos: Callable[[float], float] | None = None,
        show: bool = False,
        idx: int = 1,
    ):
        crane, w = _pendulum()
        gamma = 1.0 / 2.0 / w.damping(damping_time=tau)  # damping
        wd = np.sqrt(9.81 / w.length - gamma**2)
        T = 2 * np.pi / wd
        if 4 <= idx <= 6:
            osc = ForcedOscillator1D(k=9.81, c=gamma, a=0.001, wf=wd)
        else:
            osc = None
        w.pendulum_relax()
        w.r_v = np.array((v0, 0, 0), float)
        time, x_pos, y_pos, z_pos, speed, z_max, misc = sim_run(
            crane_spec=(crane, w),
            t_end=te,
            dt=T / 100 if idx != 3 else 0.01,
            min_z=1.0 - float(w.length),
            show_select={
                "x_pos": 1,
                "z_pos": 2,
                "speed": 1,
                "ext": 1,
            },
            title=f"test_pendulum_move[{idx}]. Movements in x-direction",
            crane_position=c_pos,  # movement function
            _ext=lambda t: (
                abs((v0 if v0 > 0 else 1.0) * np.exp(-gamma * t) * np.cos(wd * t))
                if osc is None
                else float(osc.calc(t)[0])
            ),
        )
        if idx == 0:  # v0=1.0, tau=20 - damped oscillation. Small amplitude, so that linear approximation holds!
            for t, v in zip(time, speed, strict=True):
                expected = abs(v0 * np.exp(-t / tau) * np.cos(wd * t))
                assert abs(v - expected) < 3e-2, f"Damped oscillation speed @{t}: {v}!={expected}"
        if idx == 1:  # origin moves with v=0.01, while load starts with same speed
            assert all(abs(v - v0) < 1e-3 for v in speed), f"Tail speeds: {speed[0:10]}"
            assert abs(x_pos[-1] - v0 * 100) < 1e-3, f"End position: {x_pos[-1]}!={v0 * 100}"
        elif idx == 2:  # v0=0. For t<5*T crane moves with speed 1/s and stops then
            i_change = np_index(time, 5)
            i_zero = np.absolute(x_pos[i_change : i_change + 200]).argmin()
            assert all(abs(v) < 2e-3 for v in speed[i_change + i_zero :: 100]), (
                f"Speed zero at period. speed[{i_change + i_zero}]:{speed[i_change + i_zero]}"
            )
            assert all(abs(x - 0.25) < 1e-2 for x in x_pos[-200:]), f"End position {0.25}!={x_pos[-10:]}"
        elif idx == 3:
            # Qualitative (visual) comparison with forced oscillator is very good, but oscillator is not harmonic
            # and sweep in low frequency area too fast, so that numerical comparison is difficult.
            i_low = np_index(time, 40)
            i_res = np_index(time, 167.5)
            i_high = np_index(time, 300)
            check_fit(0.0, 0.0, 0.66431, 1.9054, sine_fit(time[:i_low], x_pos[:i_low]), eps=1e-3)
            check_fit(None, 0.0013, 3.1979, 1.8262, sine_fit(time[:i_res], x_pos[:i_res]), eps=1e-3)
            check_fit(None, 3.7e-5, 5.9758, -0.7238, sine_fit(time[:i_high], x_pos[:i_high]), eps=1e-3)
        elif idx == 4:  # resonant oscillation
            _y0, _a, _w, _phi, _ = sine_fit(time, x_pos)
            logger.info(f"Amplitude:{_a}, angular_freq:{_w} - {wd}, phase:{_phi}")

    def _turn_crane(
        te: float = 100,
        tau: float = 20,
        _rpy: str = "r",
        rpy: Callable[[float], float] | None = None,
        show: bool = False,
        idx: int = 1,
    ):
        crane, w = _pendulum()
        gamma = 1.0 / 2.0 / w.damping(damping_time=tau)  # damping
        wd = np.sqrt(9.81 / w.length - gamma**2)
        T = 2 * np.pi / wd
        w.pendulum_relax()
        if _rpy == "r":
            crane_roll, crane_pitch, crane_yaw = rpy, None, None
        elif _rpy == "p":
            crane_roll, crane_pitch, crane_yaw = None, rpy, None
        elif _rpy == "y":
            crane_roll, crane_pitch, crane_yaw = None, None, rpy
        time, x_pos, y_pos, z_pos, speed, z_max, misc = sim_run(
            crane_spec=(crane, w),
            t_end=te,
            dt=T / 100,
            min_z=float(-w.length),
            show_select={
                "x_pos": 1,
                "y_pos": 2,
                "speed": 1,
                "misc": 2,
            },
            crane_roll=crane_roll,
            crane_pitch=crane_pitch,
            crane_yaw=crane_yaw,
            _ext=rpy,
            title=f"test_pendulum_turn[{idx}, {_rpy}].",
        )
        if idx == 1:  # yaw. Torsion is not included. Nothing happens
            assert all(abs(v) < 1e-8 for v in speed), "Speed should be zero"
            assert all(abs(x) < 1e-8 for x in x_pos), "x-position should be zero"
        if idx == 2:  # accelerated roll, then stop (shock)
            i_change = np_index(time, 5 + 2 * T)
            assert all([abs(x) < 1e-10 for x in x_pos]), "x should not be affected."
            assert all([abs(y + np.sin(0.25)) < 0.03 for y in y_pos[-100:]]), (
                f"y at rest with rolled crane? {y_pos[-10:]}"
            )
            _y0, _a, _w, _phi, _ = sine_fit(time[:i_change], np.array(y_pos[:i_change], float) + np.sin(0.25))
            assert abs(_w - wd) < 1e-1, f"Expected angular frequency {wd} != {_w}"
        if idx == 3:  # accelerated pitch, then stop (shock)
            i_change = np_index(time, 5 + 2 * T)
            assert all([abs(y) < 1e-10 for y in y_pos]), "y should not be affected."
            assert all([abs(x + np.sin(0.25)) < 0.03 for x in x_pos[-100:]]), (
                f"x at rest with rolled crane? {x_pos[-10:]}"
            )
            _y0, _a, _w, _phi, _ = sine_fit(time[:i_change], np.array(x_pos[:i_change], float) + np.sin(0.25))
            assert abs(_w - wd) < 2e-2, f"Expected angular frequency {wd} != {_w}"

    def _circular(
        tau: float = 1000,
        v0: float = 10.0,
        wire_l: Callable[[float], float] | None = None,
        angle: float = 0,
        show: bool = False,
        idx: int = 1,
    ):
        """Start with circular motion of load and shorten wire according to wire_l function."""
        crane, w = _pendulum()
        w.damping(damping_time=tau)
        set_wire_direction(w, (90 + angle, 0), degrees=True)
        w.r_v = np.array((0.0, v0, 0.0), float)
        time, x_pos, y_pos, z_pos, speed, z_max, misc = sim_run(
            crane_spec=(crane, w),
            t_end=10.0,
            dt=0.001,
            min_z=0,
            show_select={"x_pos": 1, "y_pos": 1, "z_pos": 2},
            title="test_pendulum. Rotating load and shortened wire length",
            wire_l=wire_l,
        )
        if idx == 0:  # No change in wire length and angle such that conic curve emerges (stable)
            assert abs(angle - 54.75550939085597) < 1e-10, f"Stability angle {angle} != 54.756"
            c_angle = np.cos(np.radians(angle))
            _x = c_angle
            _z = np.sqrt(1 - _x**2)
            assert all(abs(z - 1 + _z) < 1e-5 for z in z_pos), f"Height != {1 - _z}: {z_pos[:10]}"
            i25 = np_index(time, 2.5)
            check_fit(0.0, _x, 2 / c_angle, np.pi / 2, sine_fit(time[:i25], x_pos[:i25]), eps=1e-6)
            _y0, _a, _w, _phi, _tm = sine_fit(time[: np_index(time, 2.5)], y_pos[: np_index(time, 2.5)])
            assert abs(_phi) < 1e-6, f"y-Phase {_phi} != {np.pi / 2}"
        elif idx == 1:  # No change of wire length. Note: curves are not completely circular => accuracy reduced.
            c_angle = np.sqrt(np.sqrt(_b2 + _b2**2 / 4) - _b2 / 2)  # stability angle
            _y0, _a, _w, _phi, _tm = sine_fit(time[: np_index(time, 1.5)], z_pos[: np_index(time, 1.5)])
            assert abs(1.0 - 2**2 / 9.81 - _y0) < 5e-2, "Zentripetal in balance with gravitation. y0={_y0}."
            assert 1.0 - _y0 - _a < 2e-2, f"Energy and angular momentum conservation: y0={_y0}, a={_a}"
        elif idx == 2:  # wire shortened with 0.05*t => l0=1...lend=0.5. Should have wd1 = l0**2 / l1**2* wd0
            i02 = np_index(time, 0.2)
            check_fit(0.0, 1.0, 50, None, sine_fit(time[:i02], x_pos[:i02], eps=0.1), eps=0.16)
            _y0, _a, _w, _phi, _tm = sine_fit(time, x_pos, eps=0.1)
            check_fit(0.0, 0.5, None, None, (_y0, _a, _w, _phi), eps=0.1)
            assert abs(_w - 200) / 200 < 3e-2, f"Angular velocity {_w} != 200"

    _b2 = (2**2 / 9.81) ** 2
    stable = np.degrees(np.arccos(np.sqrt(np.sqrt(_b2 + _b2**2 / 4) - _b2 / 2)))
    _circular(tau=1e10, v0=2.0, wire_l=None, angle=stable, show=show, idx=0)
    _circular(tau=1e10, v0=2.0, wire_l=None, show=show, idx=1)
    _circular(tau=1e10, v0=50.0, wire_l=lambda t: 1.0 - 0.05 * t, show=show, idx=2)
    _move_crane(te=50, v0=0.1, tau=20, show=True, idx=0)
    _move_crane(v0=0.01, c_pos=lambda t: 0.01 * t, tau=50, show=True, idx=1)
    _move_crane(v0=0.0, c_pos=lambda t: 0.01 * t**2 if t <= 5 else 0.25, tau=20.0, show=True, idx=2)
    _move_crane(te=500, c_pos=lambda t: 0.0001 * np.sin((0.01 + 0.01 * t) * t), show=True, idx=3)
    # ??_move_crane(c_pos=lambda t: 0.001 * np.sin( wd* t), tau=tau, te=200, show=True, idx=4)
    # ??_move_crane(c_pos=lambda t: 0.1 * np.sin( 0.1*wd* t), tau=tau, te=100, show=True, idx=5)
    # ??_move_crane(c_pos=lambda t: 0.1 * np.sin( 10*wd* t), tau=tau, te=100, show=True, idx=6)
    _turn_crane(_rpy="y", rpy=lambda t: 0.01 * t**2 if t <= 5 else 0.25, tau=20, show=True, idx=1)
    _turn_crane(_rpy="r", rpy=lambda t: 0.01 * t**2 if t <= 5 else 0.25, tau=20, show=True, idx=2)
    _turn_crane(_rpy="p", rpy=lambda t: 0.01 * t**2 if t <= 5 else 0.25, tau=20, show=True, idx=3)


# @pytest.mark.skip()
def test_sequence(crane: Crane, show: bool = False):
    """Test sequence of crane movements and check that position is as intended."""

    def check_rot(r: Rot, vec: Sequence[float]):
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
    crane.calc_statics_dynamics()
    assert np.allclose(r.end, (0 + 0.5 / sqrt(2), 0, 3 + 10 - 0.5 / sqrt(2)), atol=0.1)  # somewhat lower due to length
    pendulum_relax(r, show=False)
    assert np.allclose(r.end, (0, 0, 3 + 10 - 0.5), atol=0.05), f"Found equilibrium position {r.end}"

    logger.info("boom1 45 deg up")
    b1.boom_setter((None, radians(45), None))
    r.pendulum_relax()  # includes calc_statics_dynamics()
    assert np.allclose(r.end, [10 / sqrt(2), 0, 3 - 0.5 + 10 / sqrt(2)])
    check_rot(b1._rot, (1.0 / sqrt(2), 0.0, 1.0 / sqrt(2)))
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). boom 1 at 45 degrees.")

    logger.info("wire 0.5m -> 5m")
    r.boom_setter((5.0, None, None))
    if show:
        show_crane(crane, markCOM=True, markSubCOM=True, title="test_sequence(). Wire 0.5m -> 5m")
    r.pendulum_relax()  # includes calc_statics_dynamics()
    assert np.allclose(r.end, [10 / sqrt(2), 0, 3 + 10 / (sqrt(2)) - 5]), f"Found load position {r.end}"

    logger.info("turn base 360 deg in steps of 5 deg")
    for i in range(73):  # turn in steps of 5 deg
        p.boom_setter((None, None, radians(i * 5)))
        assert np.allclose(
            b1.end, (10 / sqrt(2) * cos(radians(i * 5)), 10 / sqrt(2) * sin(radians(i * 5)), 3 + 10 / sqrt(2))
        )
        check_rot(b1._rot, (1.0 / sqrt(2) * cos(radians(i * 5)), 1.0 / sqrt(2) * sin(radians(i * 5)), 1.0 / sqrt(2)))

    check_rot(b1._rot, (1.0 / sqrt(2), 0.0, 1.0 / sqrt(2)))  # back to starting point
    r.pendulum_relax()  # includes calc_statics_dynamics()
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
def test_change_length(crane: Crane, show: bool = False):
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
def test_rotation(crane: Crane, show: bool = False):
    f, p, b1, r = [b for b in crane.booms()]
    b1.boom_setter((None, 0, None))  # b1 up
    crane.calc_statics_dynamics()  # ensure that wire is adjusted
    assert np.allclose(b1.direction, (0, 0, 1))
    assert b1.length == 10
    assert np.allclose(b1.c_m, (0, 0, 5))  # measured relative to its origin!
    assert np.allclose(r.origin, (0, 0, 3 + 10)), f"r.origin {r.origin}, != (0,0,13)"
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
def test_c_m(crane: Crane, show: bool = False):
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


def show_crane(_crane: Crane, markCOM: bool = True, markSubCOM: bool = True, title: str | None = None):
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

    def wire_end(
        x1: Sequence[float] | np.ndarray, y0: Sequence[float] | np.ndarray, length: float = 1.0, relaxed: bool = False
    ):
        if relaxed:
            return y0
        diff = np.array(y0) - np.array(x1)
        return np.array(x1) + length * (diff) / np.linalg.norm(diff)

    def crane_check(
        _p: tuple[float, ...],
        _b1: tuple[float, ...],
        _r: tuple[float, ...],
        show_it: int = 0,
        title: str = "",
        relaxed: bool = False,
    ):
        """Check the end positions of all booms."""
        logger.info(f"test_orientation {title}")
        if show & show_it >= show_it:
            show_crane(crane, True, False, title=title)
        assert np.allclose(p.end, _p), f"Expected (p) {_p}, found {p.end}"
        assert np.allclose(b1.end, _b1), f"Expected (b1) {_b1}, found {b1.end}"
        assert np.allclose(r.end, wire_end(b1.end, _r, 1.0, relaxed), atol=1e-4), (
            f"Expected (wire) {wire_end(b1.end, _r, 1.0, relaxed)}. Found {r.end}"
        )

    def rotate_calc_check(
        angles: tuple[float, ...],
        absolute: bool,
        _p: tuple[float, ...],
        _b1: tuple[float, ...],
        _r: tuple[float, ...],
        show_it: int,
        title: str,
        relaxed: bool,
    ):
        crane.rotate(angles, absolute=absolute)  # roll 90 deg
        crane.calc_statics_dynamics()
        if relaxed:
            r.pendulum_relax()
        crane_check(_p, _b1, _r, show_it=show_it, title=title, relaxed=relaxed)

    f, p, b1, r = [b for b in crane.booms()]
    r.boom_setter((1.0, None, None))  # wire 1m
    _a = radians(90)  # the angle used in the rotations

    rotate_calc_check((0, 0, 0), False, (0, 0, 3), (10, 0, 3), (10, 0, 2), 1, "test_orientation(initial).", False)
    # only roll and roll back
    rotate_calc_check((_a, 0, 0), False, (0, -3, 0), (10, -3, 0), (10, 0, 2), 2, "test_orientation(roll({_a}).", False)
    rotate_calc_check((-_a, 0, 0), False, (0, 0, 3), (10, 0, 3), (10, 0, 2), 2, "Crane rolled back.", True)
    # only pitch and pitch back
    rotate_calc_check(
        (0, _a, 0), True, (-3, 0, 0), (-3, 0, 10), (10, 0, 2), 4, f"test_orientation(pitch({_a})).", False
    )
    rotate_calc_check((0, -_a, 0), False, (0, 0, 3), (10, 0, 3), (10, 0, 2), 4, "Crane pitched back", True)
    # only yaw and yaw back
    rotate_calc_check((0, 0, _a), True, (0, 0, 3), (0, -10, 3), (10, 0, 2), 8, f"test_orientation(yaw({_a})).", False)
    rotate_calc_check((0, 0, -_a), False, (0, 0, 3), (10, 0, 3), (10, 0, 2), 8, "Crane yawed back", True)
    # roll, pitch, yaw successively
    crane.rotate((_a, 0, 0), absolute=True)
    crane.rotate((0, _a, 0), absolute=False)
    crane.rotate((0, 0, _a), absolute=False)
    crane.calc_statics_dynamics()
    crane_check((-3, 0, 0), (-3, 0, 10), (10, 0, 2), show_it=1, title="Crane stepped 90deg roll", relaxed=False)
    if show & 16 >= 16:
        show_crane(crane, True, False, title="roll, pitch, yaw successively")
    logger.info("Stepping roll 90x1deg...")
    crane.d_angular = np.array((np.radians(1), 0, 0), float)
    for i in range(90):
        crane.do_step(i, 1.0)
    r.pendulum_relax()
    crane.calc_statics_dynamics()
    if show & 32 >= 32:
        show_crane(crane, True, False, title="roll, pitch, yaw successively")
    ## re-activate later
    # crane_check((-3, 0, 0), (-3, -10, 0), (-3, -10, -1), show_it=1, title="Crane stepped 90deg roll", relaxed=True)


def test_force_torque(crane: Crane, show: bool = False):
    """Check that results for force and torque are correct with respect to crane states and movements."""
    f, p, b1, r = [b for b in crane.booms()]
    p.mass = 100
    p.mass_center = (0.5, 0, 0)
    p.boom_setter((10, None, None))
    b1.mass = 100
    r.mass = 0.0
    traces: dict[str, list[float]]
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
    _xe: list[float] = []
    _load: list[float] = []
    _force: list[float] = []
    _torque: list[float] = []
    while time < 20.0:
        crane.d_velocity = np.array((0.1 * np.sin(1.0 * time), 0, 0), float)
        _time.append(time)
        crane.do_step(time, dt)
        _xe.append(r.origin[0])
        _load.append(r.end[0])
        _force.append(crane.force[0])
        _torque.append(crane.torque[1])
        time += dt
    if show:
        do_show(
            _time,
            traces={"xe": _xe, "load": _load, "force": _force, "torque": _torque},
            selection={"xe": 1, "load": 1, "force": 2, "torque": 2},
        )
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
        traces["xe"].append(float(r.origin[0]))
        traces["load"].append(float(r.end[0]))
        traces["force"].append(float(crane.force[0]))
        traces["torque"].append(float(crane.torque[1]))
        time += dt
    if show:
        do_show(_time, traces, {"xe": 1, "load": 1, "force": 2, "torque": 2})


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
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
    # test_pendulum( show=True)
    # test_animate_pendulum(show=True)
    # test_sequence(_crane(), show=True)
    # test_change_length(_crane(), show=True)
    # test_rotation(_crane(), show=True)
    # test_c_m(_crane(), show=True)
    # test_animation(_crane(), show=True)
    # test_animation_control(_crane(), show=True)
    # test_force_torque(_crane(),show=True)
