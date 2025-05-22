import logging
from math import radians, sqrt
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from crane_fmu.boom import Boom
from crane_fmu.crane import Crane

# from mpl_toolkits.mplot3d.art3d import Line3D

logger = logging.getLogger(__name__)


def np_arrays_equal(arr1, arr2, dtype="float", eps=1e-7):
    assert isinstance(arr1, (tuple, list, np.ndarray)), f"Array 1 {arr1} not an array"
    assert isinstance(arr2, (tuple, list, np.ndarray)), f"Array 2 {arr1} not an array"
    assert len(arr1) == len(arr2), "Length not equal!"
    if isinstance(arr2, (tuple, list)):
        arr2 = np.array(arr2, dtype=dtype)
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray), (
        "At least one of the parameters is not an ndarray!"
    )
    assert arr1.dtype == arr2.dtype, f"Arrays are of type {arr1.dtype} != {arr2.dtype}"

    for i in range(len(arr1)):
        assert abs(arr1[i] - arr2[i]) < eps, f"Component {i}: {arr1[i]} != {arr2[i]}"


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
        np_arrays_equal(Mc[1], _c, eps=1e-10)

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
        np_arrays_equal(n_dir0, directions[i] / np.linalg.norm(directions[i]))


def pendulum_relax(rope: Boom, show: bool, steps: int = 1000, dt: float = 0.01):
    x = []
    for _ in range(steps):  # let the pendulum relax
        rope.calc_statics_dynamics(dt)
        x.append(rope.end[2])
    if show:
        fig, ax = plt.subplots()
        ax.plot(x)
        plt.title("Pendulum relaxation", loc="left")
        plt.show()


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
        name="boom2",
        description="The second boom. Can be lifted whole range",
        mass=100.0,
        mass_center=0.5,
        boom=(5.0, radians(-180), 0),
    )
    _ = crane.add_boom(
        name="rope",
        description="The rope fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_rng=(50, 2000),
        mass_center=1.0,
        boom=(0.5, radians(180), 0),
        damping=2.0,
    )
    return crane


def test_initial(crane):
    """Test the initial state of the crane."""
    # test indexing of booms
    booms = [b.name for b in crane.booms()]
    assert booms == ["fixation", "pedestal", "boom1", "boom2", "rope"]
    fixation, pedestal, boom1, boom2, rope = [b for b in crane.booms()]

    assert crane.boom0.name == "fixation", "Boom0 should be fixation"
    np_arrays_equal(crane.boom0.origin, (0.0, 0.0, -1e-10))  # fixation somewhat below surface@ {crane.boom0.origin}
    np_arrays_equal(crane.boom0.end, (0, 0, 0))  # fixation end at 0: {crane.boom0.end}
    bs = crane.booms()  # iterator generator for the booms based on crane object
    next(bs)
    assert next(bs).name == "pedestal", "Should be 'pedestal'"
    bs = crane.booms(reverse=True)
    assert next(bs).name == "rope", "First reversed should be 'rope'"
    assert next(bs).name == "boom2", "Next reversed should be 'boom2'"

    assert pedestal[0].name == "pedestal", "pedestal[0] should be 'pedestal'"
    assert pedestal[1].name == "boom1", "pedestal[1] should be 'boom1'"
    assert pedestal[-1].name == "rope", "pedestal[-1] should be 'rope'"
    assert pedestal[-2].name == "boom2", "pedestal[-2] should be 'boom2'"

    # for i,b in enumerate(crane.booms()):
    #    print( f"Boom {i}: {b.name}")
    assert list(crane.booms())[2].name == "boom1", "'boom1' from boom iteration expected"
    assert list(crane.booms(reverse=True))[1].name == "boom2", "'boom2' from reversed boom iteration expected"
    assert pedestal in crane.booms(), "pedestal expected as boom"

    assert pedestal.length == 3.0
    assert boom1.length == 10.0
    assert boom2.length == 5.0
    assert pedestal.anchor1.name == "boom1"
    assert boom1.anchor1.name == "boom2"
    assert pedestal.name == "pedestal"
    assert pedestal.mass == 2000.0, f"Found {pedestal.mass}"
    np_arrays_equal(pedestal.origin, (0, 0, 0))
    np_arrays_equal(pedestal.direction, (0, 0, 1))
    np_arrays_equal(pedestal.c_m, (-1, 0.8, 1.5))
    assert pedestal.length == 3
    np_arrays_equal(pedestal.end, boom1.origin)
    np_arrays_equal(boom1.origin, (0, 0, 3.0))
    np_arrays_equal(boom1.direction, (1, 0, 0))
    assert boom1.length == 10
    np_arrays_equal(boom1.end, boom2.origin)
    np_arrays_equal(boom2.origin, (10, 0, 3))
    np_arrays_equal(boom2.direction, (-1, 0, 0))
    assert boom2.length == 5
    np_arrays_equal(boom2.end, (5, 0, 3))
    np_arrays_equal(rope.origin, (5, 0, 3))
    np_arrays_equal(rope.end, (5, 0, 2.5))
    for b in crane.booms():
        np_arrays_equal(b.velocity, (0, 0, 0))

    # Check center of mass calculation
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = pedestal.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    np_arrays_equal(_c, c)

    # simplify crane and perform manual torque calculation
    pedestal.mass_center = (0.5, 0, 0)
    boom1.boom_setter((None, radians(90), None))
    boom2.boom_setter((None, radians(0), None))
    rope.mass = 1e-100
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = fixation.c_m_sub
    assert abs(_M - M) < 1e-9, f"Masses {_M} != {M}"
    np_arrays_equal(_c, c)
    np_arrays_equal(fixation.torque, (0, M * c[0] * 9.81, 0))

    # align booms and perform manual calculation
    pedestal.mass_center = (0.5, 0, 0)
    boom1.boom_setter((None, 0, None))
    boom2.boom_setter((None, 0, None))
    rope.mass = 1e-100
    M, c = mass_center(tuple((b.mass, b.origin + b.c_m) for b in crane.booms(reverse=True)))
    crane.calc_statics_dynamics()
    _M, _c = pedestal.c_m_sub
    assert abs(_M - 2300) < 1e-9, f"Masses {_M} != {M}"
    np_arrays_equal(_c, (0, 0, (2000 * 1.5 + 200 * 8 + 100 * 15.5) / 2300))
    np_arrays_equal(pedestal.torque, (0, 0, 0))


# @pytest.mark.skip()
def test_pendulum(crane, show):
    f, p, b1, b2, r = [b for b in crane.booms()]
    # rope 1m.
    r.boom_setter((1, None, None))
    np_arrays_equal(r.origin, (5, 0, 3))
    np_arrays_equal(r.end, (5, 0, 2))
    z_pos: list[float] = [2.0]
    speed: list[float] = [0.0]
    time: list[float] = [0.0]
    # => pendulum with origin at [5,0,3] with length 1m (down)
    angle: float = 0.0
    crane.current_time = 0.0
    dt: float = 0.01
    while crane.current_time < 1:
        crane.current_time += dt
        angle += 20.0 * dt
        time.append(crane.current_time)
        # crane.pedestal_boom = (None, None, angle)
        p.boom_setter((None, None, radians(angle)))
        crane.calc_statics_dynamics(dt)
        # print(f"Angle {degrees(angle)}, rope origin: {r.origin}. rope velocity: {r.velocity}")
        z_pos.append(r.end[2])
        speed_end = float(np.linalg.norm(r.velocity))
        if speed_end > 1e10:
            break
        speed.append(speed_end)
    assert abs(sum(z for z in z_pos) / len(z_pos) - 2.00532) < 1e-5
    assert abs(sum(v for v in speed) / len(speed) - 0.52193) < 1e-5
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(time, z_pos)
        ax2.plot(time, speed)
        plt.title("test_pendulum(). z_pos -> 2.0 and speed -> 0.52", loc="left")
        plt.show()


# @pytest.mark.skip()
def test_sequence(crane, show):
    f, p, b1, b2, r = [b for b in crane.booms()]
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=True,
            title="test_sequence(). Initial state. Crane folded",
        )
    np_arrays_equal(r.origin + r.length * r.direction, (5, 0, 2.5))
    np_arrays_equal(r.end, (5, 0, 2.5))
    # boom1 up
    b1.boom_setter((None, 0, None))
    np_arrays_equal(r.end, (0 + 0.5 / sqrt(2), 0, 3 + 10 - 5 - 0.5 / sqrt(2)), eps=0.1)  # somewhat lower due to length
    pendulum_relax(r, show=False)
    np_arrays_equal(r.end, (0, 0, 3 + 10 - 5 - 0.5), eps=0.001)  # equilibrium position
    # boom1 45 deg up
    b1.boom_setter((None, radians(45), None))
    pendulum_relax(r, show=False)
    np_arrays_equal(r.end, [(10 - 5) / sqrt(2), 0, 3 - 0.5 + (10 - 5) / (sqrt(2))], eps=0.001)
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=True,
            title="test_sequence(). boom 1 at 45 degrees.",
        )
    # boom2 0 deg (in line with boom1)
    b2.boom_setter((None, radians(0), None))
    pendulum_relax(r, show=False)
    np_arrays_equal(r.end, [(10 + 5) / sqrt(2), 0, 3 - 0.5 + (10 + 5) / (sqrt(2))], eps=0.001)
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=True,
            title="test_sequence(). boom2 in line with boom 1",
        )
    # rope 0.5m -> 5m
    r.boom_setter((5.0, None, None))
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=True,
            title="test_sequence(). Rope 0.5m -> 5m",
        )
    np_arrays_equal(r.end, [(10 + 5) / sqrt(2), 0, 3 - 5 + (10 + 5) / (sqrt(2))], eps=0.001)
    # turn base 45 deg
    p.boom_setter((None, None, radians(45)))
    pendulum_relax(r, show=False)
    np_arrays_equal(
        r.end,
        (
            (10 + 5) / sqrt(2) / sqrt(2),
            (10 + 5) / sqrt(2) / sqrt(2),
            3 - 5 + (10 + 5) / (sqrt(2)),
        ),
        eps=0.001,
    )
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=True,
            title="test_sequence(). Turn base 45 degrees",
        )

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
    f, p, b1, b2, r = [b for b in crane.booms()]
    if show:
        show_crane(crane, title="test_change_length(). Initial")
    b2.boom_setter((None, 0, None))
    assert r.anchor1 is None
    np_arrays_equal(b2.end, r.origin)
    r.boom_setter((3, None, None))  # increase length
    pendulum_relax(r, show=False)
    np_arrays_equal(r.end, (15.0, 0, 0), eps=0.001)
    if show:
        show_crane(crane, title="test_change_length(). rope -> 3m")


# @pytest.mark.skip()
def test_boom_position(crane):
    """Testing boom positions which are also used in MobileCrane.cases"""
    f, p, b1, b2, r = [b for b in crane.booms()]
    np_arrays_equal(p.end, (0, 0, 3))
    np_arrays_equal(b1.end, (10, 0, 3))
    np_arrays_equal(b2.end, (5, 0, 3))

    p.boom_setter((3, 0, radians(90)))  # turn pedestal around z-axis 90 deg
    np_arrays_equal(b1.end, (0, 10, 3.0))


# @pytest.mark.skip()
def test_rotation(crane, show):
    f, p, b1, b2, r = [b for b in crane.booms()]
    b1.boom_setter((None, 0, None))  # b1 up
    np_arrays_equal(b1.direction, (0, 0, 1))
    assert b1.length == 10
    np_arrays_equal(b2.origin, (0, 0, 13))
    np_arrays_equal(b2.end, (0, 0, 8))
    np_arrays_equal(b1.c_m, (0, 0, 5))  # measured relative to its origin!
    np_arrays_equal(b2.c_m, (0, 0, -2.5))  # measured relative to its origin!
    np_arrays_equal(r.origin, (0, 0, 8))
    assert abs(r.length - 0.5) < 1e-10, f"Unexpected length {r.length}"
    b1.boom_setter((None, radians(90), None))  # b1 east (as initially)
    np_arrays_equal(b1.direction, (1, 0, 0))
    assert b1.length == 10
    np_arrays_equal(b1.c_m, (5, 0, 0))
    np_arrays_equal(b2.direction, (-1, 0, 0))
    assert b2.length == 5
    np_arrays_equal(b2.c_m, (-2.5, 0, 0))
    crane.calc_statics_dynamics()
    np_arrays_equal(f.c_m_sub[1], (0, 0.68085, 1.71), eps=0.05)
    np_arrays_equal(f.torque, (-9.81 * 2350 * 0.68085, 0, 0), eps=0.5)
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=False,
            title="test_rotation(). Before rotation. b1 east.",
        )

    p.boom_setter((None, None, radians(-90)))  # turn p so that b1 south
    np_arrays_equal(b1.direction, (0, -1, 0))

    np_arrays_equal(b1.c_m, (0, -5, 0))
    np_arrays_equal(b2.direction, (0, 1, 0))
    assert b2.length == 5
    np_arrays_equal(b2.c_m, (0, 2.5, 0))
    crane.calc_statics_dynamics()
    np_arrays_equal(f.c_m_sub[1], (-2000 / 2350, -400 / 2350, 1.71), eps=0.05)
    np_arrays_equal(f.torque, (-9.81 * -400, 9.81 * -2000, 0), eps=0.5)
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=False,
            title="test_rotation(). Pedestal turned => b1 south.",
        )


# @pytest.mark.skip()
def test_c_m(crane, show):
    # Note: Boom.c_m is a local measure, calculated from Boom.origin
    f, p, b1, b2, r = [b for b in crane.booms()]
    r.mass -= 50
    if show:
        show_crane(crane, markCOM=True, markSubCOM=False, title="test_c_m(). Initial")
    # initial c_m location
    #        print("Initial c_m:", p.c_m, b1.c_m, b2.c_m)
    np_arrays_equal(p.c_m, (-1, 0.8, 1.5))  # 2000 kg
    np_arrays_equal(b1.c_m, (5, 0, 0))  # 200 kg
    np_arrays_equal(b2.c_m, (-2.5, 0, 0))  # 100 kg
    # all booms along a line in z-direction
    b2.boom_setter((None, 0, None))
    b1.boom_setter((None, 0, None))
    np_arrays_equal(b1.c_m, (0, 0, 5))
    np_arrays_equal(b2.c_m, (0, 0, 2.5))
    # update all subsystem center of mass points. Need to do that from last boom!
    crane.calc_statics_dynamics()
    if show:
        show_crane(
            crane,
            markCOM=True,
            markSubCOM=False,
            title="test_c_m(). All booms along a line in z-direction",
        )
    np_arrays_equal(b2.c_m_sub[1], b2.origin + b2.c_m)
    np_arrays_equal(b1.c_m_sub[1], (0, 0, 3 + 5 + 100 / 300 * (15.5 - 8)))
    assert p.c_m_sub[1][2] == 1.5 + 300 / 2300 * (b1.c_m_sub[1][2] - 1.5)
    b1.boom_setter((None, radians(45), None))
    b2.boom_setter((None, radians(30), None))
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
    # np_arrays_equal(r.origin, (0, -15 / sqrt(2), 3 + 15 / sqrt(2)))


def show_crane(_crane, markCOM=True, markSubCOM=True, title: str | None = None):
    # update all subsystem center of mass points. Need to do that from last boom!
    _crane.calc_statics_dynamics(dt=None)
    fig = plt.figure(figsize=(9, 9), layout="constrained")
    ax = fig.add_subplot(projection="3d")  # Note: this loads Axes3D implicitly
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 10)  # type: ignore [attr-defined] ## according to matplotlib recommendations

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


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.INFO)
    c = _crane()
    # test_initial(c)
    # test_getter_setter(c)
    # test_pendulum(c, show=True)
    # test_sequence(c, True)
    # test_change_length(c, show=True)
    # test_boom_position(c)
    # test_rotation(c, show=True)
    # test_c_m(c, show=True)
    # test_animation(c, show=True)
