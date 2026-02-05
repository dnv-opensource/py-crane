import logging
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pytest  # noqa: F401
from component_model.utils.controls import Controls
from matplotlib.animation import FuncAnimation

from py_crane.animation import AnimateCrane, AnimatePlayBackLines
from py_crane.boom import Boom
from py_crane.crane import Crane

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=4, suppress=True)


@pytest.fixture
def crane(scope: str = "session", autouse: bool = True):
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
        boom=(10.0, np.radians(90), 0),
    )
    _ = crane.add_boom(
        name="wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_rng=(50, 2000),
        mass_center=1.0,
        boom=(0.5, np.radians(90), 0),
        q_factor=10.0,
    )
    return crane


def animate_sequence(crane: Crane, seq: tuple[tuple[Boom, float], ...] = (), nSteps: int = 10):
    """Generate animation frames for a sequence of rotations. To be used as 'update' argument in FuncAnimation.
    A sequence element consists of a boom and an angle, which then is rotated in nSteps.
    """
    for b, a in seq:
        if b.name == "pedestal":  # azimuthal movement
            db = np.array((0, 0, np.radians(a / nSteps)), float)
        else:  # polar movement
            db = np.array((0, np.radians(a / nSteps), 0), float)
        for _ in range(nSteps):
            b.boom_setter(list(b.boom + db))
            # update all subsystem center of mass points. Need to do that from last boom!
            crane.calc_statics_dynamics(dt=None)
            yield (crane)


# @pytest.mark.skip("Animate crane movement")
def test_animation_sequence(crane: Crane, show: bool = False):
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

    def update(crane: Crane):
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
    lines: list[Any] = []

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


def movement(crane: Crane, dt: float = 0.01, t_end: float = 10.0) -> Generator[tuple[float, Crane], None, None]:
    """Create movement of the crane through definition and usage of Controls.
    Generator function. Returns a `Generator` which steps through time and sequentially yields updated frames in the form of tuple (time, crane) objects.
    time is defined global as a simple way to draw the current time together with the title.
    """
    # initial definition of controls and start values
    controls = Controls(limit_err=logging.WARNING)  # CRITICAL)
    f, p, b1, r = list(crane.booms())
    controls.append("turn", (None, (-0.31, 0.31), (-0.16, 0.16)))  # free rotation, max 1 turn/20sec, 2sec to max
    controls.append("luff", ((0, 1.58), (-0.18, 0.09), (-0.09, 0.05)))  # 90 deg, 5/-2.5 deg/sec, 2sec to max
    controls.append("boom", ((8, 50), (-0.2, 0.1), (-0.1, 0.05)))  # 8m..50m, 0.1/-0.2 m/sec, 2sec to max
    controls.append("wire", ((0.5, 50), (-0.1, 1.0), (-0.05, 0.1)))  # 0.5m..50m, -0.1/1 m/sec, 2sec to max
    controls.current[2][0] = 8.0  # b1 starts with 8m
    controls.current[1][0] = np.radians(90)  # b1 starts at 90 deg
    controls.current[3][0] = 0.5  # wire length starts 0.5m

    # From time 0 we set three goals
    controls.setgoal("turn", 0, np.radians(90), 0.0)  # turn pedestal 90 deg
    controls.setgoal("luff", 0, np.radians(45), 0.0)  # luff boom to 45 deg
    controls.setgoal("boom", 1, 0.1, 0.0)  # increase length 0.1m/s
    for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
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
        yield (time + dt, crane)


def test_animation_control(crane: Crane, show: bool = False):
    if not show:  # if nothing can be shown, we do not need to run it
        return
    ani = AnimateCrane(crane, movement, dt=0.1, t_end=20)  # type: ignore  ## It is a Generator!
    ani.do_animation()


def test_playback_lines(show: bool = False):
    data = (
        np.array((0, 1, 2, 3), float),  # time
        np.zeros(12).reshape((4, 3)),  # base
        np.array(((0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4)), float),  # end of base
        np.array(((1, 0, 1), (2, 1, 2), (3, 2, 3), (4, 3, 4)), float),  # end of next
        np.array(((1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 3, 0)), float),  # 'wire'
    )
    ani = AnimatePlayBackLines(
        data=data, lw=(10, 2, 1), figsize=(10, 10), interval=1000, title="Test of AnimatePlayBackLines"
    )
    assert np.allclose(ani.axes_lim, [[-1, 5], [-1, 5], [-1, 5]])
    if show:
        ani.do_animation()


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.DEBUG)
    parsolog = logging.getLogger("parso")
    parsolog.setLevel(logging.WARNING)
    pillog = logging.getLogger("PIL")
    pillog.setLevel(logging.WARNING)

    # test_animation_sequence(_crane(), show=True)
    test_animation_control(_crane(), show=True)
    # test_playback_lines(show=True)
