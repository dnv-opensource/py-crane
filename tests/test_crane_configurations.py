import logging
from functools import partial
from math import sqrt
from typing import Generator

import numpy as np
import pytest  # noqa: F401
from component_model.utils.controls import Control, Controls

from py_crane.animation import AnimateCrane
from py_crane.boom import Boom, Wire
from py_crane.crane import Crane

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
np.set_printoptions(precision=4, suppress=True)


def _pendulum():
    crane = Crane()
    w = crane.add_boom(
        "wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_center=1.0,
        boom=(1.0, np.pi, 0),
        q_factor=10.0,
    )
    crane.position = np.array((0, 0, 1), float)
    assert isinstance(w, Wire)
    w.pendulum_relax()
    return crane


def _mobile_crane():
    crane = Crane()
    _p = crane.add_boom(
        "pedestal",
        description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
        mass=2000.0,
        mass_center=(0.5, -1, 0.8),
        boom=(3.0, 0, 0),
    )
    _b = crane.add_boom(
        "boom1",
        description="The first boom. Can be lifted",
        mass=200.0,
        mass_center=0.5,
        boom=(10.0, np.radians(90), 0),
    )
    _w = crane.add_boom(
        "wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_center=1.0,
        boom=(1.0, np.radians(90), 0),
        q_factor=10.0,
    )
    return crane


def _boom(b: Boom, idx: int, x: float | None = None) -> float:
    """Move boom 'b', or return current setting. Used as partial() to define Control."""
    if x is not None:
        arg = [None if i != idx else x for i in range(3)]
        b.boom_setter(arg)
    return b.boom[idx]


def move_mobile_crane(
    crane: Crane, dt: float = 0.01, t_end: float = 10.0
) -> Generator[tuple[float, Crane], None, None]:
    """Create movement of the crane through definition and usage of Controls.
    Generator function. Returns a `Generator` which steps through time and sequentially yields updated frames in the form of tuple (time, crane) objects.
    time is defined global as a simple way to draw the current time together with the title.
    """
    # initial definition of controls and start values
    controls = Controls(limit_err=logging.WARNING)  # CRITICAL)
    f, p, b1, w = list(crane.booms())
    controls.extend(
        (
            Control("turn", (None, (-0.31, 0.31), (-0.16, 0.16)), rw=partial(_boom, p, 2)),
            Control("luff", ((0, 1.58), (-0.18, 0.09), (-0.09, 0.05)), rw=partial(_boom, b1, 1)),
            Control("boom", ((8, 50), (-0.2, 0.1), (-0.1, 0.05)), rw=partial(_boom, b1, 0)),
            Control("wire", ((0.5, 50), (-0.1, 1.0), (-0.05, 0.1)), rw=partial(_boom, w, 0)),
        )
    )

    # From time 0 we set three goals
    controls["turn"].setgoal(0, np.radians(90))  # turn pedestal 90 deg
    controls["luff"].setgoal(0, np.radians(45))  # luff boom to 45 deg
    controls["boom"].setgoal(1, 0.1)  # increase length 0.1m/s
    for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
        if time > 10 and not len(controls[3].goal):  # Start to increase wire length with 1m/s
            controls["wire"].setgoal(1, 1.0)
        controls.step(time, dt)
        crane.do_step(time, dt)
        yield (time + dt, crane)


def test_mobile_crane(show: bool = False):
    crane = _mobile_crane()
    if not show:  # if nothing can be shown, we do not need to run it
        return
    ani = AnimateCrane(crane, move_mobile_crane, dt=0.1, t_end=15.0)  # type: ignore  ## It is a Generator!
    ani.do_animation()


def _knuckle_boom_crane():
    crane = Crane()
    _p = crane.add_boom(
        "pedestal",
        description="The crane base, on one side fixed to the vessel and on the other side the first crane boom is fixed to it. The mass should include all additional items fixed to it, like the operator's cab",
        mass=2000.0,
        mass_center=(0.5, -1, 0.8),
        boom=(1.0, 0, 0),
    )
    _b1 = crane.add_boom(
        "boom1",
        description="The first boom. Can be lifted",
        mass=200.0,
        mass_center=0.5,
        boom=(3.0, np.radians(45), 0),
    )
    _b2 = crane.add_boom(
        "boom2",
        description="The second boom. Can be lifted",
        mass=200.0,
        mass_center=0.5,
        boom=(5.0, np.radians(170), np.pi),
    )
    _b3 = crane.add_boom(
        "boom3",
        description="The third boom. Can be lifted",
        mass=200.0,
        mass_center=0.5,
        boom=(7.0, np.radians(170), 0.0),
    )
    _w = crane.add_boom(
        "wire",
        description="The wire fixed to the last boom. Flexible connection",
        mass=50.0,  # so far basically the hook
        mass_center=1.0,
        boom=(1e-6, 0, 0),
        q_factor=100.0,
    )
    assert isinstance(_w, Wire)
    _w.pendulum_relax()
    return crane


def move_knuckle_boom_crane(
    crane: Crane, dt: float = 0.01, t_end: float = 10.0
) -> Generator[tuple[float, Crane], None, None]:
    """Create movement of the crane through definition and usage of Controls.
    Generator function. Returns a `Generator` which steps through time and sequentially yields updated frames in the form of tuple (time, crane) objects.
    time is defined global as a simple way to draw the current time together with the title.
    """
    # initial definition of controls and start values
    controls = Controls(limit_err=logging.WARNING)  # CRITICAL)
    f, p, b1, b2, b3, w = list(crane.booms())
    assert isinstance(w, Wire)
    controls.extend(
        (
            Control("luff3", (None, (-0.09, 0.09), (-0.09, 0.05)), rw=partial(_boom, b3, 1)),
            Control("luff2", (None, (-0.18, 0.09), (-0.09, 0.05)), rw=partial(_boom, b2, 1)),
            Control("luff1", (None, (-0.09, 0.04), (-0.09, 0.05)), rw=partial(_boom, b1, 1)),
            Control("wire", ((0, 50), (-0.1, 1.0), (-0.5, 0.5)), rw=partial(_boom, w, 0)),
            Control("turn", (None, (-0.9, 0.9), (-0.9, 0.9)), rw=partial(_boom, p, 2)),
        )
    )

    # Set initial goals
    controls["luff3"].setgoal(0, np.radians(0))
    controls["luff2"].setgoal(0, np.radians(45))  # luff boom to 45 deg
    controls["luff1"].setgoal(0, 0.0)  # luff to align with pedestal
    for time in np.linspace(0.0, t_end, int(t_end / dt) + 1):
        if abs(time - 10.0) < 1e-6:  # goal2 reached
            assert np.allclose(p.end, (0, 0, 1)), "Pedestal unchanged (straight up)"
            assert np.allclose(b1.end, (0, 0, 4), atol=2e-3), f"Boom 1 aligned with pedestal? {b1.end}"
        if abs(time - 35.0) < 1e-6:  # start to lower hook
            assert np.allclose(p.end, (0, 0, 1)), "Pedestal unchanged (straight up)?"
            _b2 = (-5 / sqrt(2), 0, 4 + 5 / sqrt(2))
            _b3 = (-12 / sqrt(2), 0, 4 + 12 / sqrt(2))
            assert np.allclose(b2.end, _b2, atol=0.01), f"45 deg in negative x-direction? {b2.end}!={_b2}"
            assert np.allclose(b3.end, _b3, atol=0.01), f"45 deg in negative x-direction? {b3.end}!={_b3}"
            controls["wire"].setgoal(0, 10.0)
        if abs(time - 47.0) < 1e-6:  # turn the crane (causes pendulum actions)
            _w = (-12 / sqrt(2), 0, 4 + 12 / sqrt(2) - 10)
            assert np.allclose(w.end, _w, atol=0.01), f"Load position before turn {w.end}!={_w}"
            w.damping(q_factor=100.0)
            controls["turn"].setgoal(2, 0.09)
        if abs(time - 54.0) < 1e-6:  # stop turning (causes pendulum actions)
            controls["turn"].setgoal(2, None)

        controls.step(time, dt)
        crane.do_step(time, dt)
        yield (time + dt, crane)


def test_knuckle_boom_crane(show: bool = False):
    crane = _knuckle_boom_crane()
    if not show:  # if nothing can be shown, we do not need to run it
        return
    ani = AnimateCrane(crane, move_knuckle_boom_crane, dt=0.2, t_end=70.0)  # type: ignore  ## It is a Generator!
    ani.do_animation()


if __name__ == "__main__":
    retcode = 0  # pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    logging.basicConfig(level=logging.DEBUG)
    parsolog = logging.getLogger("parso")
    parsolog.setLevel(logging.WARNING)
    pillog = logging.getLogger("PIL")
    pillog.setLevel(logging.WARNING)

    # test_mobile_crane(show=True)
    test_knuckle_boom_crane(show=True)
