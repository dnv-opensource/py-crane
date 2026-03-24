# pyright: reportUnknownMemberType=false
import logging
from typing import Any, Generator, Protocol, Sequence, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

from py_crane.boom import Boom, Wire
from py_crane.crane import Crane

logger = logging.getLogger(__name__)

FrameGenerator: TypeAlias = Generator[tuple[float, Crane], None, None]


class CraneMovement(Protocol):
    """Protocol for a generator function which creates a movement of a crane through definition and usage of controls.

    A function implementing this protocol acts as a time-domain simulator of a crane:
    It is expected to step through time and sequentially update a crane's state, yielding a crane movement over time
    (in other words, a time-domain simulation of a crane's movement).

    Returns a `Generator` object yielding sequentially updated frames in the form of tuple (time, crane) objects.

    Time is defined global as a simple way to draw the current time together with the title.

    Parameters
    ----------
    crane : Crane
        The crane object to be animated.
    dt : float
        Time step size.
    t_end : float
        End time of the animation.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    FrameGenerator
        A generator which yields sequentially updated frames in the form of tuple (time, crane) objects.

    """

    def __call__(self, crane: Crane, dt: float, t_end: float, **kwargs: Any) -> FrameGenerator: ...


class AnimateCrane(object):
    """Wraps the matplotlib FuncAnimation for usage of animating the crane as line animation.

    Args:
        crane (Crane): Instantiated crane object
        frame_gen (CraneMovement): A function implementing the CraneMovement protocol
        dt (float) = 0.1: step size for time
        t_end (float) = 10.0: end time of the animation
        figsize (tuple) = (8,8): figure size in inches
        axes_lim (tuple) = ((-10,10), (-10,10), (0,10)): Axes limits for x, y and z axes
        interval (int) = 200: Time in milliseconds between animation frames
        title (str) = "Crane animation": Figure title
        **kwargs: optional additional keyword arguments to generator function
    """

    def __init__(
        self,
        crane: Crane,
        movement: CraneMovement,
        dt: float = 0.1,
        t_end: float = 10.0,
        figsize: tuple[int, int] = (8, 8),
        axes_lim: tuple[tuple[int, int], ...] = ((-10, 10), (-10, 10), (0, 10)),
        interval: int = 200,
        repeat: bool = False,
        title: str = "Crane animation",
        **kwargs: Any,
    ):
        self.crane = crane
        self.movement = movement
        self.dt = dt
        self.t_end = t_end
        self.figsize = figsize
        self.axes_lim = axes_lim
        self.interval = interval
        self.repeat = repeat
        self.title = title
        self.kwargs = kwargs
        self.lines: list[list[Line3D]] = []
        self.fig: Figure = plt.figure(figsize=self.figsize, layout=None)  # "constrained")

    def init_fig(self) -> None:
        """Perform the needed initializations."""
        ax: Axes3D = plt.axes(projection="3d")  # , data=line)  # pyright: ignore[reportAssignmentType]
        ax.set_xlim(*self.axes_lim[0])
        ax.set_ylim(*self.axes_lim[1])
        ax.set_zlim(*self.axes_lim[2])
        for b in self.crane.booms():
            lw = {"pedestal": 10, "rope": 1}.get(b.name, 4)
            self.lines.append(
                cast(
                    list[Line3D],
                    ax.plot(
                        [b.origin[0], b.end[0]],
                        [b.origin[1], b.end[1]],
                        [b.origin[2], b.end[2]],
                        linewidth=lw,
                    ),
                )
            )

    def update(self, frame: tuple[float, Crane]):
        """Receives the frames from FuncAnimation with updated crane objects. Draw the booms as lines."""
        time, crane = frame
        for i, b in enumerate(crane.booms()):
            self.lines[i][0].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )
        _ = plt.title(f"{self.title} ({time:.1f})", loc="left")

    def do_animation(self):
        """Do the animation. It generates frames and updates the animation plot."""
        _ = FuncAnimation(
            self.fig,
            self.update,  # type: ignore[reportArgumentType, arg-type]  # ok as long as `blit=False`
            frames=self.movement(self.crane, dt=self.dt, t_end=self.t_end, **self.kwargs),
            init_func=self.init_fig,  # type: ignore[reportArgumentType, arg-type]  # ok as long as `blit=False`
            interval=self.interval,
            repeat=self.repeat,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def close(self):
        """Close the animation."""
        plt.close(self.fig)


class AnimatePendulum(object):
    """Wraps the matplotlib FuncAnimation for usage of animating the pendulum movement of the wire-load (fixed origin).

    Args:
        origin (ndarray) = (0,0,0): the wire origin position
        length (float) = 1.0: the wire length
        angles (tuple[float,float]) = (90,0): initial polar and azimuth angle of wire
        degrees: bool = True: Whether angles are provided in degrees
        v0 (ndarray) = (0,1.0,0): initial speed of load
        q_factor (float) = 100: the quality factor (damping) of the pendulum
        dt (float) = 0.1: step size for time
        t_end (float) = 10.0: end time of the animation
        figsize (tuple) = (8,8): figure size in inches
        axes_lim (tuple) = ((-10,10), (-10,10), (0,10)): Axes limits for x, y and z axes
        interval (int) = 200: Time in milliseconds between animation frames
        title (str) = "Pendulum animation": Figure title
        buffer (int)=None: length of buffer for pendulum trace. None: no limitation
    """

    def __init__(
        self,
        origin: np.ndarray | None = None,
        length: float = 1.0,
        angles: tuple[float, float] = (90.0, 0),
        degrees: bool = True,
        v0: np.ndarray | None = None,
        q_factor: float = 100.0,
        dt: float = 0.01,
        t_end: float = 10.0,
        figsize: tuple[int, int] = (8, 8),
        axes_lim: tuple[tuple[int, int], ...] = ((-10, 10), (-10, 10), (0, 10)),
        interval: int = 10,
        repeat: bool = False,
        title: str = "Pendulum animation",
        buffer: int | None = None,
    ):
        origin = origin if origin is not None else np.array((0.0, 0.0, 0.0), dtype=np.float64)
        v0 = v0 if v0 is not None else np.array((0.0, 1.0, 0.0), dtype=np.float64)
        # instantiate a very simple crane
        self.crane = Crane()
        self.wire = self.crane.add_boom(
            "wire",
            description="The wire fixed to the fixation. Flexible connection",
            mass=1.0,
            mass_center=1.0,
            boom=(length, *(np.radians(angles) if degrees else angles)),
            q_factor=q_factor,
        )
        assert isinstance(self.wire, Wire)
        self.wire.cm_v = v0
        self.dt = dt
        self.t_end = t_end
        self.figsize = figsize
        self.axes_lim = axes_lim
        self.interval = interval
        self.repeat = repeat
        self.title = title
        self.fig = plt.figure(figsize=self.figsize, layout=None)  # "constrained")
        self.buffer = buffer
        self.line: Line3D
        self.rope: Line3D

    def init_fig(self):
        """Perform the needed initializations."""
        ax: Axes3D = plt.axes(projection="3d")  # , data=line)  # pyright: ignore[reportAssignmentType]
        ax.set_xlim(*self.axes_lim[0])
        ax.set_ylim(*self.axes_lim[1])
        ax.set_zlim(*self.axes_lim[2])
        self.line = Line3D([], [], [], color="blue")
        self.rope = Line3D(
            [self.wire.origin[0], self.wire.end[0]],
            [self.wire.origin[1], self.wire.end[1]],
            [self.wire.origin[2], self.wire.end[2]],
            color="black",
            linewidth=3,
        )
        _ = ax.add_line(self.line)
        _ = ax.add_line(self.rope)
        self.xs = [self.wire.end[0]]
        self.ys = [self.wire.end[1]]
        self.zs = [self.wire.end[2]]

    def update(self, frame: tuple[float, Boom]) -> None:
        """Receives the frames from FuncAnimation with updated wire objects. Draws the wire as a line."""
        time, wire = frame
        self.xs.append(wire.end[0])
        self.ys.append(wire.end[1])
        self.zs.append(wire.end[2])
        if isinstance(self.buffer, int) and len(self.xs) > self.buffer:
            self.xs = self.xs[-self.buffer :]
            self.ys = self.ys[-self.buffer :]
            self.zs = self.zs[-self.buffer :]
        self.line.set_data_3d(self.xs, self.ys, self.zs)
        self.rope.set_data_3d(
            [self.wire.origin[0], self.wire.end[0]],
            [self.wire.origin[1], self.wire.end[1]],
            [self.wire.origin[2], self.wire.end[2]],
        )
        _ = plt.title(f"{self.title} ({time:.1f})", loc="left")

    def frame_gen(self):
        for time in np.linspace(0.0, self.t_end, int(self.t_end / self.dt) + 1):
            self.wire.calc_statics_dynamics(self.dt)
            yield (time + self.dt, self.wire)

    def do_animation(self):
        """Do the animation. It generates frames and updates the animation plot."""
        _ = FuncAnimation(
            self.fig,
            self.update,  # type: ignore[reportArgumentType, arg-type]  # ok as long as `blit=False`
            frames=self.frame_gen(),
            init_func=self.init_fig,  # type: ignore[reportArgumentType, arg-type]  # ok as long as `blit=False`
            interval=self.interval,
            repeat=self.repeat,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


class AnimatePlayBackLines(object):
    """Uses the matplotlib FuncAnimation for re-playing lines recorded beforehand.

    Args:
        data (tuple[np.ndarray]): the input data. time is expected as the first array,
           then the base point and then any number of end-points (connected lines)
        lw (tuple[int,...]): Optional specification of the line width of the drawn lines. Default: 1 for all
        figsize (tuple) = (8,8): figure size in inches
        interval (int) = 200: Time in milliseconds between animation frames
        title (str) = "Lines re-player": Figure title
    """

    def __init__(
        self,
        data: Sequence[np.ndarray],
        lw: Sequence[int] | None = None,
        figsize: tuple[int, int] = (8, 8),
        interval: int = 200,
        title: str = "Crane animation",
    ):
        assert len(data) > 0, "No data. Cannot animate."
        self.data = data
        self.lw = (1,) * (len(data) - 1) if lw is None else lw
        self.times = data[0]
        self.figsize = figsize
        self.axes_lim = self._get_axes_lim(data)
        self.interval = interval
        self.title = title
        self.lines: list[list[Line3D]] = []
        self.fig: Figure = plt.figure(figsize=self.figsize, layout=None)  # "constrained")

    def _get_axes_lim(self, data: Sequence[np.ndarray]) -> list[list[float]]:
        """Get the limits of time (data[0]) and all points. Time shall be sorted in ascening order."""
        length = len(data[0])
        assert all(len(data[i + 1]) == length for i in range(len(data) - 1)), (
            f"Columns of 'data' not equal length {length}"
        )
        assert all(data[i].shape == (length, 3) for i in range(1, len(data))), "Data points shall be 3D"
        assert all(data[0][i] < data[0][i + 1] for i in range(length - 1)), "Column 0 (time) is unsorted"
        axes_lim = [[float("inf"), float("-inf")], [float("inf"), float("-inf")], [float("inf"), float("-inf")]]
        for i in range(len(data) - 1):
            for k in range(3):
                axes_lim[k][0] = min(axes_lim[k][0], int(np.min(data[i + 1]) - 1))
                axes_lim[k][1] = max(axes_lim[k][1], int(np.max(data[i + 1]) + 1))
        return axes_lim

    def init_fig(self) -> None:
        """Perform the needed initializations."""
        ax: Axes3D = plt.axes(projection="3d")  # , data=line)  # pyright: ignore[reportAssignmentType]
        ax.set_xlim(self.axes_lim[0])
        ax.set_ylim(self.axes_lim[1])
        ax.set_zlim(self.axes_lim[2])
        start: np.ndarray
        end: np.ndarray

        for i in range(1, len(self.data) - 1):
            start, end, lw = self.data[i][0], self.data[i + 1][0], self.lw[i - 1]
            # start, end, lw in zip(self.data[1:-1], self.data[2:], self.lw, strict=True):
            self.lines.append(
                cast(
                    list[Line3D],
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], linewidth=lw),
                )
            )

    def update(self, row: int):
        """Receives the frames from FuncAnimation with updated line data. Draw the booms as lines by data rows."""
        time = self.data[0][row]
        for i in range(len(self.data) - 2):  # all lines
            self.lines[i][0].set_data_3d(
                [self.data[i + 1][row][0], self.data[i + 2][row][0]],
                [self.data[i + 1][row][1], self.data[i + 2][row][1]],
                [self.data[i + 1][row][2], self.data[i + 2][row][2]],
            )
        _ = plt.title(f"{self.title} ({time:.1f})", loc="left")

    def do_animation(self):
        """Do the animation. It generates frames and updates the animation plot."""
        _ = FuncAnimation(
            self.fig,
            self.update,  # type: ignore[reportArgumentType, arg-type]  # ok as long as `blit=False`
            frames=range(len(self.data[0])),
            init_func=self.init_fig,  # type: ignore[reportArgumentType, arg-type]  # ok as long as `blit=False`
            interval=self.interval,
            repeat=False,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
