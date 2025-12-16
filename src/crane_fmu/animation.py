import logging
from typing import Generator

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from crane_fmu.crane import Crane

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AnimateCrane(object):
    """Wraps the matplotlib FuncAnimation for usage of animating the crane as line animation.

    Args:
        crane (Crane): Instantiated crane object
        frame_gen (Generator): Generator function yielding the tuple (time, crane) with updated crane after time dt
        dt (float) = 0.1: step size for time
        end_time (float) = 10.0: end time of the animation
        figsize (tuple) = (8,8): figure size in inches
        axes_lim (tuple) = ((-10,10), (-10,10), (0,10)): Axes limits for x, y and z axes
        interval (int) = 200: Time in milliseconds between animation frames
        title (str) = "Crane animation": Figure title
    """

    def __init__(
        self,
        crane: Crane,
        frame_gen: Generator,
        dt: float = 0.1,
        end_time: float = 10.0,
        figsize: tuple[int, int] = (8, 8),
        axes_lim: tuple[tuple[int, int]] = ((-10, 10), (-10, 10), (0, 10)),
        interval: int = 200,
        title: str = "Crane animation",
    ):
        self.crane = crane
        self.frame_gen = frame_gen
        self.dt = dt
        self.end_time = end_time
        self.figsize = figsize
        self.axes_lim = axes_lim
        self.interval = interval
        self.title = title
        self.lines: list = []
        self.fig = plt.figure(figsize=self.figsize, layout=None)  # "constrained")

    def init_fig(self):
        """Perform the needed initializations."""
        ax = plt.axes(projection="3d")  # , data=line)
        ax.set_xlim(*self.axes_lim[0])
        ax.set_ylim(*self.axes_lim[1])
        ax.set_zlim(*self.axes_lim[2])  # type: ignore [attr-defined] ## according to matplotlib recommendations
        for b in self.crane.booms():
            lw = {"pedestal": 10, "rope": 1}.get(b.name, 4)
            self.lines.append(
                ax.plot(
                    [b.origin[0], b.end[0]],
                    [b.origin[1], b.end[1]],
                    [b.origin[2], b.end[2]],
                    linewidth=lw,
                )
            )

    def update(self, frame):
        """Receives the frames from FuncAnimation with updated crane objects. Draw the booms as lines."""
        time, crane = frame
        ends = []
        for i, b in enumerate(crane.booms()):
            self.lines[i][0].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )
            ends.append(b.end)
        plt.title(f"{self.title} ({time:.1f})", loc="left")

    def do_animation(self):
        _ = FuncAnimation(
            self.fig,
            self.update,  # type: ignore  ## this is a function!
            frames=self.frame_gen(self.crane, dt=self.dt, t_end=self.end_time),  # yields crane object as frame
            init_func=self.init_fig,  # type: ignore  ## this is a function!
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
