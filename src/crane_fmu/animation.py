import logging
from typing import Generator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.animation import FuncAnimation
import numpy as np

from crane_fmu.boom import Boom
from crane_fmu.crane import Crane


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AnimateCrane(object):
    """Wraps the matplotlib FuncAnimation for usage of animating the crane as line animation.

    Args:
        crane (Crane): Instantiated crane object
        frame_gen (Generator): Generator function yielding the tuple (time, crane) with updated crane after time dt
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
        frame_gen: Generator,
        dt: float = 0.1,
        t_end: float = 10.0,
        figsize: tuple[int, int] = (8, 8),
        axes_lim: tuple[tuple[int, int], ...] = ((-10, 10), (-10, 10), (0, 10)),
        interval: int = 200,
        repeat: bool = False,
        title: str = "Crane animation",
        **kwargs, 
    ):
        self.crane = crane
        self.frame_gen = frame_gen
        self.dt = dt
        self.t_end = t_end
        self.figsize = figsize
        self.axes_lim = axes_lim
        self.interval = interval
        self.repeat = repeat
        self.title = title
        self.kwargs = kwargs
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
        for i, b in enumerate(crane.booms()):
            self.lines[i][0].set_data_3d(
                [b.origin[0], b.end[0]],
                [b.origin[1], b.end[1]],
                [b.origin[2], b.end[2]],
            )
        plt.title(f"{self.title} ({time:.1f})", loc="left")

    def do_animation(self):
        """Do the animation. It generates frames and updates the animation plot."""
        _ = FuncAnimation(
            self.fig,
            self.update,  # type: ignore  ## this is a function!
            frames=self.frame_gen(self.crane, dt=self.dt, t_end=self.t_end, **self.kwargs),  # type: ignore
            init_func=self.init_fig,  # type: ignore  ## this is a function!
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
    def __init__(self,
                 origin: np.ndarray = np.array( (0.0,0), float),
                 length: float = 1.0,
                 angles: tuple[float,float] = (90.0,0),
                 degrees: bool = True,
                 v0: np.ndarray = np.array( (0, 1.0, 0), float),
                 q_factor: float = 100.0,
                 dt: float = 0.01,
                 t_end: float = 10.0,
                 figsize: tuple[int, int] = (8, 8),
                 axes_lim: tuple[tuple[int, int],...] = ((-10, 10), (-10, 10), (-10, 10)),
                 interval: int = 10,
                 repeat: bool = False,
                 title: str = "Pendulum animation",
                 buffer: int|None = None,
                 ):
        # instantiate a very simple crane
        self.crane = Crane()
        self.wire = self.crane.add_boom(name="wire",
                                        description="The wire fixed to the fixation. Flexible connection",
                                        mass=1.0,
                                        mass_center=1.0,
                                        boom=(length, *(np.radians(angles) if degrees else angles)),
                                        q_factor=q_factor
                                       )
        self.wire.r_v = v0        
        self.dt = dt
        self.t_end = t_end
        self.figsize = figsize
        self.axes_lim = axes_lim
        self.interval = interval
        self.repeat = repeat
        self.title = title
        self.fig = plt.figure(figsize=self.figsize, layout=None)  # "constrained")
        self.buffer = buffer

    def init_fig(self):
        """Perform the needed initializations."""
        ax = plt.axes(projection="3d")  # , data=line)
        ax.set_xlim(*self.axes_lim[0])
        ax.set_ylim(*self.axes_lim[1])
        ax.set_zlim(*self.axes_lim[2])  # type: ignore [attr-defined] ## according to matplotlib recommendations
        self.line = Line3D( [], [], [], color='blue')
        self.rope = Line3D( [self.wire.origin[0], self.wire.end[0]],
                            [self.wire.origin[1], self.wire.end[1]],
                            [self.wire.origin[2], self.wire.end[2]],
                            color='black',
                            linewidth = 3)
        ax.add_line( self.line)
        ax.add_line( self.rope)
        self.xs = [self.wire.end[0]]
        self.ys = [self.wire.end[1]]
        self.zs = [self.wire.end[2]]

    def update(self, frame):
        """Receives the frames from FuncAnimation with updated crane objects. Draw the booms as lines."""
        time, wire = frame
        self.xs.append(wire.end[0])
        self.ys.append(wire.end[1])
        self.zs.append(wire.end[2])
        if isinstance( self.buffer, int) and len(self.xs) > self.buffer:
            self.xs = self.xs[-self.buffer:]
            self.ys = self.ys[-self.buffer:]
            self.zs = self.zs[-self.buffer:]
        self.line.set_data_3d( self.xs, self.ys, self.zs)
        self.rope.set_data_3d( [self.wire.origin[0], self.wire.end[0]],
                               [self.wire.origin[1], self.wire.end[1]],
                               [self.wire.origin[2], self.wire.end[2]])
        plt.title(f"{self.title} ({time:.1f})", loc="left")

    def frame_gen(self):
        for time in np.linspace(0.0, self.t_end, int(self.t_end / self.dt) + 1):
            self.wire.calc_statics_dynamics(self.dt)
            yield (time + self.dt, self.wire)

    def do_animation(self):
        """Do the animation. It generates frames and updates the animation plot."""
        _ = FuncAnimation(
            self.fig,
            self.update,  # type: ignore  ## this is a function!
            frames=self.frame_gen(),  # type: ignore
            init_func=self.init_fig,  # type: ignore  ## this is a function!
            interval=self.interval,
            repeat=self.repeat,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
