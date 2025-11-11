from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
from component_model.model import Model
from component_model.variable import Variable
from component_model.variable_naming import VariableNamingConvention
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from crane_fmu.boom_fmu import BoomFMU
from crane_fmu.crane import Crane

logger = logging.getLogger(__name__)


class CraneFMU(Model, Crane):
    """A crane object built from stiff booms
    including a defined FMI interface, ready to be packaged as FMU through `component_model` and `PythonFMU`.
    The crane should first be instantiated and then the booms added, using `.add_boom()` .
    The basic boom `fixation` is automatically added and accessible through `.boom0`
    and can be used to access the other added booms through `booms(reverse=False)` .

    Note that this is still an abstract crane. It needs another extension to define concrete booms and interfaces.
       See MobileCrane on how this can be done.

    Args:
        name (str): the name of the crane instant
        description (str) = None: An (optional
        author (str) = "Siegfried Eisinger (DNV)
        version (str) = "0.1"
        u_length (str) = "m": length display units (internally m are used)
        u_time (str) = 's': time display units (internally seconds are used)
    """

    def __init__(
        self,
        name: str,
        description: str = "A crane model",
        author: str = "Siegfried Eisinger (DNV)",
        version: str = "0.1",
        degrees: bool = False,
        u_length: str = "m",
        u_time: str = "s",
        **kwargs: Any,
    ):
        """Initialize the crane object."""
        super().__init__(name=name, description=description, author=author, version=version, **kwargs)
        Crane.__init__(self, to_crane_angle=None)
        self.degrees = degrees
        self.variable_naming = VariableNamingConvention.structured
        self.u_length = u_length
        self.u_time = u_time
        self._boom0 = BoomFMU(
            self,
            "fixation",
            "Fixation point of the crane to its parent object or fixed ground. Pseudo-boom object",
            anchor0=None,
            mass="1e-10 kg",
            mass_rng=None,
            boom=(1e-10, 0, 0),
            boom_rng=(None, (), ()),
        )
        self._velocity = Variable(
            self,
            "velocity",
            "Crane change of position per time unit (speed) in 3D",
            causality="input",
            variability="continuous",
            start=("0.0",) * 3,
        )
        self._d_velocity = Variable(
            self,
            "d_velocity",
            "Crane change of velocity per time unit (acceleration) in 3D",
            causality="input",
            variability="continuous",
            start=("0.0",) * 3,
        )
        _ = Variable(
            self,
            "angular",
            "Crane angle as 3D Euler roll-pitch-yaw angle",
            causality="input",
            variability="continuous",
            start=("0.0 deg",) * 3 if self.degrees else ("0.0 rad",) * 3,
        )
        self._d_angular = Variable(
            self,
            "d_angular",
            "Crane change of angle per time unit (angular velocity) as 3D Euler roll-pitch-yaw angle",
            causality="input",
            variability="continuous",
            start=(f"0.0 deg/{u_time}",) * 3 if self.degrees else (f"0.0 rad/{u_time}",) * 3,
        )
        self._d2_angular = Variable(
            self,
            "der(d_angular)",
            "Angualar acceleration of crane in rad/s**2:",
            causality="input",
            variability="continuous",
            start=(f"0.0 deg/{u_time}**2",) * 3 if self.degrees else (f"0.0 rad/{u_time}**2",) * 3,
            local_name="d2_angular",
        )
        self._torque = Variable(
            self,
            "torque",
            "Crane change of position per time unit (speed) in 3D",
            causality="output",
            variability="continuous",
            start=("0.0 N.m",) * 3,
        )

    def add_boom(self, *args, **kvargs):
        """Add a boom to the crane. Overridden to ensure that a BoomFMU is added.

        This method represents the recommended way to contruct a crane and then add the booms.
        The `model` and `anchor0` parameters are automatically added to the boom when it is instantiated.
        `args` and `kwargs` thus include all `Boom` parameters, but the `model` and the `anchor0`
        """
        if "anchor0" not in kvargs:
            last = next(self.booms(reverse=True))
            kvargs.update({"anchor0": last})
        return BoomFMU(self, *args, **kvargs)

    def ensure_boom(self, boom: BoomFMU):
        """Ensure that the boom is registered before structured variables are added to it.
        Otherwise their owner does not exist.
        """
        if not hasattr(self, boom.name):
            setattr(self, boom.name, boom)

    def do_step(self, current_time: float, step_size: float) -> bool:
        status = Model.do_step(self, current_time, step_size)
        Crane.do_step(self, current_time, step_size)
        return status

    # properties and functions available from Crane
    #  boom0 property and setter
    #  booms(self, reverse=False)               Boom iterator
    #  boom_by_name(self, name: str)            -> Boom | None:
    #  add_boom(self, *args, **kvargs)          Add a boom to the crane.
    #  calc_statics_dynamics(self, dt=None)     Run `calc_statics_dynamics()` on all booms in reverse order


class Animation:
    """Animation of the crane via matplotlib.
    Due to issues with multiple CPU processes, this can currently not be used in conjunction with OSP.

    Args:
        crane (Crane): a reference to the crane which shall be animated
        elements (dict)={}: a dict of visual elements to include in the animation.
          Each dictionary element is represented by an empty list which is filled by the element references during init,
          so that their position, ... can be changed during the animation
        interval (float)=0.1: waiting interval between simulation steps in s
        viewAngle (tuple)=(20,45,0): Optional change of initial view angle as (elevation, azimuth, roll) (in degrees)
    """

    def __init__(
        self,
        crane: Crane,
        elements: dict[str, list] | None = None,
        interval: float = 0.1,
        figsize: tuple[float, float] = (9, 9),
        xlim: tuple[float, float] = (-10, 10),
        ylim: tuple[float, float] = (-10, 10),
        zlim: tuple[float, float] = (0, 10),
        viewAngle: tuple[float, float, float] = (20, 45, 0),
    ):
        """Perform the needed initializations of an animation."""
        self.crane: Crane = crane
        self.elements: dict[str, Any] | None = elements
        self.interval: float = interval

        _ = plt.ion()  # run the GUI event loop
        self.fig: Figure = plt.figure(figsize=figsize, layout="constrained")
        ax: Axes3D = Axes3D(fig=self.fig)
        #        ax = plt.axes(projection="3d")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=viewAngle[0], azim=viewAngle[1], roll=viewAngle[2])
        sub: list[list] = [[], [], []]
        if isinstance(self.elements, dict):
            for b in self.crane.booms():  # walk along the series of booms
                if "booms" in self.elements:  # draw booms
                    self.elements["booms"].append(
                        ax.plot(
                            [b.origin[0], b.end[0]],
                            [b.origin[1], b.end[1]],
                            [b.origin[2], b.end[2]],
                            linewidth=b.animationLW,
                        )
                    )
                if "c_m" in self.elements:  # write mass of boom as string on center of mass point
                    self.elements["c_m"].append(
                        ax.text(
                            b.c_m[0],
                            b.c_m[1],
                            b.c_m[2],
                            s=str(int(b.mass.start)),  # type: ignore ## pyright confusion about 3D plots
                            color="black",
                        )
                    )
                if "c_m_sub" in self.elements:
                    for i in range(3):
                        sub[i].append(b.c_m_sub[1][i])
            if "c_m_sub" in self.elements and len(sub[0]):
                self.elements["c_m_sub"].append(
                    ax.plot(sub[0], sub[1], sub[2], marker="*", color="red", linestyle="")
                )  # need to put them in as plot and not scatter3d, such that coordinates can be changed in a good way
            if "current_time" in self.elements:
                self.elements["current_time"].append(
                    ax.text(
                        ax.get_xlim()[0],
                        ax.get_ylim()[0],
                        ax.get_zlim()[0],
                        s="time=0",
                        color="blue",
                    )
                )

    def update(self, current_time=None):
        """Based on the updated crane, update data as defined in elements."""
        sub: list[list] = [[], [], []]
        assert isinstance(self.elements, dict), "elements dict required at this stage"
        for i, b in enumerate(self.crane.booms()):
            if "booms" in self.elements:
                assert self.elements["booms"] is not None
                self.elements["booms"][i][0].set_data_3d(
                    [b.origin[0], b.end[0]],
                    [b.origin[1], b.end[1]],
                    [b.origin[2], b.end[2]],
                )
            if "c_m" in self.elements:
                assert self.elements["c_m"] is not None
                self.elements["c_m"][i].set_x(b.c_m_sub[1][0])
                self.elements["c_m"][i].set_y(b.c_m_sub[1][1])
                self.elements["c_m"][i].set_z(b.c_m_sub[1][2])
            if "c_m_sub" in self.elements:
                for i in range(3):
                    sub[i].append(b.c_m_sub[1][i])
        if "c_m_sub" in self.elements and len(sub[0]):
            self.elements["c_m_sub"][0][0].set_data_3d(sub[0], sub[1], sub[2])
        if "current_time" in self.elements and current_time is not None:
            self.elements["current_time"][0].set_text("time=" + str(round(current_time, 1)))

        self.fig.canvas.draw_idle()  # drawing updated values
        self.fig.canvas.flush_events()  # This will run the GUI event loop until all UI events currently waiting have been processed
        # time.sleep( self.interval)

    def interactive_off(self):
        plt.ioff()
