from __future__ import annotations

import logging
from typing import Any, Callable, Generator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from component_model.variable import euler_rot_spherical
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

from crane_fmu.boom import Boom
from crane_fmu.enum import Change

logger = logging.getLogger(__name__)


class Crane(object):
    """A basic crane model object for mounting on fixed and movable structures.

    The crane consists of stiff booms (see boom.py), which are connected to each other through

    * stiff connection (`damping=0`): The polar angle in direction of the previous boom can only be changed through new settings.
    * loose connection (`damping>0`): The joint to the previous boom is loose
       and the boom can move freely in all polar directions, i.e. it represents a wire,
       exhibiting pendulum movement with respect to the local center of mass.

    The basic boom `fixation` is automatically added and accessible through `.boom0`.
    Initially the direction of the fixation is set to 'up' (z-axis).
    The fixation direction can be changed through the `d_orientation` 3d angular velocity variable of the crane.
    `orientation` represents the current 3d angle (roll, pitch, yaw, according to ISO 1151–2) vector
    of the crane with respect to its initial orientation.

    The crane should first be instantiated and then the booms added, using `.add_boom()` .
    Added booms can be accessed through the `.booms(reverse=False)` iterator.

    The position of the fixation with respect to the structure where the crane is mounted does not concern the crane.
    For moveable structures (e.g. vessels) it turns out that the crane needs to know

    * the linear acceleration (change of velocity) of the structure, i.e `d_velocity`,
       since the acceleration leads to a force and the pendulum movement of loosely connected booms is affected.
       Consequently, the crane needs to keep track of the velocity of its mounting structure,
       even if it is fixed to it.
    * the angular direction of the fixation with respect to the initial direction, i.e. `orientation` (see above).
    * the booms use polar coordinates for their direction, defined with respect to the direction of the previous boom.
      This implies that the third direction variable is missing for booms, with the consequence that internal boom torsion
      cannot be addressed.
      For fixed booms that is a good approximation,
      but for the load (i.e. the center of mass of the loose connection) it represents the limitation that the load cannot turn.
    """

    to_crane_angle: Callable

    def __init__(self, to_crane_angle: Callable | None = None):
        """Initialize the crane object."""
        Crane.to_crane_angle: Callable = Crane.to_crane_angle_default  # args: (angle, degrees:bool = False)
        if to_crane_angle is not None:  # non-default transformation function
            Crane.to_crane_angle = to_crane_angle
        self._rot: Rot = Rot.from_euler("XYZ", (0, 0, 0))  # placeholder for current rotations (yaw,pitch,roll) of crane
        self._d_rot: Rot | None = None  # placeholder for current angular movement
        self._position = np.array((0.0, 0.0, 0.0), float)
        self.velocity = np.array((0.0, 0.0, 0.0), float)
        self.d_velocity = np.array((0.0, 0.0, 0.0), float)
        self._boom0 = Boom(
            self,
            "fixation",
            "Fixation point of the crane to its parent object or fixed ground. Pseudo-boom object",
            anchor0=None,
            mass=1e-10,
            boom=(1e-10, 0.0, 0.0),
        )

    @property
    def boom0(self) -> Boom:
        return self._boom0

    @boom0.setter
    def boom0(self, newVal: Boom):
        assert isinstance(newVal, Boom), f"A boom object expected as first boom on crane. Got {type(newVal)}"
        self._boom0 = newVal

    def booms(self, reverse=False) -> Generator[Boom]:
        """Iterate through the booms.
        If reverse=True, the last element is first found and the iteration produces the booms from end to start.
        """
        boom: Boom | None = self._boom0
        if reverse:
            while boom is not None and boom.anchor1 is not None:  # walk to the end of the crane
                boom = boom.anchor1
        while boom is not None:
            yield boom
            boom = boom.anchor0 if reverse else boom.anchor1  # type: ignore ## boom on rhs cannot be None

    def boom_by_name(self, name: str) -> Boom | None:
        """Retrieve a boom object by name. None if not found."""
        for b in self.booms():
            if b.name == name:
                return b
        return None

    def add_boom(self, *args, **kvargs):
        """Add a boom to the crane.

        This method represents the recommended way to contruct a crane and then add the booms.
        The `model` and `anchor0` parameters are automatically added to the boom when it is instantiated.
        `args` and `kwargs` thus include all `Boom` parameters, but the `model` and the `anchor0`
        """
        if "anchor0" not in kvargs:
            last = next(self.booms(reverse=True))
            kvargs.update({"anchor0": last})
        return Boom(self, *args, **kvargs)

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, newval: np.ndarray):
        self._position = newval
        self.boom0.origin = newval
        self.boom0.update_child(change=Change.POS)

    @property
    def rot(self) -> Rot:
        return self._rot

    @rot.setter
    def rot(self, rpy: Sequence | np.ndarray):
        """Set a new absolute rotation through an Euler angle."""
        self._rot = self.rotate(rpy, degrees=False, absolute=True)

    def d_rot(self, rpy: Sequence | np.ndarray | None = None) -> Rot | None:
        """Set a new relative rotation through an Euler angle.
        Note: Only the rotation object is defined. The crane is not rotated.
        """
        if rpy is not None:  # set a new value and return the result
            angle = Crane.to_crane_angle(np.array(rpy), degrees=False)
            if np.isclose(angle, (0, 0, 0)):  # stop rotation
                self._d_rot = None
            else:
                self._d_rot = Rot.from_euler("XYZ", angle)  # 0: roll, 1: pitch, 2: yaw
        return self._d_rot

    @classmethod
    def to_crane_angle_default(cls, rpy: np.ndarray, degrees: bool = False):
        """Transform the given extrinsic euler angles into the the coordinate system used by the crane.
        Note: In maritime applications, the North-East-Down is often used,
           while crane uses naturally a North-West-Up system. Both are right handed.
        """
        _angle = np.radians(rpy) if degrees else rpy
        _angle[1] = -_angle[1]
        _angle[2] = -_angle[2]
        return _angle

    def rotate(self, rpy: Sequence | np.ndarray | Rot, degrees: bool = False, absolute: bool = False):
        """Set the orientation to a new value according to ISO 1151–2 (roll,pitch,yaw) - rotations.

        Args:
            rpy (Sequence): Sequence of roll, pitch and yaw angles or a rotation object
            degrees (bool)=False: Optional possibility to supply the angles in degrees. Default is radians.
            absolute (bool)=False: euler rotation as absolute angle or relative to current self._rot

        The current orientation is maintained as scipy rotation object
        based on euler angles (roll,pitch,yaw)
        with x in vessel direction, y to starboard, z down
        """
        if isinstance(rpy, Rot):  # rotation object provided
            self._rot = rpy if absolute else self._rot * rpy
            angle = self._rot.as_euler("XYZ")
        else:  # euler angle provided
            angle = Crane.to_crane_angle(np.array(rpy), degrees)
            rot_angle = Rot.from_euler("XYZ", angle)  # 0: roll, 1: pitch, 2: yaw
            self._rot = rot_angle if absolute else rot_angle * self._rot  # absolute or relative angle
            # print(f"Angle {np.degrees(angle)}. => matrix \n{self._rot.as_matrix()}")
            # print(f"x,y,z->{self._rot.apply((1,0,0))}, {self._rot.apply((0,1,0))}, {self._rot.apply((0,0,1))}")
        self.boom0.boom_setter(euler_rot_spherical(angle, (None, 0.0, 0.0)))  # fixation spherical angle
        return self._rot

    def calc_statics_dynamics(self, dt=None):
        """Run `calc_statics_dynamics()` on all booms in reverse order,
        to get all Boom._c_m_sub and dynamics updated.
        """
        try:
            next(self.booms(reverse=True)).calc_statics_dynamics(dt)
        except StopIteration:
            pass

    def do_step(self, current_time: float, step_size: float) -> bool:
        """Do a simulation step of size `dt` at `time` ."""
        logger.debug(f"CRANE.do_step {current_time}:")
        if any(acc != 0 for acc in self.d_velocity):  # linear acceleration ongoing
            self.velocity += self.d_velocity * step_size
        if any(v != 0 for v in self.velocity):  # position change ongoing
            self.position += self.velocity * step_size  # Note: changes also origin of fixture and all children
        # orientation changes...
        if self._d_rot is not None:
            self.rotate(self._d_rot * step_size, absolute=False)  # Note: changes also rot, ... and all children

        # after all changed input variables are taken into account, update the statics and dynamics of the system
        self.calc_statics_dynamics(step_size)
        # res = "".join( x.name+":"+str(x.end) for x in self.booms())
        logger.debug(f"Crane step {current_time} done. torque:{self.boom0.torque}")
        return True


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
