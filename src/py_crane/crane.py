from __future__ import annotations

import logging
from typing import Any, Callable, Generator, Sequence, TypeAlias

import numpy as np
from component_model.utils.transform import cartesian_to_spherical
from scipy.spatial.transform import Rotation as Rot

from py_crane.boom import Boom, Wire
from py_crane.enums import Change

logger = logging.getLogger(__name__)

CoordinateTransform: TypeAlias = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
]


class Crane(object):
    """A basic crane model object for mounting on fixed or movable structures.

    The crane consists of stiff booms (see `boom.py`), which are connected to each other through

    stiff connection (`q_factor=0`):
        The angles with respect to the previous boom can only be changed through new settings.
    loose connection (`damping>0`):
        The joint to the previous boom is loose and the boom can move freely in all spherical directions,
        i.e. it represents a wire, exhibiting pendulum movement with respect to the local center of mass.

    The basic boom `fixation` is automatically added and accessible through `.boom0`.
    Initially the direction of the fixation is set to 'up' (z-axis).
    The fixation direction can be changed through the `d_orientation` 3d angular velocity variable of the crane.
    `orientation` represents the current 3d angle (roll, pitch, yaw, according to ISO 1151–2) vector
    of the crane with respect to its initial orientation.

    The crane should first be instantiated and then the booms added, using `.add_boom()` .
    Added booms can be accessed through the `.booms()` iterator from fixation to last boom
    or in reverse direction (`.booms(reverse=True`).
    In addition the function `.boom_by_name(boom-name)` is available to get access to a Boom object of the crane.

    The position and the orientation of the crane with respect to the fixation can be changed.
    For cranes mounted on clompletely fixed platforms, these settings never change,
    but for cranes mounted on moveable structures (e.g. vessels) they the following effects:

    linear acceleration of the structure, i.e `d_velocity`
        The acceleration leads to a force on the fixation and the pendulum movement of loosely connected booms is affected.
        The `.position` and `.velocity` getter and setter functions, together with the property `.d_velocity`
        provide access to current values. Note that a constant velocity has no effect on the crane.
    angular direction, i.e. `.angular`
        The angular direction creates a torque on the fixation.
        The property getter and setter functions `.angular` facilitate instantaneous setting of the Euler angle
        of the fixation with respect to the structure. The initial angle is (roll=0,pitch=0,yaw=0).
        The function `.rotate(euler-angle)` is recommended to be used to
    angular acceleration, i.e. `.d_angular`
        Angular direction changes cause both torques and pendulum action.
        The property getter and setter functions `.d_angular` facilitate instantaneous changes of the Euler angle
        of the fixation with respect to the structure.

    Instantaneous changes are in general non-physical. In reality changes happen more or less smooth, which implies
    that changes should be ramped up/down (in principle using infinite orders of derivatives).
    In practice we limit ourselves to usage of up to second order derivatives,
    i.e. the assumption that acceleration changes happen instantaneously, or that forces can be applied instantaneously.
    Since cranes are often driven electrically and since electric motors can deliver high torques rather instantaneously,
    the approximation seems reasonable.

    The introduction of derivatives introduces the notion of time. The function `.calc_statics_dynamics( dt)`
    updates all changes (forces, torque, pendulum movement) over the next time interval given the current settings.
    In simulations this function shall be called as part of every simulation step.

    .. assumption:: The dynamics of changes is simulated to second order, accepting instantaneous acceleration changes.

    Other things to note:

    * The booms use polar coordinates for their direction, defined with respect to the direction of the previous boom.
      This implies that the third direction variable is missing for booms, with the consequence that internal boom torsion
      cannot be addressed. For fixed booms that is a good approximation, but for the load
      (i.e. the center of mass of the loose connection) it implies the limitation that the load cannot turn.
    * all angles inside the crane are in radians, lengths are in meters and masses are in kg.
      Only when the interface to the outside world is defined as FMU,
      the possibility exists that parameters, inputs and outputs can be defined in other `display units`, e.g. degrees.
    * forces and torque are only calculated in relation to the fixation (the whole crane),
      not in relation to moving joints.

    Args:
        to_crane_angle (Callable) = None: Optional possibility to specify a non-default transformation
          from vessel Euler angles to crane coordinate system. Default: (north-east-down as r-p-y + north-west-up)
          Should be a function of an Euler angle, corresponding also to the choice of 'degrees'.
    """

    to_crane_angle: CoordinateTransform

    def __init__(self, to_crane_angle: CoordinateTransform | None = None):
        """Initialize the crane object."""
        self.to_crane_angle: CoordinateTransform = self.to_crane_angle_default  # args: (angle, degrees:bool = False)
        if to_crane_angle is not None:  # non-default transformation function
            self.to_crane_angle = to_crane_angle
        self._rot: Rot = Rot.from_euler("XYZ", (0, 0, 0))  # current rotations (roll,pitch,yaw) of crane
        self._angular = np.array((0.0, 0.0, 0.0), dtype=np.float64)  # current angle as (roll,pitch,yaw)
        self._d_angular = np.array((0.0, 0.0, 0.0), dtype=np.float64)  # current angular speed as (roll,pitch,yaw)
        self.d2_angular = np.array(
            (0.0, 0.0, 0.0), dtype=np.float64
        )  # current angular acceleration as (roll,pitch,yaw)
        self._position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self._velocity = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self.d_velocity = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        self._boom0 = Boom(
            self,
            "fixation",
            "Fixation point of the crane to its parent object or fixed ground. Pseudo-boom object",
            anchor0=None,
            mass=1e-10,
            boom=(1e-10, 0.0, 0.0),
        )
        self.boom_: Boom = self.boom0  # keep track of the last boom
        self.torque = np.array((0, 0, 0), dtype=np.float64)
        self.force = np.array((0, 0, 0), dtype=np.float64)
        self.current_time: float = 0.0  # used to make current_time from do_step known for the whole crane

    @property
    def boom0(self) -> Boom:
        return self._boom0

    @boom0.setter
    def boom0(self, newVal: Boom):
        assert isinstance(newVal, Boom), f"A boom object expected as first boom on crane. Got {type(newVal)}"
        self._boom0 = newVal

    def booms(self, *, reverse: bool = False) -> Generator[Boom]:
        """Iterate through the booms.
        If reverse=True, the last element is first found and the iteration produces the booms from end to start.
        """
        boom: Boom | None
        # Determine the start boom, i.e. the boom the iteration shall start from
        start_boom: Boom = self._boom0
        if reverse:
            while start_boom.anchor1 is not None:  # walk to the end of the crane
                start_boom = start_boom.anchor1
        # Iterate through the booms
        boom = start_boom
        while boom is not None:
            yield boom
            boom = boom.anchor0 if reverse else boom.anchor1

    def boom_by_name(self, name: str) -> Boom | None:
        """Retrieve a boom object by name. None if not found."""
        for b in self.booms():
            if b.name == name:
                return b
        return None

    def add_boom(
        self,
        name: str,
        /,
        **kwargs: Any,
    ) -> Boom:
        """Add a boom to the crane.

        This method represents the recommended way to contruct a crane and then add the booms.
        The `model` and `anchor0` parameters are automatically added to the boom when it is instantiated.
        `args` and `kwargs` thus include all `Boom` parameters, but the `model` and the `anchor0`
        """
        if "anchor0" not in kwargs:
            last = next(self.booms(reverse=True))
            kwargs.update({"anchor0": last})
        if "q_factor" in kwargs and kwargs["q_factor"] > 0.0:
            self.boom_ = Wire(self, name, **kwargs)
        else:
            self.boom_ = Boom(self, name, **kwargs)
        return self.boom_  # the new last boom

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, newval: np.ndarray):
        """Instantaneous movement of the crane. Not meant for dynamical analysis. Use d_velocity for that."""
        self._position = newval
        self.boom0.origin = newval
        self.boom0.update_child(change=Change.POS)

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, newval: np.ndarray):
        """Instantaneous change of crane velocity. Not meant for dynamical analysis. Use d_velocity for that."""
        self._velocity = newval

    @property
    def angular(self) -> np.ndarray:
        """Get the current euler angle of the crane (internally stored as Rot)."""
        return self._angular  # retrieve the stored value, while the orientation setting is stored in _rot

    @angular.setter
    def angular(self, newval: np.ndarray):
        """Instantaneous change of euler angle of the crane. Stored internally as Rot.
        Not meant for dynamical analysis. Use d2_angular for that.
        """
        self.rotate(newval, absolute=True)
        self._angular = newval  # remember, so that we can retrieve it. The orientation itself is in _rot

    @property
    def d_angular(self) -> np.ndarray:
        """Get the current angular velocity (euler angles)."""
        return self._d_angular

    @d_angular.setter
    def d_angular(self, newval: np.ndarray):
        """Instantaneeous change of angular valocity. Not menat for dynamical analysis. Use d2_angular for that."""
        self._d_angular = newval

    def rot(self, rpy: Sequence[float] | np.ndarray[tuple[int], np.dtype[np.float64]] | None = None):
        """Get/Set a new absolute rotation through an Euler angle."""
        if rpy is not None:  # set a new value
            self._rot = self.rotate(rpy, absolute=True)
        return self._rot

    def to_crane_angle_default(
        self, rpy: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Transform the given extrinsic euler angles into the the coordinate system used by the crane.

        Note: In maritime applications, the North-East-Down is often used,
            while crane uses naturally a North-West-Up system. Both are right handed.
        """
        _angle = np.array(rpy, dtype=np.float64)  # radians expected here!
        _angle[1] = -_angle[1]
        _angle[2] = -_angle[2]
        return _angle

    def rotate(
        self,
        rpy: Sequence[float] | np.ndarray[tuple[int], np.dtype[np.float64]] | Rot,
        absolute: bool = False,
    ):
        """Set the orientation to a new value according to ISO 1151–2 (roll,pitch,yaw) - rotations.

        Args:
            rpy (Sequence): Sequence of roll, pitch and yaw angles or a rotation object
            absolute (bool)=False: euler rotation as absolute angle or relative to current self._rot

        The current orientation is maintained as scipy rotation object
        based on euler angles (roll,pitch,yaw)
        with x in vessel direction, y to starboard, z down
        """
        if isinstance(rpy, Rot):  # rotation object provided
            self._rot = rpy if absolute else self._rot * rpy
            self._angular = self._rot.as_euler("XYZ")
        else:  # euler angle provided
            angle = self.to_crane_angle(np.array(rpy, dtype=np.float64))
            rot_angle = Rot.from_euler("XYZ", angle)  # 0: roll, 1: pitch, 2: yaw
            self._rot = rot_angle if absolute else rot_angle * self._rot  # absolute or relative angle
            self._angular = (
                np.array(rpy, dtype=np.float64) if absolute else self._angular + np.array(rpy, dtype=np.float64)
            )
        fixation_boom = cartesian_to_spherical(self._rot.apply((0.0, 0.0, 1e-10)))
        fixation_boom[0] = None
        self.boom0.boom_setter(list(fixation_boom), ch=Change.ROT.value)  # fixation spherical angle
        return self._rot

    def calc_statics_dynamics(self, dt: float | None = None):
        """Run `calc_statics_dynamics()` on all booms in reverse order,
        to get all Boom._c_m_sub and dynamics updated.
        """
        try:
            next(self.booms(reverse=True)).calc_statics_dynamics(dt)
        except StopIteration:
            pass
        _M0, _c0 = self.boom0.c_m_sub
        _M1, _c1 = self.boom_.c_m_sub
        self.c_m_sub = (_M0 + _M1, (_M0 * _c0 + _M1 * _c1) / (_M0 + _M1))  # c_m_sub for whole crane
        # after the boom properties are updated, we can calculate the crane forces and torques
        self.torque = self.boom0.torque + self.boom_.torque
        self.force = self.boom0.force + self.boom_.force
        # print(f"dt:{dt}. Force: {self.force}, torque: {self.torque}, v:{self._velocity}, a:{self.boom0.acceleration}")

    #         if dt is not None:
    #             self._velocity = (c_m_sub - c_m_sub1) / dt
    #             acceleration = (self._velocity - velocity0) / dt
    #                 assert np.linalg.norm(self._velocity) < 1e50, f"Very high velocity {self._velocity}. Check time intervals!"
    #                 if self.model.calc_boom_forces_torques:
    #                     self.torque += m * np.cross(c_m_sub, acceleration)  # type: ignore ## np issue
    #                     # linear force due to acceleration in boom direction
    #                     self.force = m * np.dot(self.direction, acceleration) * self.direction  # type: ignore

    def do_step(self, current_time: float, step_size: float) -> bool:
        """Do a simulation step of size `dt` at `time` ."""
        self.current_time = current_time
        if any(acc != 0 for acc in self.d_velocity):  # linear acceleration ongoing
            self._velocity += self.d_velocity * step_size
        if any(v != 0 for v in self._velocity):  # position change ongoing
            self.position += self._velocity * step_size  # Note: changes also origin of fixture and all children
        # orientation changes...
        if not np.allclose(self.d2_angular, (0.0, 0.0, 0.0)):
            self._d_angular += step_size * self.d2_angular
        if not np.allclose(self._d_angular, (0.0, 0.0, 0.0)):
            self.rotate(self._d_angular * step_size, absolute=False)  # Note: changes also rot, ... and all children

        # after all changed input variables are taken into account, update the statics and dynamics of the system
        self.calc_statics_dynamics(step_size)
        # res = "".join( x.name+":"+str(x.end) for x in self.booms())
        logger.debug(f"Crane step {current_time} done. torque:{self.boom0.torque}")
        return True
