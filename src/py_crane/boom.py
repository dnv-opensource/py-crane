from __future__ import annotations

import logging
from math import isnan, nan, sqrt
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from component_model.utils.transform import (
    cartesian_to_spherical,
    normalized,
    rot_from_spherical,
    rot_from_vectors,
)
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot

from py_crane.enums import Change

if TYPE_CHECKING:
    import py_crane.crane

# Type Alias for a 1-dim array with 3 elements. Used throughout the code to denote 3D vectors.
TVector: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

logger = logging.getLogger(__name__)


class Boom(object):
    r"""Basic Boom model object, representing one element of a crane,
    modelled as stiff rod with length, mass and mass-center.

    .. assumption:: All booms of the crane are completely stiff, excluding bending, stretching and vibration
      modes of motion. This includes also loose connections,
      i.e. the wire is assumed straight and without stretch and torsion.

    .. assumption:: Per default, the mass center of a boom is located at the geometric center of the boom.
      This can be changed using the tuple form of the :py:attr:`mass_center` parameter.

    .. limitation:: The degrees of freedom for the boom (possible ranges in which changes of parameters are allowed)
      cannot be directly defined within the basic Boom object.
      These can be defined as part of the `Variable` definitions when packaging a crane as FMU,
      see :doc:`py_crane.crane_fmu` and :doc:`py_crane.boom_fmu`.
      Alternatively the control module from the package ``component-model`` can be used to define control goals,
      which also includes limits on value, change of value and acceleration of value.

    .. assumption:: All units are assumed as basic units (meter, radian, kg).
      Additional display units can be defined when packaging a crane as FMU.

    Alternative to instantiating a Boom as :py:class:`Boom` (...),
    :py:meth:`.Crane.add_boom` of the related crane can be called (recommended).
    In this case the parameters ``model`` and ``anchor0`` to :py:class:`Boom` shall not be included (are automatically added).

    There are two fundamentally different types of booms

    stiff connection boom (:py:attr:`q_factor` =0):
        The spherical angles with respect to the previous boom can only be changed through new settings.
    loose connection boom(:py:attr:`q_factor` >0):
        The joint to the previous boom is loose and the boom can move freely in all spherical directions,
        i.e. it represents a wire (with load), exhibiting pendulum movement with respect to the local center of mass.
        :py:attr:`q_factor` serves at the same time as a definition of the pendulum damping.

    .. limitation:: The wire is always streched (stiff) and free-falling movement of the load
       (e.g. due to high accelerations of parent boom or length changes) lead to error messages.

    Basic boom movements are

    length change, i.e. :py:meth:`.boom_setter` ( (new-length, None, None))
        instantaneously sets a new length to the boom and updates all children booms so that they are properly connected.
    rotation, i.e. `.boom_setter` ( (None, new-polar, new-azimuth))
        instantaneously rotates the boom around its origin.
        The reference direction is the direction of the parent boom (i.e. as if the parent boom was in z-direction).
        Successively all child booms are updated (i.e. rotated accordingly) to obey the stiffness of the crane.
        Note that loose connection booms are not allowed to rotate actively,
        since that would interfere with pendulum movement.
    mass change, i.e. `.mass`:
      instantanteous mass change of boom (e.g. add or remove load).
      Note that instantaneous changes make sense in this case, since loads are changed instantaneously
      and since wire bending is not modelled.

    After any movement

    * the internal variables (center of mass, positions, etc.) are re-calculated
      using :py:meth:`calc_statics_dynamics`. If `dt=None` instantaneous changes are assumed.
    * the attached booms are moved in accordance
    * and the center of mass of the sub-system (the boom with attached booms) is re-calculated.
    * Finally the parent booms are informed about the change of the sub-system center of mass,
      leading to re-calculation of their internal variables.

    .. note:: initialization variables are designed for easy management of parameters,
       while derived internal variables are more complex (often 3D vectors)

    Args:
        model (py_crane.crane.Crane): The model object owning the boom
        name (str): The short name of the boom (unique within crane)
        description (str) = '':  An optional description of the boom
        anchor0 (Boom): the boom object to which this Boom is attached.
            There is exactly one boom where this is set to None, which is the crane `fixation`
            and which is automatically provided by the crane, i.e. for real booms this is never None.
        mass (float): Parameter denoting the mass of the boom in kg.
          For most booms this is fixed, but when modelling loads as part of the (last) boom this can change.
        mass_center (float,tuple): Parameter denoting the position of the center of mass of the boom,
            provided as portion of the length (as float).
            Optionally the absolute displacements in x- and y-direction (assuming the boom in z-direction) can be added
            e.g. (0.5,-0.5, 1): (in m) halfway down the boom displaced 0.5m in the -x direction and 1m in the y direction
        boom (tuple): A tuple defining the boom relative to its parent in spherical (ISO 80000) coordinates.
            From the parent boom the following attributes are automatically inferred:

            * origin: end of the parent boom => cartesian coordinate of the base point of the boom
            * pole axis: direction vector of the parent boom => local cartesian z-axis
            * reference direction in equator plane:
              crane x-direction or azimuth angle of connecting boom => local cartesian x-axis

            The boom is then defined in local polar coordinates:

            * length: the length of the boom (in m)
            * polar: a rotation angle for a rotation around the negative y-axis (away from z-axis) against the clock.
            * azimuth: a rotation angle for a rotation around the positive z-axis (away from the x-axis) against the clock.

           :: note..: The boom is used to keep length and local coordinate system up-to-date,
               while the active work variables are the cartesian origin, direction and length
        q_factor (float)=0.0: optional possibility to implement a loose connection between booms.

            * if q_factor=0.0, the connection to the parent boom is stiff according to the boom angle settings
            * if q_factor > 0, the crane boom (the wire) is implemented as a stiff rod
              with a loose connection hanging from the parent boom.
            * The q_factor denotes the dimensionless quality factor (energy stored/energy lost per radian),
              which is also equal to
              :math:`2 \ln( A_i/A_{i+1}) = \tau \sqrt{\omega_0^2 - \gamma^2}`,
              :math:`\gamma = 1/(2 \tau)`,
              with
              :math:`\tau > 1/(2\omega_0)`
              and
              `A`: Amplitude,
              :math:`\omega_0=\sqrt{g/L}`: natural angular frequency  of pendulum,
              :math:`\tau`: characteristic damping time
              :math:`\gamma`: pendulum damping.
              This does not cover critically and over-damped systems, which are in any case not realistic for crane wires).
              In many systems the `q_factor` is a fixed parameter, but for testing purposes and if the wire length changes,
              :py:meth:`.damping` is included, which allows changing of related parameters in a consistent way.


        limits (tuple): Optional tuple of control limits for active boom control. See ControlLimits
        tolerance (float)=1e-5: Accuracy during pendulum calculations

    .. note:: This offers basic functionality. Variable changes and related updates are performed directly on the variables
    .. assumption:: Center of mass: `_c_m` is the local mass-center measured relative to origin. `_c_m_sub` is a global quantity
    """

    def __init__(
        self,
        model: "py_crane.crane.Crane",
        name: str,
        description: str = "",
        anchor0: Boom | None = None,
        mass: float | str = 1.0,
        mass_center: float | tuple[float, float, float] = 0.5,
        boom: tuple[float, float, float] | list[float] = (1.0, 0.0, 0.0),
        q_factor: float = 0.0,
        limits: tuple[float, float, float] | None = None,
        tolerance: float = 1e-5,
        additional_checks: bool = False,
        **kwargs: Any,  # for compatibility with derived classes
    ):
        self._model: py_crane.crane.Crane = model
        self.anchor0: Boom | None = anchor0
        self.anchor1: Boom | None = None  # so far. If a boom is added, this is changed
        self._name: str = name
        self.description: str = description
        self.q_factor: float = q_factor
        self.tolerance: float = tolerance
        self.additional_checks = additional_checks
        self.direction: TVector = np.array((0.0, 0.0, -1.0), dtype=np.float64)  # default for non-fixed booms
        self.r_v: TVector = np.array(
            (0.0, 0.0, 0.0), dtype=np.float64
        )  # velocity of CoM relative to .origin (pendulum)
        self.r_acc: TVector = np.array((0.0, 0.0, 0.0), dtype=np.float64)  # acc. of CoM relative to .origin (pendulum)
        # default animation line width of boom. Can be changed, e.g. 10 pedestal and 2 for wire.
        # self.animationLW: int = 5
        self.origin: TVector
        if self.anchor0 is None:  # this defines the fixation of the crane as a 'pseudo-boom'
            boom = (1e-10, 0.0, 0.0)  # z-axis in spherical coordinates
            self.origin = np.array((0.0, 0.0, -1e-10), dtype=np.float64)
        else:
            self.origin = self.anchor0.end
            assert self.anchor0.q_factor == 0.0, (
                f"Trying to attach boom {self.name} to flexible boom {self.anchor0.name}. Not implemented"
            )
            self.anchor0.anchor1 = self
        assert isinstance(mass, float), f"At this stage mass should be a float. Found {type(mass)}"
        self.mass: float = mass
        self.mass_center: tuple[float, float, float] = (
            mass_center if isinstance(mass_center, tuple) else (mass_center, 0.0, 0.0)
        )
        self.boom = np.array(boom, dtype=np.float64)  # length, polar, azimuth
        # rot denotes the rotation which turns (0,0,1) into the direction
        self._rot: Rot = Rot.identity() if self.anchor0 is None else self.anchor0._rot * rot_from_spherical(boom[1:])
        self.direction = self._rot.apply((0.0, 0.0, 1.0))
        self._new_len: float | None = None  # place holder for length changes
        # self._c_m = np.array( (0.0, 0.0, 0.0), dtype=np.float64) # just to make _c_m known. Updated by method c_m
        # save the current value, running method self.c_m
        self._c_m: TVector = self.c_m
        self._c_m_sub: tuple[float, TVector] = (
            self.mass,
            self._c_m,
        )  # updated by calc_statics_dynamics
        _ = self.damping(q_factor=self.q_factor)  # pre-calculate self._damping_time for usage in pendulum
        # self.control = Control( ('len','polar','azimuth'), limits) # control object (without goals)
        self.calc_statics_dynamics(dt=None)
        logger.info(f"BOOM {self._name} {self.origin}->{self.end}. |{self.length} | {self.direction} | {self.q_factor}")

    def __getitem__(self, idx: int | str):
        """Facilitate subscripting booms. 'idx' denotes the connected boom with respect to self.
        Negative indices count from the tail. str indices identify booms by name.
        """
        b: Boom | None = self
        assert b is not None
        if isinstance(idx, str):  # retrieve by name
            while True:
                if b is None or b.name != idx and b.anchor1 is None:
                    return None
                elif b.name == idx:
                    return b
                else:
                    b = b.anchor1

        elif idx >= 0:
            for _ in range(idx):
                if b.anchor1 is None:
                    logger.critical(f"Erroneous index {idx} with respect to boom {self.name}")
                    raise IndexError(f"Erroneous index {idx} with respect to boom {self.name}")
                b = b.anchor1
            return b
        else:
            while b.anchor1 is not None:  # spool to tail
                b = b.anchor1
            for _ in range(abs(idx) - 1):  # go back from tail
                if b.anchor0 is None:
                    logger.critical(f"Erroneous index {idx} with respect to boom {self.name}")
                    raise IndexError(f"Erroneous index {idx} with respect to boom {self.name}")
                b = b.anchor0
            return b

    # @property
    # def boom(self):
    #     return getattr(self._model, self._name + ".boom")  # access to value (owned by model)

    def rot(self, newval: Rot | None = None) -> Rot:
        """Access the rotation object from outside the boom.
        Since this is a function for the model, we make it a function also here.
        """
        if newval is not None:
            self._rot = newval
        return self._rot

    def boom_setter(self, val: tuple[float | None, ...] | list[float | None], ch: int = 0):
        """Set length and angles of boom (if allowed) and ensure consistency with other booms.
        This is called from the general setter function after the units and range are checked
        and before the variable value itself is changed within the model.

        Note: boom_setter initiates an internal change in the crane, which then affects ._rot, .direction, .length
           External changes (parent booms) do not affect .boom.

        Args:
            val (array-like): new value of boom. Elements of the array can be set to None (keep value)
            ch (int) = 0: track change type or set it initially to force a type of change,
        """
        assert hasattr(self, "boom"), f"self.boom of {self.name} not yet initialized. Unexpected!"
        for i, v in enumerate(val):
            if v is not None and not isnan(v) and v != self.boom[i]:
                if i == 0 and Change.POS not in Change(ch):
                    ch += Change.POS.value
                elif Change.ROT not in Change(ch):
                    ch += Change.ROT.value
                if i == 0 and self.q_factor != 0 and abs(self.direction[2]) < 1.0 - 1e-10:
                    self._new_len = v  # pendulum length change while swinging. Defer to .calc_statics_dynamics()
                else:
                    self.boom[i] = v
        if Change.ROT in Change(ch):  # direction change
            assert self.q_factor == 0, "Attempt to directly set the angle of a wire. Does not make sense"
            if self.anchor0 is None:
                self._rot = self.model.rot()
            else:
                self._rot = self.anchor0.rot() * rot_from_spherical(self.boom[1:])
            self.direction = self._rot.apply(np.array((0, 0, 1), float))
        self.c_m  # noqa: B018  ## (not useless!) updates the local center-of-mass
        if self.anchor1 is not None:
            self.anchor1.update_child(change=Change(ch))
        return self.boom

    @property
    def model(self):
        """Get the Crane object which this boom is connected to."""
        return self._model

    @property
    def name(self):
        """Get the unique (within the crane) name of the boom."""
        return self._name

    @property
    def length(self) -> np.floating:
        """Extract the length parameter from .boom."""
        return self.boom[0]

    @property
    def end(self) -> TVector:
        """Calculate the cartesian coordinates of the end of the boom from .origin, .length and .direction."""
        return self.origin + self.length * self.direction

    @property
    def c_m(self) -> TVector:
        """Updates and returns the local center of mass point relative to self.origin."""
        self._c_m = self.mass_center[0] * self.length * self.direction + (
            np.array((self.mass_center[1], self.mass_center[2], 0), float)
        )
        return self._c_m

    @property
    def c_m_sub(self):
        """Return the system center of mass as absolute position
        The _c_m_sub needs to be calculated/updated using calc_statics_dynamics.
        """
        return self._c_m_sub

    def damping(self, q_factor: float | None = None, damping_time: float | None = None):
        """Change/set the damping properties of a flexible boom.
        Ensure that _damping_time and _new_len is set correctly.
        """
        if self.q_factor != 0.0:
            assert self.anchor0 is not None, "Flexible first booms are so far not implemented"
            # pre-calculated loss term used in .pendulum()
            length = self.length if self._new_len is None else (self.length + self._new_len) / 2
            if q_factor is not None:
                self.q_factor = q_factor
                self._damping_time = sqrt(length / 9.81 * (self.q_factor**2 + 0.25))
            elif damping_time is not None:  # new damping time. Change q_factor
                self.q_factor = sqrt(damping_time**2 * 9.81 / length - 0.25)
                self._damping_time = damping_time
        else:
            self._damping_time = 0.0
        return self._damping_time

    def update_child(self, change: Change = Change.ALL):
        """Update this boom after the parent boom has changed length, angles, position or rotation.

        change(Change) = Change.ALL: Possibility to specify which type of change is performed. Default: ALL
        """
        if self.q_factor > 0.0:  # flexible boom. Angle or origin change. c_m stays about fixed
            return  # update of pendulum is deferred to calc_statics_dynamics, since step time is needed

        if self.anchor0 is not None:  # does not apply to fixation
            self.origin = self.anchor0.end
            # if self.name=='wire': print(f"Wire@{change}: {self.origin}->{self.end}, angle {self.boom[1]}, {self.model.euler}")
            if Change.ROT in change:
                self._rot = self.anchor0.rot() * rot_from_spherical(self.boom)
        self.direction = self._rot.apply(np.array((0, 0, 1), float))
        logger.debug(f"New direction {self.name}, dir:{self.direction}, origin:{self.origin}, end:{self.end}")

        if self.anchor1 is not None:
            self.anchor1.update_child(change)

    @property
    def torque(self):
        """Calculate the torque from the CoM of this boom with respect to the fixation.
        Note that this is in practice only calculated for the load and the rest of the crane.
        These two are calculated separately, since the load exhibits a pendulum action.
        The Torque is always calculated relative to the crane position.
        """
        _p_c_m = self._c_m_sub[1] - self.model.position
        if self.q_factor > 0:  # pendulum action
            return self.mass * np.cross(
                _p_c_m,
                (
                    np.array((0, 0, -9.81), float)
                    + (self.model.velocity + self.r_v) ** 2 * self.direction
                    + self.model.d_velocity
                ),
            )
        else:  # other fixed boom. Calculate including all children, but load. Assume self._c_m_sub updated
            m = self._c_m_sub[0]
            # print(f"Torque {self.name}: m:{m}, pos:{_p_c_m}, c_m_sub:{self.c_m_sub[1]}, acc:{self.model.d_velocity}")
            return m * np.cross(_p_c_m, (np.array((0, 0, -9.81), float) + self.model.d_velocity))

    @property
    def force(self):
        """Calculate the force resulting from linear acceleration."""
        return np.dot(self._c_m_sub[0], -self.model.d_velocity + np.array((0, 0, -9.81), float))

    def calc_statics_dynamics(self, dt: float | None = None):
        """After any movement the local c_m and the c_m of connected booms have changed.
        Thus, after the movement has been transmitted to connected booms, the _c_m_sub of all booms can be updated in reverse order.
        The local _c_m_sub is updated by calling this function, assuming that child booms are updated.

        Since there might happen multiple movements within a time interval, the function is called explicitly, i.e. from crane.
        Note that c_m_sub includes all child booms, but the wire (as this can swing).
        For the wire, the function .update_child() is not run, so that neither .origin nor .rot or .direction is updated.

        Args:
            dt (float)=None: for dynamic calculations, the time for the movement must be provided
              and is then used to calculate torque and force
        """
        if self.q_factor != 0.0:  # non-fixed boom. Might be moving or there might be pending origin/length changes
            if dt is None:
                self._c_m_sub = (self.mass, self.origin + self.c_m)
                self._pendulum_instantaneous()
            else:
                self.pendulum(dt)
        else:
            if self.anchor1 is None or self.anchor1.anchor1 is None:  # there are no attached booms or attached wire
                # assuming that _c_m is updated. Note that _c_m_sub is a global vector
                self._c_m_sub = (self.mass, self.origin + self.c_m)  #! use .c_m to ensure update!
            else:  # there are attached boom(s)
                # this should be updated if calc_statics_dynamics has been run for attached boom
                [mS, posS] = self.anchor1.c_m_sub
                m = self.mass
                cs = self.origin + self.c_m  # the local center of mass as absolute position. .c_m to ensure update!
                # updated _c_m_sub as absolute position
                self._c_m_sub = (mS + m, (cs * m + mS * posS) / (mS + m))

        if self.anchor0 is not None:
            self.anchor0.calc_statics_dynamics(dt)

    def _pendulum_instantaneous(self):
        """Move the pendulum instantaneously - origin or length changes are instantaneous and
        the load stays as much as possible where it was.

        Note: a new length may be stored as self._new_len and the origin might need updating to .anchor0.end
        """
        assert self.anchor0 is not None, "Function cannot be called on fixation."
        if self._new_len is not None:  # a new length has been stored, but is not affectuated
            self.boom[0] = self._new_len  #! we assume that the direction does not change
            self._new_len = None
        clen0 = self.boom[0] * self.mass_center[0]  # distance from origin to the mass center (constant)
        cm0 = self.origin + clen0 * self.direction  # center-of-mass before movement
        origin1 = self.anchor0.end
        to_cm0 = cm0 - origin1  # vector from new origin to cm0
        clen1 = np.linalg.norm(to_cm0)
        if clen0 > clen1:  # cm falling
            cm1 = cm0 + np.array((0, 0, -1), float) * (-to_cm0[2] + np.sqrt(clen0**2 - to_cm0[0] ** 2 - to_cm0[1] ** 2))
            end1 = origin1 + self.boom[0] * normalized(cm1 - origin1)
        else:  # elif clen0 < clen1: # cm dragged in clen1 direction
            end1 = origin1 + self.boom[0] * normalized(to_cm0)
        self.direction = normalized(end1 - origin1)
        rel_direction = self.anchor0.rot().apply(self.direction, inverse=True)  # dir. relative to previous boom
        self.boom[1:] = cartesian_to_spherical(rel_direction)[1:]
        self.origin = self.anchor0.end
        self._rot = self.anchor0.rot() * rot_from_spherical(self.boom)

    @staticmethod
    def _energy(mass: float, c_m: np.ndarray, speed: np.ndarray):
        """Calculate the current energy in the pendulum. Should not be used for other booms.

        Implemented as staticmethod to not get many duplicates of the function made.

        Args:
            mass (float): mass of the boom (load)
            c_m (float): vector from origin to center of mass
            speed: the current speed of the c_m
        """
        return mass * (
            9.81 * (np.linalg.norm(c_m) - np.dot(c_m, np.array((0, 0, -1), float))) + 0.5 * np.dot(speed, speed)
        )

    @staticmethod
    def _angular_momentum(mass: float, c_m: np.ndarray, speed: np.ndarray):
        """Calculate the current energy in the pendulum. Should not be used for other booms.

        Implemented as staticmethod to not get many duplicates of the function made.

        Args:
            mass (float): mass of the boom (load)
            c_m (float): vector from origin to center of mass
            speed: the current speed of the c_m
        """
        return mass * np.cross(c_m, speed)

    def pendulum(self, dt: float):
        r"""For a non-stiff connection, if the _c_m is not exactly below origin, the _c_m acts as a damped pendulum.

        More detailed information on Spherical pendulum and Q-factor:
        * `Sphärisches Pendel <https://de.wikipedia.org/wiki/Sph%C3%A4risches_Pendel>`_
        (the English article is not as good) for detailed formulas
        * `q_factor: <https://en.wikipedia.org/wiki/q_factor>`_

        .. limitation:: Falling movements (i.e. wire acceleration larger than g) are not allowed (raise error).

        .. assumption:: The center of mass is on the boom line at mass_center[0] relative distance from origin.

        .. limitation:: Damping implemented as reduction on speed, which is accurate only over whole period,
          since potential enery is not included.
          Note also that damping_time is defined as damping on energy, not on amplitude!

        The following differential equation is used for pendulum movement

        .. math::

            \ddot{\vec{r}} = -\frac{\vec{r} \times (\vec{r} \times \vec{g})}{R^2} - \frac{\dot{\vec{r}}^2}{R^2} \vec{r}

        covering general spherical pendulum movement. For the crane we get the additional complications,
        that the pendulum origin may be moving (because the parent boom is moving)
        or that the length of the pendulum may be changing. To accomodate for that within a time interval :math:`\tau t`
        the origin is initially left unchanged and is over the time interval :math:`\tau t`
        moved to the target position (end point of the parent boom).
        The same strategy is applied to the length of the wire,
        adapting it to the target length over the time interval :math:`\tau t`.


        Pendulum movements are integrated into `.calc_statics_dynamics(dt)` if the connection is non-stiff (q_factor!=0).

        Args:
            dt (float): The time interval for which the pendulum movement is calculated

        Returns
        -------
            updated velocity and acceleration (of c_m)

        """

        def ivp_fun(
            t: float, y: np.ndarray, r2: float, g: np.ndarray, dr_dt: np.ndarray | None, l0: float, dl_dt: float | None
        ):
            """Solve the initial value problem of the pendulum dr/dt = f(t,r), r(t=0) = r0. Without losses.

            Args:
                t (float): the independent variable (time)
                y (np.ndarray): the vector of the 3D position and the 3D speed of the COM of the pendulum,
                  relative to origin
                r2 (float): the squared pendulum radius with respect to COM
                g (float): gravitational acceleration as ndarray
                dr_dt (ndarray): Optional movement of the origin through dt
                l0 (float): start length of wire. Used only if wire length changes
                dl_dt (float): Optional change of wire length through dt: l(t) = l0 + dl_dt* t
            """
            r = y[:3]
            v = y[3:] if dr_dt is None else y[3:] - dr_dt
            if dl_dt is not None:
                l0 += dl_dt * t
                r2 = l0 * l0
            tangential = np.cross(r, np.cross(r, g))
            centripetal = np.dot(v, v) * r
            self.r_acc = -((tangential + centripetal) / r2)
            return np.append(v, self.r_acc)

        assert self.anchor0 is not None, f"pendulum() called on fixation {self.name}"
        assert self.q_factor != 0.0, f"pendulum() called on fixed boom {self.name}"
        if (
            (
                abs(self.direction[2]) < 1.0 - 1e-10  # pendulum moving or not updated
                or np.linalg.norm(self.r_v[:2]) > 1e-10
                or np.linalg.norm(self.r_acc[:2]) > 1e-10
                or not np.allclose(self.origin, self.anchor0.end)
                or self._new_len is not None
            )
            and self.mass_center[0] * self.length > 1e-10  # pendulum length not too short
        ):
            R = self.mass_center[0] * self.length  # pendulum radius wrt. center of mass
            if self.additional_checks:
                e0 = Boom._energy(self.mass, self._c_m, self.r_v)
                lz0 = Boom._angular_momentum(self.mass, self._c_m, self.r_v)
            if np.allclose(self.origin, self.anchor0.end):
                dr_dt = None
            else:
                dr_dt = (self.anchor0.end - self.origin) / dt  # translation in time interval
            if self._new_len is None:
                dl_dt = None
            else:
                dl_dt = (self._new_len - self.length) / dt
                self.damping(q_factor=self.q_factor)  # re-calculate due to new length
                self.boom[0] = self._new_len
                self._new_len = None

            if R > 1e-6:
                r2 = R * R
                g = np.array((0.0, 0.0, -9.81), float)
                r = R * self.direction
                v = self.r_v
                sol = solve_ivp(  # type: ignore
                    ivp_fun,
                    t_span=[0, dt],
                    y0=np.append(r, v),
                    method="RK45",  #'RK23' 'DOP853' 'Radau' 'LSODA' "BDF" "RK45"
                    args=(r2, g, dr_dt, R, dl_dt),  # type: ignore[reportArgumentType]  ## according to spec
                    atol=self.tolerance,
                )
                if sol.status != 0:
                    logger.error(f"r:{r}, v:{v}, r2:{r2}, dr/dt:{dr_dt}, l0:{R}, dl_dt:{dl_dt}")
                    raise AssertionError(f"Pendulum solver did not succeed with dt:{dt}. Status:{sol.status}") from None
                y_t = sol.y[:, -1]  # last column
                position = y_t[:3]  # if dr_dt is None else y_t[:3] + dr_dt * dt  # position of CoM relative to origin
                self.r_v = y_t[3:]  # if dr_dt is None else y_t[3:] + dr_dt
                if dt >= self._damping_time:  # pendulum stops within dt
                    self.r_v = np.array((0, 0, 0), float)
                else:
                    self.r_v *= 1 - dt / self._damping_time  # see note

                self.direction = normalized(position)
                rel_direction = self.anchor0.rot().apply(self.direction, inverse=True)  # dir. relative to previous boom
                self.boom[1:] = cartesian_to_spherical(rel_direction)[1:]
                self._c_m = position
            self.origin = self.anchor0.end
            self._c_m_sub = (self.mass, self.origin + self.c_m)

            if self.model.current_time > 0.0 and np.dot(self.r_acc, self._c_m) > 1e-10:
                logger.warning(
                    f"Free fall of load detected at time {self.model.current_time}. Acc.:{np.dot(self.r_acc, self._c_m)}"
                )
            if self.additional_checks and dr_dt is None and dl_dt is None and self.model.current_time > 0.0:
                e1 = Boom._energy(self.mass, self._c_m, self.r_v)
                if e1 - e0 > 1e-7:
                    logger.error(f"Pendulum energy change @{self.model.current_time} {e0}->{e1}")
                    logger.error(f"   c_m:{self._c_m}, speed:{self.r_v}, damping:{self._damping_time}")
                    logger.error(
                        f"   tangential:{np.cross(self._c_m, np.cross(self._c_m, g))}, centripetal:{np.dot(self.r_v, self.r_v) * self._c_m}"
                    )

                lz1 = Boom._angular_momentum(self.mass, self._c_m, self.r_v)
                if abs(lz1[2]) - abs(lz0[2]) > 1e-7:  # not np.allclose( lz0, lz1):
                    logger.error(f"Pendulum angular momentum increase @{self.model.current_time} {lz0[2]}->{lz1[2]}")
                    logger.error(f"   c_m:{self._c_m}, speed:{self.r_v}, damping:{self._damping_time}")
            # print(f"origin:{self.origin[0]}, d_end:{self.end[0]-self.origin[0]}, speed:{self.r_v[0]}")

    def pendulum_relax(self):
        """Relax the pendulum movement, leading to the CoM strictly downward.
        Since no time is used this action is strictly unphysical and should only be used for testing purposes.
        """
        if (
            self.q_factor != 0.0  # non-fixed connection to previous boom (wire)
            and abs(self.direction[2]) > 1e-10  # pendulum not relaxed
            and self.mass_center[0] * self.length > 1e-10  # pendulum length not too short
        ):
            assert self.anchor0 is not None, "anchor0 needed at this point"
            self.model.calc_statics_dynamics()
            self.direction = np.array((0, 0, -1), float)
            self.r_v = np.array((0, 0, 0), float)
            self.r_acc = np.array((0, 0, 0), float)
            rel_direction = self.anchor0.rot().apply(self.direction, inverse=True)  # dir. relative to previous boom
            self.boom[1:] = cartesian_to_spherical(rel_direction)[1:]
            crane_dir = normalized(self.model.rot().apply((0, 0, 1)))
            self._rot = rot_from_vectors(crane_dir, self.direction)
            self.model.calc_statics_dynamics()

    def _calc_decayrate(self, newLength: float) -> float:
        if self.q_factor == 0.0:
            return nan
        elif newLength == 0.0:
            return 0
        else:
            return sqrt(9.81 / (newLength * self.mass_center[0])) / sqrt(4 * self.q_factor - 1)
