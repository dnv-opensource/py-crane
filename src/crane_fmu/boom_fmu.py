from __future__ import annotations

import logging

import numpy as np
from component_model.model import Model

from crane_fmu.boom import Boom

# from crane_fmu.crane import Crane

logger = logging.getLogger(__name__)


class BoomFMU(Boom):
    """Boom object, representing one element of a crane,
    modelled as stiff rod with length, mass, mass-center and given degrees of freedom.

    Alternative to instantiating a Boom by calling its `__init__()` ,
    the `add_boom()` method of the crane can be called.
    In this case the `model` and `anchor0` parameters (see below) shall not be included (are automatically added).

    Basic boom movements are

    * Rotation: active rotation around the hinge, obeying the defined degrees of freedom, or passive initiated by parent booms (around their hinge)
    * Translation: the boom is moved linearly, due to linear movement of the hinge
    * length change: length change of booms (e.g. wire) can be one defined degree of freedom. It is similar to translation, but the hinge does not move, only the end.
    * mass change: mass change of booms (e.g. add or remove load) can be one defined degree of freedom.

    After any movement

    * the internal variables (center of mass, positions, etc.) are re-calculated,
    * the attached booms are moved in accordance
    * and the center of mass of the sub-system (the boom with attached booms) is re-calculated.
    * Finally the parent booms are informed about the change of the sub-system center of mass, leading to re-calculation of their internal variables.

    .. note:: initialization variables are designed for easy management of parameters,
       while derived internal variables are more complex (often 3D vectors)

    Args:
        model (Model): The model object owning the boom.
        name (str): The short name of the boom (unique within FMU)
        description (str) = '':  An optional description of the boom
        anchor0 (Boom): the boom object to which this Boom is attached.
            There is exactly one boom where this is set to None, which is the crane `fixation`
            and which is automatically provided by the crane, i.e. for real booms this is never None.
        mass (float): Parameter denoting the (assumed fixed) mass of the boom
        mass_rng (tuple): Optional range of the mass, if the mass can be changed.
            Normally only the last boom (the wire) has a variable mass (the load).
        mass_center (float,tuple): Parameter denoting the (assumed fixed) position of the center of mass of the boom,
            provided as portion of the length (as float).
            Optionally the absolute displacements in x- and y-direction (assuming the boom in z-direction) can be added
            e.g. (0.5,'-0.5 m','1m'): halfway down the boom displaced 0.5m in the -x direction and 1m in the y direction
        boom (tuple): A tuple defining the boom relative to its parent in spherical (ISO 80000) coordinates.
            From the parent boom the following attributes are automatically inferred:

            * origin: end of the parent boom => cartesian origin
            * pole axis: direction vector of the parent boom => local cartesian z-axis
            * reference direction in equator plane: crane x-direction or azimuth angle of connecting boom => local cartesian x-axis

            The boom is then defined in local polar coordinates:

            * length: the length of the boom (in length units)
            * polar: a rotation angle for a rotation around the negative x-axis (away from z-axis) against the clock.
            * azimuth: a rotation angle for a rotation around the positive z-axis against the clock.

           :: note..: The boom and its range is used to keep length and local coordinate system up-to-date,
               while the active work variables are the cartesian origin, direction and length
        boom_rng (tuple): Range for each of the boom components,
            i.e. how much the boom length can be changed and how (much) it can be rotated
            As normal, range components specified as None denote fixed components.
            Most booms have only one (rotation) degree of freedom.
        q_factor (float)=0.0: optional possibility to implement a loose connection between booms.

            * if q_factor=0.0, the connection to the parent boom is stiff according to the boom angle setting
            * if 0<q_factor<=0.5, the crane boom (the wire) is implemented as a stiff rod
                with a loose connection hanging from the parent boom.

            The q_factor denotes the dimensionless quality factor (energy stored/energy lost per radian),
            which is also equal to `2*ln( amplitude/amplitude next period)`, or `pi*frequency*decayTime`
        animationLW (int)=5: Optional possibility to change the default line width when performing animations.
            E.g. the pedestal might be drawn with larger and the wire with smaller line width

    With a crane object `crane` , instantiate like:

    .. code-block:: python

       pedestal = crane.add_boom(
           name ='pedestal',
           description = "The vertical crane base, on one side fixed to the vessel and
                          on the other side the pedestal is fixed to it (can rotate around z-axis).
                          The mass should include all additional items fixed to it, like the operator's cab",
           mass = '2000.0 kg',
           mass_center = (0.5, 0,'2 m'),
           boom = ('5.0 m', 0, '0deg'),
           boom_rng = (None, (0,'360 deg'), None)
           )


    .. todo:: determine the range of forces
    .. limitation:: The mass and the mass_center setting of booms is assumed constant. With respect to wire and hook of a crane this means that basically only the mass of the hook is modelled.
    .. assumption:: Center of mass: `_c_m` is the local mass-center measured relative to origin. `_c_m_sub` is a global quantity
    """

    def __init__(
        self,
        model: Model,
        name: str,
        description: str = "",
        anchor0: Boom | None = None,
        mass: str = "1 kg",
        mass_rng: tuple | None = None,
        mass_center: float | tuple = 0.5,
        boom: tuple = (1, 0, 0),
        boom_rng: tuple = tuple(),
        q_factor: float = 0.0,
        animationLW: int = 5,
    ):
        from crane_fmu.crane_fmu import CraneFMU

        self._name = name
        assert isinstance(model, CraneFMU), f"BoomFMU must link to a CraneFMU. Found {type(model)}"
        model.ensure_boom(self)  # ensure that the boom object is registered with the crane
        u_time = model.u_time
        u_length = model.u_length
        u_angle = "deg" if model.degrees else "rad"

        # Interface specifications. When we have the start values we can instantiate the Boom
        _c, _v = ("parameter", "fixed") if mass_rng is None else ("input", "continuous")
        self._mass = model.add_variable(
            f"{name}.mass", description=f"Mass of boom {name}", causality=_c, variability=_v, start=mass, rng=mass_rng
        )
        if not len(self._mass.unit):
            logger.warning(f"Warning: Missing unit for mass of boom {self._name}. Include in the 'mass' parameter")
        mass0 = self._mass.getter()[0]
        assert isinstance(mass0, float)

        assert isinstance(boom, (tuple, list, np.ndarray)), f"boom {self.name} invalid 3D start value. Found {boom}"
        _boom = list(boom)  # make it changeable
        if _boom[0] == 0:
            _boom[0] = f"0 {u_length}"
        for i in range(1, 3):
            if _boom[i] == 0:
                _boom[i] = "0" + u_angle
            elif not isinstance(_boom[i], str) or u_angle not in _boom[i]:
                logger.error(f"All angles shall be provided as {u_angle}")
                _boom[i] = f"{_boom[i]}{u_angle}"
        for i in range(1, 2):
            assert (
                boom_rng is None
                or boom_rng[i] is None
                or not len(boom_rng[i])
                or (boom_rng[i][0] > float("-inf") and boom_rng[i][1] < float("inf"))
            ), f"The range of {self.name}[{i}] should not be limited, as radian variables are periodic"
        self._boom = model.add_variable(
            f"{name}.boom",
            description=f"Length [m] and direction [rad] of {name} from anchor point in spherical coordinates",
            causality="input",
            variability="continuous",
            start=_boom,
            rng=boom_rng,
            on_set=self.boom_setter,
        )

        super().__init__(
            model,
            name,
            description,
            anchor0,
            mass=mass0,
            mass_center=mass_center,  # this could be made an interface variable in advanced cranes
            boom=getattr(self._boom.owner, self._boom.local_name),  #! not getter()! boom.py uses internal variables!
            q_factor=q_factor,
            animationLW=animationLW,
        )
        # additional output variables
        self._end = model.add_variable(
            f"{name}.end",
            description="Cartesian vector of the end of the boom",
            causality="output",
            variability="continuous",
            start=self.end,
        )
        # additional derivative variables (but not for fixation, as these are Euler movements on the crane!)
        if self.name != "fixation":
            self._der1_boom = model.add_variable(
                f"der({name}.boom)",
                description="Continuous change to the boom (length, polar-rotation, azimuthal-rotation) wrt. origin",
                causality="input",
                variability="continuous",
                start=(f"0 m/{u_time}", f"0 {u_angle}/{u_time}", f"0 {u_angle}/{u_time}"),
            )
            self._der2_boom = model.add_variable(
                f"der(der({name}.boom))",
                description="Acceleration to the boom (length, polar-rotation, azimuthal-rotation) wrt. origin",
                causality="input",
                variability="continuous",
                start=(f"0 m/{u_time}**2", f"0 {u_angle}/{u_time}**2", f"0 {u_angle}/{u_time}**2"),
            )
            if self._mass.range[0][0] != self._mass.range[0][1]:  # mass is changeable (normally the load)
                self._der1_mass = model.add_variable(
                    f"der({name}.mass)",
                    description="Continuous change to the mass (i.e. load change)",
                    causality="input",
                    variability="continuous",
                    start=f"0 kg/{u_time}",
                )
