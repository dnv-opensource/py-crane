from __future__ import annotations

import logging
from functools import partial
from typing import Any, Never, Sequence, cast

import numpy as np

from py_crane.boom import Boom, Wire
from py_crane.crane import Crane
from py_crane.enums import Change

logger = logging.getLogger(__name__)


class BoomFMU(Boom):
    """The FMU definitions of a :py:class:`.Boom` -
    a stiff boom, attached to a parent boom, attached to the ``fixation`` boom of a crane.

    Defines the `FMI interface <https://fmi-standard.org/>`_ of a :py:class:`.Boom`
    using the ``component_model.variable`` module.
    :py:class:`BoomFMU` should be instantiated through :py:meth:`.CraneFMU.add_boom`,
    omitting the ``model`` and ``anchor0`` arguments (these are added automatically).

    Here we only list and explain additional arguments and attributes of :py:class:`BoomFMU`.
    Refer to :py:class:`.Boom` for details on basic Boom arguments, which are also needed here.

    For every Boom the following interface variables are defined

    mass
        The mass of the boom. If ``mass_range`` is not defined the variable is a fixed parameter,
        otherwise a continuous input variable (the load).
    boom
        The length, polar angle and azimuth angle of the boom.
        The ``boom_rng`` defines the degrees of freedom for the boom (see below).
    end
        The cartesian position of the end point of the boom as read only (output) variable.
        Continuous change to the boom (length, polar-rotation, azimuthal-rotation) wrt. origin
    der(<name>.boom) (not defined for the fixation)
       This is an example of a structured variable facilitating continuous (not abrupt) change the boom parameters.
    der(der(<name>.boom)) (not defined for fixation
        Acceleration to the boom (length, polar-rotation, azimuthal-rotation) wrt. origin
        Another structured variable as aid for boom control.
    der(<name>.mass) (only if `mass_rng is not None`)
        Continuous change to the mass (i.e. load change) if abrupt load changes are not desirable.

    Args:
        mass_rng (tuple): Optional range of the mass, if the mass can be changed.
            Normally only the last boom (the wire) has a variable mass (the load).
        boom_rng (tuple): Range for each of the boom components,
            i.e. how much the boom length can be changed and how (much) it can be rotated
            As normal, range elements specified as `None` denote fixed components and `()` freely moveable elements.
            E.g. ( ('1m','50m'), None, ()) denotes a boom which length can be changed in the range 1 to 50 meters,
            which polar angle is fixed to the initial value and which can be freely rotated around the z-axis.

    .. todo:: determine the range of forces
    """

    def __init__(
        self,
        model: Crane,
        name: str,
        /,
        mass_rng: tuple[str, str] | None = None,
        boom_rng: tuple[tuple[Any, Any] | None | Sequence[Never], ...] = tuple(),
        **kwargs: Any,
    ):
        from py_crane.crane_fmu import CraneFMU

        # we need 'early access' to some of the properties
        assert isinstance(model, CraneFMU), f"BoomFMU must link to a CraneFMU. Found {type(model)}"
        self._name = name
        model.ensure_boom(self)  # ensure that the boom object is registered with the crane
        u_time = model.u_time
        u_length = model.u_length
        u_angle = "deg" if model.degrees else "rad"
        # make some super-arguments 'visible' direct for usage here
        mass = kwargs.pop("mass") if "mass" in kwargs else "1.0kg"
        boom = kwargs.pop("boom") if "boom" in kwargs else (1.0, 0.0, 0.0)
        q_factor = kwargs.pop("q_factor") if "q_factor" in kwargs else 0.0

        # Interface specifications. When we have the start values we can instantiate the Boom
        _c, _v = ("parameter", "fixed") if mass_rng is None else ("input", "continuous")
        self._mass = model.add_variable(
            f"{self._name}.mass",
            description=f"Mass of boom {self._name}",
            causality=_c,
            variability=_v,
            start=mass,
            rng=mass_rng,
        )
        if not len(self._mass.unit):
            logger.warning(f"Warning: Missing unit for mass of boom {self._name}. Include in the 'mass' parameter")
        mass0 = self._mass.getter()[0]
        assert isinstance(mass0, float)

        assert isinstance(boom, (tuple, list, np.ndarray)), f"boom {self.name} invalid 3D start value. Found {boom}"
        _boom: list[float | str] = list(boom)  # make it changeable
        if _boom[0] == 0:
            _boom[0] = f"0 {u_length}"
        for i in range(1, 3):  # the two spherical angles
            if _boom[i] == 0:
                _boom[i] = f"0{u_angle}"
            elif not isinstance(_boom[i], str) or u_angle not in cast(str, _boom[i]):
                logger.error(f"All angles shall be provided as {u_angle}")
                _boom[i] = f"{_boom[i]}{u_angle}"
            assert (
                boom_rng is None
                or boom_rng[i] is None
                or not len(boom_rng[i])  # type: ignore[arg-type]  ## should have a length at this point
                or (boom_rng[i][0] > float("-inf") and boom_rng[i][1] < float("inf"))  # type: ignore[index]
            ), f"The range of {self.name}[{i}] should not be limited, as radian variables are periodic"
        self._boom = model.add_variable(
            f"{self._name}.boom",
            description=f"Length [m] and direction [rad] of {self._name} from anchor point in spherical coordinates",
            causality="input",
            variability="continuous",
            start=_boom,
            rng=boom_rng,
            on_set=partial(self.boom_setter, ch=Change.ROT.value if q_factor == 0 else 0),
        )
        assert isinstance(self._boom.owner, BoomFMU), f"Owner of variable {self._boom}: {self._boom.owner}"
        if q_factor == 0.0:
            Boom.__init__(
                self,
                model,
                name,
                mass=mass0,
                boom=getattr(
                    self._boom.owner, self._boom.local_name
                ),  #! not getter()! boom.py uses internal variables!
                q_factor=0.0,
                **kwargs,
            )
        else:
            Wire.__init__(
                self,  # type: ignore[arg-type]  ## call sequence ensures that it is called from Wire
                model,
                name,
                mass=mass0,
                boom=getattr(
                    self._boom.owner, self._boom.local_name
                ),  #! not getter()! boom.py uses internal variables!
                q_factor=q_factor,
                **kwargs,
            )
        # additional output variables
        self._end = model.add_variable(  # pyright: ignore[reportUnknownMemberType]  # should become obsolete once component_model is updated.
            f"{self._name}.end",
            description="Cartesian vector of the end of the boom",
            causality="output",
            variability="continuous",
            start=self.end,
        )
        # additional derivative variables (but not for fixation, as these are Euler movements on the crane!)
        if self.name != "fixation":
            self._der1_boom = model.add_variable(  # pyright: ignore[reportUnknownMemberType]  # should become obsolete once component_model is updated.
                f"der({self._name}.boom)",
                description="Continuous change to the boom (length, polar-rotation, azimuthal-rotation) wrt. origin",
                causality="input",
                variability="continuous",
                start=(f"0 m/{u_time}", f"0 {u_angle}/{u_time}", f"0 {u_angle}/{u_time}"),
            )
            self._der2_boom = model.add_variable(  # pyright: ignore[reportUnknownMemberType]  # should become obsolete once component_model is updated.
                f"der(der({self._name}.boom))",
                description="Acceleration to the boom (length, polar-rotation, azimuthal-rotation) wrt. origin",
                causality="input",
                variability="continuous",
                start=(f"0 m/{u_time}**2", f"0 {u_angle}/{u_time}**2", f"0 {u_angle}/{u_time}**2"),
            )
            if self._mass.range[0].rng[0] != self._mass.range[0].rng[1]:  # mass is changeable (normally the load)
                self._der1_mass = model.add_variable(  # pyright: ignore[reportUnknownMemberType]  # should become obsolete once component_model is updated.
                    f"der({self._name}.mass)",
                    description="Continuous change to the mass (i.e. load change)",
                    causality="input",
                    variability="continuous",
                    start=f"0 kg/{u_time}",
                )


class WireFMU(BoomFMU, Wire):
    """The FMU definitions of a :py:class:`.Wire` -
    a special stiff boom, attached with a flexible joint to a parent boom -
    normally the last boom of a crane.
    """

    def __init__(
        self,
        model: Crane,
        name: str,
        /,
        mass_rng: tuple[str, str] | None = None,
        boom_rng: tuple[tuple[Any, Any] | None | Sequence[Never], ...] = tuple(),
        **kwargs: Any,
    ):
        super().__init__(model, name, mass_rng, boom_rng, **kwargs)
