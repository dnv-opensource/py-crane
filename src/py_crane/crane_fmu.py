from __future__ import annotations

import logging
from typing import Any

from component_model.model import Model
from component_model.variable import Variable
from component_model.variable_naming import VariableNamingConvention

from py_crane.boom import Boom, Wire
from py_crane.boom_fmu import BoomFMU, WireFMU
from py_crane.crane import Crane

logger = logging.getLogger(__name__)


class CraneFMU(Model, Crane):
    """The FMU definition and packaging of a :py:class:`.Crane` object built from stiff booms.

    Defines the `FMI interface <https://fmi-standard.org/>`_ of a :py:class:`.Crane`
    using the ``component_model`` and the ``PythonFMU`` packages.
    The crane should first be instantiated and then the booms added, using :py:meth:`.CraneFMU.add_boom()`.

    With respect to Crane and Boom refer to the dedicated modules :py:class:`.Crane` and :py:class:`.Boom`
    where the available arguments, properties and methods are listed and explained.
    Here only the additional FMU interface definition is documented.

    :py:class:`CraneFMU` defines the attributes :py:attr:`.Crane.velocity`, :py:attr:`.Crane.d_velocity`,
    :py:attr:`.Crane.angular`, :py:attr:`.Crane.d_angular`, :py:attr:`.Crane.d2_angular`,
    :py:attr:`.Crane.force` and :py:attr:`.Crane.torque` as interface variables to make them accessible in simulations.

    Additional methods :py:meth:`.CraneFMU.add_boom`, :py:meth:`.CraneFMU.ensure_boom`,
    :py:meth:`.CraneFMU.exit_initialization_mode` and :py:meth:`.CraneFMU.do_step`
    are used automatically in the background and should not be a concern for users of the module.
    Most of the interface variables are defined in :py:class:`.BoomFMU`.

    .. note:: :py:class:`CraneFMU` is still an abstract crane.
      It needs another extension to define concrete booms and interfaces.
      See MobileCrane on how this can be done.

    Args:
        u_length (str) = "m": Definition of the common crane length units. Default: meters
        u_time (str) = 's': Definition of the common crane time units. Default: seconds
        **kwargs (Any): Arguments forwarded to :py:class:`.Crane` (and sub-class ``Model``)
            The arguments name, description, author and version should be remembered,
            since they are required by the ``Model`` sub-class.
    """

    def __init__(
        self,
        degrees: bool = False,
        u_length: str = "m",
        u_time: str = "s",
        **kwargs: Any,
    ):
        """Initialize the crane object."""
        super().__init__(**kwargs)
        Crane.__init__(self, to_crane_angle=None)
        self.degrees = degrees
        self.variable_naming = VariableNamingConvention.structured
        self.u_length = u_length
        self.u_time = u_time
        _ = Variable(
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
        _ = Variable(
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
        self._force = Variable(
            self,
            "force",
            "Crane linear 3D force on fixation in N",
            causality="output",
            variability="continuous",
            start=("0.0 N",) * 3,
        )
        self._torque = Variable(
            self,
            "torque",
            "Crane 3D torque with respect to fiation in N.m",
            causality="output",
            variability="continuous",
            start=("0.0 N.m",) * 3,
        )

    def add_boom(
        self,
        name: str,
        /,
        **kwargs: Any,
    ) -> Boom:
        """Add a boom to the crane. Overrides :py:meth:`.Crane.add_boom` to ensure that :py:class:`BoomFMU` is added.

        This method represents the recommended way to contruct a crane and then add the booms.
        The ``model`` and ``anchor0`` parameters are automatically added to the boom when it is instantiated.

        Args:
            *args: all :py:class:`Boom` positional parameters, excluding ``model`` and ``anchor0``
            **kwargs: all :py:class:`Boom` keyword parameters, excluding ``model`` and ``anchor0``
        """
        if "anchor0" not in kwargs:
            last = self.boom0[-1]
            kwargs.update({"anchor0": last})
        if kwargs.get("q_factor", 0.0) == 0.0:
            return BoomFMU(self, name, **kwargs)
        else:
            return WireFMU(self, name, **kwargs)

    def ensure_boom(self, boom: BoomFMU):
        """Ensure that the boom is registered before structured variables are added to it.
        Otherwise their owner does not exist.
        """
        if not hasattr(self, boom.name):
            setattr(self, boom.name, boom)

    def exit_initialization_mode(self):
        """Initialize the model after initial variables are set.
        It is important that the crane wire is stabilized after initial boom settings,
        otherwise it would perform wild initial movements.
        """
        self.dirty_do()  # run on_set on all dirty variables
        wire = self.boom0[-1]
        if isinstance(wire, Wire):
            wire.pendulum_relax()

    def do_step(self, current_time: float, step_size: float) -> bool:
        status = Model.do_step(self, current_time, step_size)
        Crane.do_step(self, current_time, step_size)
        return status
