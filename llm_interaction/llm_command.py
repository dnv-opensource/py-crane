from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CommandType(Enum):
    """LLM command types."""

    EXTERNAL_FORCE = "external_force"  # Apply external force
    WIRE_INIT_VELOCITY = "wire_init_velocity"  # Set initial velocity to wire
    BOOM_ROTATE = "boom_rotate"  # Rotate boom (pedestal azimuth)
    BOOM_EXTEND = "boom_extend"  # Extend/retract boom (boom length)
    BOOM_LUFF = "boom_luff"  # Luff boom (boom polar angle)


@dataclass
class Command:
    """Command object parsed from LLM."""

    type: CommandType
    target: str  # "wire", "boom", "pedestal"
    start_time: float  # seconds
    duration: Optional[float] = None  # seconds, None for instantaneous

    # Specific parameters
    force: Optional[list] = None  # External force [fx, fy, fz] (Newton)
    velocity: Optional[list] = None  # Velocity [vx, vy, vz] (m/s)
    angle: Optional[float] = None  # Angle (radians) - relative change
    angular_velocity: Optional[float] = None  # Angular velocity (rad/s) - relative

    # Physical constraint parameters
    angular_acceleration: Optional[float] = None  #  (rad/s²)
    linear_acceleration: Optional[float] = None  #  (m/s²)

    def __repr__(self):
        return f"Command({self.type.value}, target={self.target}, start={self.start_time}s)"
