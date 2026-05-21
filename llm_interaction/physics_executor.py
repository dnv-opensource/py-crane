import logging
import numpy as np
from functools import partial
from llm_interaction.llm_command import Command, CommandType
from component_model.utils.controls import Control, Controls
from src.py_crane.mobile_crane import MobileCrane


class PhysicsExecutor:
    def __init__(self, crane: MobileCrane):
        self.crane = crane
        self.booms = list(crane.booms())

        self.pedestal = self.booms[1]
        self.boom = self.booms[2]
        self.rope = self.booms[3]

        self.controls = Controls(limit_err=logging.WARNING)
        self._setup_controls()

        self.command_state = {}
        self.rotation_tracker = {}  

    # ------------------------------------------------------------------

    def _setup_controls(self):
        def ped_az(val=None):
            if val is not None:
                self.pedestal.boom_setter([None, None, val])
            return self.pedestal.boom[2]

        def boom_len(val=None):
            if val is not None:
                self.boom.boom_setter([val, None, None])
            return self.boom.boom[0]

        def boom_pol(val=None):
            if val is not None:
                self.boom.boom_setter([None, val, None])
            return self.boom.boom[1]

        def wire_len(val=None):
            if val is not None:
                self.rope.boom_setter([val, None, None])
            return self.rope.boom[0]

        self.controls.extend(
            [
                Control(
                    "pedestal_azimuth",
                    limits=(None, (-0.5, 0.5), (-0.2, 0.2)),  
                    rw=partial(ped_az),
                ),
                Control(
                    "boom_length",
                    limits=((20, 50), (-5, 5), (-1, 1)),
                    rw=partial(boom_len),
                ),
                Control(
                    "boom_polar",
                    limits=((0, np.pi), (-0.5, 0.5), (-0.2, 0.2)),
                    rw=partial(boom_pol),
                ),
                Control(
                    "wire_length",
                    limits=((0.5, 50), (-1, 1), (-0.5, 0.5)),
                    rw=partial(wire_len),
                ),
            ]
        )

    # ------------------------------------------------------------------

    def execute_command(self, command: Command, t: float, dt: float):
        cid = id(command)

        if cid not in self.command_state:
            self.command_state[cid] = {"started": False, "finished": False}

        state = self.command_state[cid]

        t0 = command.start_time
        t1 = t0 + command.duration if command.duration else None

        # ---------------- start ----------------
        if not state["started"]:
            if abs(t - t0) < 1e-6:
                state["started"] = True

                if command.type == CommandType.BOOM_ROTATE:
                    self._start_rotation(command, cid)

                elif command.type == CommandType.BOOM_LUFF:
                    print(
                        f"[CMD_RECEIVED] type={command.type.value}, angle={command.angle}, ang_vel={command.angular_velocity}, duration={command.duration}"
                    )
                    self._start_luffing(command, cid)

                elif command.type == CommandType.BOOM_EXTEND:
                    self._start_extension(command, cid)
            return

        # ---------------- update rotation angle ----------------
        if state["started"] and not state["finished"]:
            if command.type == CommandType.BOOM_ROTATE:
                self._update_rotation(command, cid)
            elif command.type == CommandType.BOOM_LUFF:
                self._update_luffing(command, cid)
            elif command.type == CommandType.BOOM_EXTEND:
                self._update_extension(command, cid)

        # ---------------- finish ----------------
        if state["started"] and not state["finished"]:
            if t1 is not None and t > t1 + 1e-6:
                state["finished"] = True

                if command.type == CommandType.BOOM_ROTATE:
                    self.controls["pedestal_azimuth"].setgoal(1, None)
                elif command.type == CommandType.BOOM_LUFF:
                    self.controls["boom_polar"].setgoal(1, None)
                elif command.type == CommandType.BOOM_EXTEND:
                    self.controls["boom_length"].setgoal(1, None)

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _start_rotation(self, cmd: Command, cid):
        start_angle = self.pedestal.boom[2]

        self.rotation_tracker[cid] = {
            "start": start_angle,
            "target": start_angle + cmd.angle,
        }

        self.controls["pedestal_azimuth"].setgoal(1, cmd.angular_velocity)

    def _update_rotation(self, cmd: Command, cid):
        current = self.pedestal.boom[2]
        target = self.rotation_tracker[cid]["target"]

        if cmd.angular_velocity > 0:
            if current >= target:
                self.controls["pedestal_azimuth"].setgoal(1, None)
        else:
            if current <= target:
                self.controls["pedestal_azimuth"].setgoal(1, None)

    # ------------------------------------------------------------------
    #  Luffing
    # ------------------------------------------------------------------

    def _start_luffing(self, cmd: Command, cid):
        current = self.boom.boom[1]

        # Calculate target angle (INVERTED because boom[1] increases means moving DOWN)
        # boom[1]=0 is straight up, boom[1]=π is straight down
        # So: positive angle_delta (upward) means DECREASE boom[1]
        target = current - cmd.angle

        # Check if clamping will occur
        unclamped_target = target
        target = max(0.0, min(np.pi, target))

        clamp_msg = ""
        if unclamped_target != target:
            clamp_msg = f" [CLAMPED from {unclamped_target:.4f}]"

        print(
            f"[LUFF_START] angle_delta={cmd.angle:.4f}, ang_vel={cmd.angular_velocity:.4f}, current_boom[1]={current:.4f}, target={target:.4f}{clamp_msg}"
        )

        # Track target like rotation does
        self.rotation_tracker[cid] = {
            "start": current,
            "target": target,
        }

        # Use velocity mode to avoid exceeding speed limits
        # INVERT velocity because we inverted the angle
        self.controls["boom_polar"].setgoal(1, -cmd.angular_velocity)
        print(f"[LUFF_SETGOAL] Set velocity goal to {-cmd.angular_velocity:.6f}")

    def _update_luffing(self, cmd: Command, cid):
        current = self.boom.boom[1]
        target = self.rotation_tracker[cid]["target"]
        tolerance = 1e-4  

        # Debug: Check if we're actually moving
        if abs(cmd.angular_velocity) > 1e-6:  # Only print if velocity is being used
            print(
                f"[LUFF_UPDATE] current={current:.6f}, target={target:.4f}, vel={cmd.angular_velocity:.6f}, dist={current - target:.6f}"
            )

        # Check if target reached (remember: velocity is inverted from cmd.angular_velocity)
        # So when cmd.angular_velocity > 0 (upward), actual velocity is negative
        if cmd.angular_velocity > 0:
            # Upward: actual velocity is negative, so current should decrease
            if current <= target + tolerance:
                print(f"[LUFF_DONE] target={target:.4f}, current={current:.4f}, finished")
                self.controls["boom_polar"].setgoal(1, None)
        else:
            # Downward: actual velocity is positive, so current should increase
            if current >= target - tolerance:
                print(f"[LUFF_DONE] target={target:.4f}, current={current:.4f}, finished")
                self.controls["boom_polar"].setgoal(1, None)

    # ------------------------------------------------------------------
    # EXTEND
    # ------------------------------------------------------------------

    def _start_extension(self, cmd: Command, cid):
        current = self.boom.boom[0]

        target = current + cmd.velocity * cmd.duration

        # clamp
        # unclamped = target
        target = max(20.0, min(50.0, target))

        print(f"[EXTEND_START] current={current:.2f}, target={target:.2f}")

        self.rotation_tracker[cid] = {
            "start": current,
            "target": target,
        }

        self.controls["boom_length"].setgoal(1, cmd.velocity)

    def _update_extension(self, cmd: Command, cid):
        current = self.boom.boom[0]
        target = self.rotation_tracker[cid]["target"]

        tol = 1e-4

        print(f"[EXTEND_UPDATE] current={current:.4f}, target={target:.4f}")

        if cmd.velocity > 0:
            if current >= target - tol:
                print("[EXTEND_DONE]")
                self.controls["boom_length"].setgoal(1, None)
        else:
            if current <= target + tol:
                print("[EXTEND_DONE]")
                self.controls["boom_length"].setgoal(1, None)

    # ------------------------------------------------------------------

    def step(self, t: float, dt: float):

        self.controls.step(t, dt)
