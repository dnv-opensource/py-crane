"""
test_pendulum_damping.py

Physics ground-truth test for Wire pendulum damping in py_crane.

Validates that a freely swinging pendulum (zero crane acceleration) damps its
amplitude at the rate predicted by the Q-factor definition:

    t_half = 2 * Q * ln(2) / omega_n

For default parameters (L=10 m, Q=50), amplitude should halve after ~70 simulation
steps at dt=1.0 s. The buggy implementation (boom.py line 629 pre-fix) produces
half-amplitude at ~35 steps — a factor-of-2 error that this test detects.
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_crane.boom import Wire
from py_crane.crane import Crane

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
np.set_printoptions(precision=4, suppress=True)


def build_test_crane(length: float = 10.0, q_factor: float = 50.0) -> Crane:
    """Build a minimal crane with a single pendulum wire for physics testing."""
    crane = Crane()
    crane.add_boom(
        "pedestal",
        description="Fixed support",
        mass=100.0,
        boom=(length, 0.0, 0.0),
    )
    crane.add_boom(
        "wire",
        description="Pendulum wire under test",
        mass=1.0,
        mass_center=1.0,
        boom=(length, np.pi, 0.0),
        q_factor=q_factor,
    )
    crane.calc_statics_dynamics(None)
    return crane


@pytest.mark.parametrize(
    "length,q_factor",
    [
        (10.0, 50.0),  # default crane parameters
        (5.0, 30.0),  # shorter wire, lower Q
        (20.0, 80.0),  # longer wire, higher Q
    ],
)
def test_free_decay_half_amplitude(length: float, q_factor: float) -> None:
    """
    Free-decay test: pendulum with zero crane acceleration should halve its
    angular velocity amplitude in t_half = 2*Q*ln2/omega_n seconds.

    This test is the physics ground truth for boom.py line 629. It catches the
    known bug where `v *= 1 - dt/damping_time` was used instead of
    `v *= exp(-dt / (2 * damping_time))`, which caused the pendulum to damp
    approximately 2x faster than the Q-factor definition requires.

    Parameters
    ----------
    length : float
        Wire length in metres.
    q_factor : float
        Quality factor Q (energy stored / energy lost per radian).
    """
    # --- Setup ---
    dt = 1.0  # simulation timestep (seconds)
    g = 9.81  # gravity (m/s²)
    start_speed = 1.0  # initial load velocity (m/s) -> theta_dot = start_speed / length

    crane = build_test_crane(length=length, q_factor=q_factor)
    wire = crane.boom_by_name("wire")
    assert isinstance(wire, Wire), "Not only Boom"

    # Kick the pendulum: set initial load velocity, leave crane stationary
    wire.cm_v[0] = start_speed
    crane.d_velocity = np.zeros(3)  # zero crane acceleration throughout

    # --- Analytically derived expected half-amplitude step ---
    omega_n = math.sqrt(g / length)
    # gamma = omega_n / (2*Q) is the amplitude decay rate
    # t_half = ln(2) / gamma = 2*Q*ln(2) / omega_n
    t_half_expected = 2.0 * q_factor * math.log(2.0) / omega_n
    tolerance = 0.15  # ±15% — wide enough for discrete peak-detection error (~5%),
    #          narrow enough to catch the factor-of-2 bug (~100% off)

    # Run for 3× the expected half-amplitude time to ensure we observe the crossing
    n_steps = int(3 * t_half_expected / dt) + 1

    # --- Simulate free decay ---
    theta_dot_peaks: list[tuple[int, float]] = []
    theta_dots: list[float] = []

    for step in range(n_steps):
        crane.do_step(step * dt, dt)
        theta_dot = abs(wire.cm_v[0]) / wire.length
        theta_dots.append(theta_dot)

        # Detect local maxima (oscillation peaks)
        if step >= 2:
            prev, curr, _ = theta_dots[-3], theta_dots[-2], theta_dots[-1]
            if curr >= prev and curr >= theta_dots[-1]:
                theta_dot_peaks.append((step - 1, curr))

    assert len(theta_dot_peaks) >= 3, (
        f"Too few oscillation peaks detected ({len(theta_dot_peaks)}) — "
        f"check that the pendulum is actually oscillating. "
        f"L={length}, Q={q_factor}"
    )

    # --- Find step at which peak amplitude first drops below half of initial peak ---
    initial_peak_amplitude = theta_dot_peaks[0][1]
    half_amplitude = initial_peak_amplitude / 2.0

    half_amplitude_step = None
    for step_i, peak_v in theta_dot_peaks:
        if peak_v < half_amplitude:
            half_amplitude_step = step_i
            break

    assert half_amplitude_step is not None, (
        f"Amplitude never halved within {n_steps} steps. "
        f"L={length}, Q={q_factor}, expected at step ~{t_half_expected:.0f}"
    )

    # --- Assert within tolerance of analytical prediction ---
    lower = t_half_expected * (1.0 - tolerance)
    upper = t_half_expected * (1.0 + tolerance)

    assert lower <= half_amplitude_step <= upper, (
        f"Pendulum amplitude halved at step {half_amplitude_step}, "
        f"but expected between {lower:.0f} and {upper:.0f} steps "
        f"(analytical: {t_half_expected:.1f} steps = 2*Q*ln2/omega_n). "
        f"L={length}, Q={q_factor}. "
        f"If half-amplitude step is ~{t_half_expected / 2:.0f}, "
        f"boom.py line 629 is using the energy decay constant instead of the "
        f"amplitude decay constant — see bug report."
    )


def test_free_decay_default_parameters(*, show: bool) -> None:
    """
    Regression test for default crane parameters (L=10 m, Q=50).

    For these parameters, amplitude should halve after approximately 70 simulation
    steps (dt=1.0 s). This is the exact configuration in which the original bug
    was detected — the buggy code produced half-amplitude at step ~35.

    Empirically measured (correct physics): step 67.
    Analytical prediction: 70.0 steps.
    """
    # This is a dedicated regression test for the specific configuration
    # used in crane-controller's reward_comparison.md — do not change parameters.
    L = 10.0
    Q = 50.0  # 50.0
    dt = 1.0
    g = 9.81

    crane = build_test_crane(length=L, q_factor=Q)
    wire = crane.boom_by_name("wire")
    assert isinstance(wire, Wire), "Not only Boom"
    wire.cm_v[0] = 1.0  # 1 m/s initial load velocity
    crane.d_velocity = np.zeros(3)

    theta_dot_peaks: list[tuple[int, float]] = []
    theta_dots: list[float] = []
    times: list[float] = [0.0]
    speeds: list[float] = [wire.cm_v[0]]

    for step in range(210):  # 3× expected half-amplitude time of 70 steps
        crane.do_step(step * dt, dt)
        times.append(step * dt)
        speeds.append(wire.cm_v[0])
        theta_dot = abs(wire.cm_v[0]) / wire.length
        theta_dots.append(theta_dot)
        if step >= 2:
            prev, curr = theta_dots[-3], theta_dots[-2]
            if curr >= prev and curr >= theta_dots[-1]:
                theta_dot_peaks.append((step - 1, curr))

    if show:
        _, ax = plt.subplots(1)
        ax.plot(times, speeds, label="speed")
        ax.plot(times, [np.exp(-t / wire.damping_time) for t in times], label="damping")
        plt.title(f"Damping time: {wire.damping_time}, Q: {wire.q_factor}")
        plt.show()

    initial_peak = theta_dot_peaks[0][1]
    half_amplitude_step = next((s for s, v in theta_dot_peaks if v < initial_peak / 2), None)

    omega_n = math.sqrt(g / L)
    expected = 2.0 * Q * math.log(2.0) / omega_n  # 70.0 steps

    assert half_amplitude_step is not None, "Amplitude never halved — check damping implementation"

    assert abs(half_amplitude_step - expected) / expected < 0.15, (
        f"Half-amplitude at step {half_amplitude_step}, expected ~{expected:.0f} "
        f"(±15%). "
        f"Buggy boom.py produces ~35 steps (energy constant used on amplitude). "
        f"Correct implementation produces ~70 steps (amplitude constant exp(-dt/(2*tau)))."
    )


if __name__ == "__main__":
    retcode = pytest.main(["-rA", "-v", "--rootdir", "../", "--show", "False", __file__])
    assert retcode == 0, f"Non-zero return code {retcode}"
    test_free_decay_default_parameters(show=True)
    test_free_decay_half_amplitude(10, 50)  # default crane parameters
    test_free_decay_half_amplitude(5.0, 30.0)  # shorter wire, lower Q
    test_free_decay_half_amplitude(20.0, 80.0)  # longer wire, higher Q
