from math import acos, atan2, cos, degrees, pi, radians, sin
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot


def test_atan():
    """Check the atan2 with respect to small angle changes."""
    dr = radians(1.0)  # small angle change in radians
    for a in range(360):
        r0 = radians(a)
        print(f"{r0 + dr} =? {r0} + {dr / (1 + r0**2)} : {r0 + dr / (1 + r0**2)}")


def test_d_atan():
    """Check the arctan derivative formula."""
    for y in range(100):
        a = atan2(y, 1.0)  # angle of tangent in y direction
        dat1 = atan2(y + 1.0, 1.0) - atan2(y, 1.0)
        dat2 = 1.0 / (1.0 + y**2)
        print(f"Angle {degrees(a)}: derivative: {dat2} =? {dat1} (difference)")


def test_turn():
    def turn(theta: float, phi: float, alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, approx: int = 1):
        """Euler-turn the vector by the small yaw, pitch, roll angles (alpha, beta, gamma)."""
        st = sin(theta)
        ct = cos(theta)
        sp = sin(phi)
        cp = cos(phi)
        if approx == 1:
            theta = theta + beta * cp - gamma * sp
            phi = atan2(alpha * st * cp + st * sp - gamma * ct, st * cp - alpha * st * sp + beta * ct)
        elif approx == 2:
            #             return (theta + beta*cp - gamma*sp +(beta**2+gamma**2)/tan(theta)/2.0,
            #                     atan2(alpha*st*cp + (1-alpha**2/2-gamma**2/2)*st*sp - (gamma-alpha*beta)*ct,
            #                           (1-alpha**2/2-beta**2/2)*st*cp - (alpha-beta*gamma)*st*sp +(beta+alpha*gamma)*ct))
            sa = alpha
            ca = 1 - alpha * alpha / 2
            sb = beta
            cb = 1 - beta * beta / 2
            sg = gamma
            cg = 1 - gamma * gamma / 2
            theta = acos(-sb * st * cp + cb * sg * st * sp + cb * cg * ct)
            phi = atan2(
                sa * cb * st * cp + (sa * sb * sg + ca * cg) * st * sp + (sa * sb * cg - ca * sg) * ct,
                ca * cb * st * cp + (ca * sb * sg - sa * cg) * st * sp + (ca * sb * cg + sa * sg) * ct,
            )
        elif approx == -1:
            sa = sin(alpha)
            ca = cos(alpha)
            sb = sin(beta)
            cb = cos(beta)
            sg = sin(gamma)
            cg = cos(gamma)
            theta = acos(-sb * st * cp + cb * sg * st * sp + cb * cg * ct)
            phi = atan2(
                sa * cb * st * cp + (sa * sb * sg + ca * cg) * st * sp + (sa * sb * cg - ca * sg) * ct,
                ca * cb * st * cp + (ca * sb * sg - sa * cg) * st * sp + (ca * sb * cg + sa * sg) * ct,
            )
        elif approx == -2:
            r = Rot.from_euler("xyz", (gamma, beta, alpha))  # 0: roll, 1: pitch, 2: yaw
            # print(f"Matrix:{r.as_matrix}")
            x = r.apply((st * cp, st * sp, ct))
            # print(f"({st*cp}, {st*sp}, {ct}) -> {x}")
            theta = acos(x[2])
            phi = atan2(x[1], x[0])
        else:
            raise NotImplementedError(f"Unknown value for approx: {approx}") from None
        return (theta, phi)

    # theta, phi = turn( radians(0), radians(0), beta=radians(10), approx=-2)
    # return
    a1 = radians(1)
    theta = pi / 2
    phi = 0.0
    for a in np.linspace(0, 360, 361):
        # print(f"Yaw angle {a}: ({degrees(theta)}, {degrees(phi)})")
        assert abs(theta - pi / 2) < 1e-10, f"theta:{theta}, expected: {pi / 2}"
        if a <= 180:
            assert abs(phi - radians(a)) < 1e-4, f"phi:{phi}, expected: {radians(a)}"
        else:
            assert abs(2 * pi + phi - radians(a)) < 1e-3, f"phi:{phi}, expected: {-radians(a)}"
        theta, phi = turn(theta, phi, alpha=a1, approx=-2)

    theta = 0.0
    phi = 0.0
    for a in np.linspace(0, 179, 180):
        # print(f"Pitch angle {a}: ({degrees(theta)}, {degrees(phi)})")
        assert abs(phi) < 1e-10, f"phi:{phi} != 0"
        assert abs(theta - radians(a)) < 1e-4, f"phi:{phi}, expected: {radians(a)}"

        theta, phi = turn(theta, phi, beta=a1, approx=-2)

    theta = 0
    phi = -pi / 2.0
    for a in np.linspace(0, 360, 361):
        # print(f"Roll angle {a}: ({degrees(theta)}, {degrees(phi)})")
        if a < 180:
            assert abs(theta - radians(a)) < 1e-10, f"theta:{theta}, expected: {radians(a)}"
            assert abs(phi + pi / 2) < 1e-3, f"phi:{phi}, expected: {radians(-90)}"
        else:
            assert abs(theta - 2 * pi + radians(a)) < 1e-10, f"theta:{theta}, expected: {2 * pi - radians(a)}"
            assert abs(phi - pi / 2) < 1e-3, f"phi:{phi}, expected: {radians(90)}"

        theta, phi = turn(theta, phi, gamma=a1, approx=-2)


if __name__ == "__main__":
    retcode = pytest.main(args=["-rA", "-v", __file__, "--show", "True"])
    assert retcode == 0, f"Non-zero return code {retcode}"
    import os

    os.chdir(Path(__file__).parent.absolute() / "test_working_directory")
    # test_atan()
    # test_d_atan()
    # test_turn()
