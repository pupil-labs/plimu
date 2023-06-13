import math

import cv2
import numpy as np


def euler_from_quaternion_xyz(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def euler_from_quaternion_yxz(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around y in radians (counterclockwise)
    pitch is rotation around x in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = -2.0 * (x * z - y * w)
    t1 = +2.0 * (w * w + z * z) - 1.0
    roll_y = math.atan2(t0, t1)

    t2 = +2.0 * (x * w + y * z)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_x = math.asin(t2)

    t3 = -2.0 * (x * y - z * w)
    t4 = +2.0 * (y * y + w * w) - 1.0
    yaw_z = math.atan2(t3, t4)

    return roll_y, pitch_x, yaw_z  # in radians


def R_from_angles_body(angles, convention="xyz"):
    axes = {
        "x": np.asarray([1.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 1.0, 0.0]),
        "z": np.asarray([0.0, 0.0, 1.0]),
    }

    r3, r2, r1 = convention
    R1 = cv2.Rodrigues(axes[r1] * angles[2])[0]
    ax2_prime = R1 @ axes[r2]
    R2 = cv2.Rodrigues(ax2_prime * angles[1])[0]
    ax3_prime = R2 @ R1 @ axes[r3]
    R3 = cv2.Rodrigues(ax3_prime * angles[0])[0]

    return R3 @ R2 @ R1
