"""
squat.py -- Squat rep counter and posture checker.

Tracks the hip-knee-ankle angle on the right side.
Extension phase: leg nearly straight (~160 deg, standing).
Flexion  phase: knee bent    (~100 deg, squatting).
"""

from collections import deque
import math


# ---------------------------------------------------------------------------
def calculate_angle(a, b, c) -> float:
    """
    2-D angle at joint B formed by segments B-A and B-C.
    Each point is (x, y).
    """
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot     = ax * cx + ay * cy
    mag     = math.sqrt(ax*ax + ay*ay) * math.sqrt(cx*cx + cy*cy)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


# ---------------------------------------------------------------------------
class SquatCounter:
    """
    Counts squat reps using a rolling-average angle buffer.

    Parameters
    ----------
    extension_angle : float
        Hip-knee-ankle angle that counts as "standing" (default 160 deg).
    flexion_angle   : float
        Hip-knee-ankle angle that counts as "squatted"  (default 100 deg).
    """

    BUFFER = 5

    def __init__(self, extension_angle: float = 160.0,
                 flexion_angle: float = 100.0) -> None:
        self.extension_angle = extension_angle
        self.flexion_angle   = flexion_angle
        self.reps   = 0
        self.phase  = "waiting"           # waiting | extended | flexed
        self._buf   = deque(maxlen=self.BUFFER)

    def update(self, raw_angle: float):
        """
        Feed one angle sample; returns (reps, phase).
        """
        self._buf.append(raw_angle)
        angle = sum(self._buf) / len(self._buf)

        if self.phase == "waiting":
            if angle >= self.extension_angle:
                self.phase = "extended"

        elif self.phase == "extended":
            if angle <= self.flexion_angle:
                self.phase = "flexed"

        elif self.phase == "flexed":
            if angle >= self.extension_angle:
                self.reps += 1
                self.phase = "extended"

        return self.reps, self.phase


# ---------------------------------------------------------------------------
def check_squat_posture(hip, knee, ankle, shoulder):
    """
    Posture checks for a squat using right-side MediaPipe landmarks.

    Parameters
    ----------
    hip      : landmark 24 (right hip)
    knee     : landmark 26 (right knee)
    ankle    : landmark 28 (right ankle)
    shoulder : landmark 12 (right shoulder)

    Returns
    -------
    (message: str, is_valid: bool)
    """
    # 1. Visibility gate
    min_vis = min(hip.visibility, knee.visibility, ankle.visibility,
                  shoulder.visibility)
    if min_vis < 0.5:
        return "Keep your full body visible", False

    # 2. Knee should be roughly above ankle (not collapsing inward/outward)
    knee_ankle_dx = abs(knee.x - ankle.x)
    if knee_ankle_dx > 0.12:
        return "Keep knee over foot", False

    # 3. Shoulder should stay roughly above hip (no excessive forward lean)
    shoulder_hip_dx = shoulder.x - hip.x
    if abs(shoulder_hip_dx) > 0.15:
        return "Keep your back upright", False

    # 4. Hip should be below shoulder (basic sanity: person is upright)
    if hip.y < shoulder.y:
        return "Full body not visible", False

    return "Good form", True
