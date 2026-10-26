"""
shoulder_press.py -- Shoulder press rep counter and posture checker.

Tracks the right elbow angle (shoulder-elbow-wrist).
Flexion  phase: arms bent, bar at shoulder level (~90 deg).
Extension phase: arms fully pressed overhead         (~160 deg).
"""

from collections import deque
import math


# ---------------------------------------------------------------------------
def calculate_angle(a, b, c) -> float:
    """Angle at joint B from 2-D coords (x, y)."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = math.sqrt(ax*ax + ay*ay) * math.sqrt(cx*cx + cy*cy)
    if mag < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


# ---------------------------------------------------------------------------
class ShoulderPressCounter:
    """
    Counts shoulder-press reps using a rolling-average angle buffer.

    Parameters
    ----------
    extension_angle : float
        Elbow angle that counts as "arms fully pressed" (default 160 deg).
    flexion_angle   : float
        Elbow angle that counts as "bar at shoulder"   (default 90 deg).
    """

    BUFFER = 5

    def __init__(self, extension_angle: float = 160.0,
                 flexion_angle: float = 90.0) -> None:
        self.extension_angle = extension_angle
        self.flexion_angle   = flexion_angle
        self.reps   = 0
        self.phase  = "waiting"         # waiting | flexed | extended
        self._buf   = deque(maxlen=self.BUFFER)

    def update(self, raw_angle: float):
        """Feed one angle sample; returns (reps, phase)."""
        self._buf.append(raw_angle)
        angle = sum(self._buf) / len(self._buf)

        if self.phase == "waiting":
            # Wait until arms are in the starting (flexed) position
            if angle <= self.flexion_angle:
                self.phase = "flexed"

        elif self.phase == "flexed":
            if angle >= self.extension_angle:
                self.phase = "extended"

        elif self.phase == "extended":
            if angle <= self.flexion_angle:
                self.reps += 1
                self.phase = "flexed"

        return self.reps, self.phase


# ---------------------------------------------------------------------------
def check_shoulder_press_posture(shoulder, elbow, wrist, hip):
    """
    Posture checks for a shoulder press using right-side MediaPipe landmarks.

    Parameters
    ----------
    shoulder : landmark 12 (right shoulder)
    elbow    : landmark 14 (right elbow)
    wrist    : landmark 16 (right wrist)
    hip      : landmark 24 (right hip)

    Returns
    -------
    (message: str, is_valid: bool)
    """
    # 1. Visibility gate
    min_vis = min(shoulder.visibility, elbow.visibility,
                  wrist.visibility, hip.visibility)
    if min_vis < 0.5:
        return "Keep your arm and shoulder visible", False

    # 2. Elbow must start at or above shoulder height (pressing overhead)
    if elbow.y > shoulder.y + 0.15:
        return "Raise your elbow to shoulder height", False

    # 3. Wrist should be above elbow when pressing (vertical movement)
    if wrist.y > elbow.y + 0.05:
        return "Press upward -- wrist above elbow", False

    # 4. Elbow should not drift too far from the body laterally
    elbow_drift = abs(elbow.x - shoulder.x)
    if elbow_drift > 0.25:
        return "Keep elbows in -- don't flare out", False

    # 5. Shoulder should stay roughly above hip (no excessive lean)
    shoulder_lean = abs(shoulder.x - hip.x)
    if shoulder_lean > 0.18:
        return "Keep your back upright", False

    return "Good form", True
