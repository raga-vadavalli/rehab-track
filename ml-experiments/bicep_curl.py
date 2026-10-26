import math


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    Each point is an (x, y) coordinate.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


class BicepCurlCounter:
    def __init__(self, extension_angle=150, flexion_angle=60):
        self.reps          = 0
        self.phase         = "waiting"
        self.extension_angle = extension_angle
        self.flexion_angle   = flexion_angle

        # Smoothing: keep a small rolling buffer of angles
        self._buffer     = []
        self._buffer_size = 5

    def _smooth(self, angle):
        self._buffer.append(angle)
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)
        return sum(self._buffer) / len(self._buffer)

    def update(self, elbow_angle):
        if elbow_angle is None:
            return self.reps, self.phase

        angle = self._smooth(elbow_angle)

        if angle > self.extension_angle:
            if self.phase == "flexed":
                self.reps += 1
            self.phase = "extended"

        elif angle < self.flexion_angle:
            if self.phase == "extended":
                self.phase = "flexed"

        return self.reps, self.phase


def check_bicep_curl_posture(
    shoulder,
    elbow,
    wrist,
    hip,
    visibility_threshold=0.5,
):
    """
    shoulder, elbow, wrist, hip are MediaPipe landmarks.
    Returns (message: str, is_valid: bool, shoulder_angle: float|None).
    is_valid=True only when form is good enough to count a rep.
    """

    # ── 1. Visibility ────────────────────────────────────────────────────────
    for point, name in [(shoulder, "shoulder"), (elbow, "elbow"),
                        (wrist, "wrist"), (hip, "hip")]:
        if getattr(point, "visibility", 0.0) < visibility_threshold:
            return "Move into clear camera view", False, None

    # ── 2. Shoulder angle (elbow → shoulder → hip) ───────────────────────────
    #   During a proper curl the upper arm stays close to the torso,
    #   so the shoulder angle (elbow-shoulder-hip) should stay roughly
    #   between 0° and 40°.  If it rises above ~50° the patient is
    #   swinging the elbow forward / using momentum.
    shoulder_angle = calculate_angle(
        (elbow.x,    elbow.y),
        (shoulder.x, shoulder.y),
        (hip.x,      hip.y),
    )

    if shoulder_angle is not None and shoulder_angle > 50:
        return (
            f"Don't swing your elbow forward ({shoulder_angle:.0f}°)",
            False,
            shoulder_angle,
        )

    # ── 3. Elbow must not flare out sideways ─────────────────────────────────
    elbow_drift = abs(elbow.x - shoulder.x)
    if elbow_drift > 0.15:
        return "Keep your elbow close to your body", False, shoulder_angle

    # ── 4. Elbow must be below the shoulder ──────────────────────────────────
    if elbow.y < shoulder.y:
        return "Lower your elbow below your shoulder", False, shoulder_angle

    # ── 5. Upper arm must have some vertical extent ───────────────────────────
    if abs(shoulder.y - elbow.y) < 0.05:
        return "Keep your upper arm steady", False, shoulder_angle

    # ── 6. Wrist must not dangle below elbow ─────────────────────────────────
    if wrist.y > elbow.y + 0.25:
        return "Raise your wrist — complete the curl", False, shoulder_angle

    # ── 7. Shoulder must not be shrugging (shoulder too high) ────────────────
    if shoulder_angle is not None and shoulder_angle < 0:
        return "Relax your shoulder — don't shrug", False, shoulder_angle

    return "Good form", True, shoulder_angle
