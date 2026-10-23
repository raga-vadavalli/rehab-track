
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
    def __init__(self):
        self.reps = 0
        self.phase = "waiting"

    def update(self, elbow_angle):
        if elbow_angle is None:
            return self.reps, self.phase

        # Arm extended: ready position
        if elbow_angle > 150:
            if self.phase == "flexed":
                self.reps += 1
            self.phase = "extended"

        # Arm curled
        elif elbow_angle < 60:
            if self.phase == "extended":
                self.phase = "flexed"

        return self.reps, self.phase


def check_bicep_curl_posture(
    shoulder,
    elbow,
    wrist,
    visibility_threshold=0.5,
):
    """
    shoulder, elbow, wrist are MediaPipe landmarks.
    Returns a simple feedback message.
    """

    points = [shoulder, elbow, wrist]

    if any(
        getattr(point, "visibility", 0.0) < visibility_threshold
        for point in points
    ):
        return "Move into clear camera view"

    # Compare shoulder and elbow vertical positions.
    # Image y increases downward.
    arm_vertical_difference = abs(shoulder.y - elbow.y)

    if arm_vertical_difference < 0.05:
        return "Keep your upper arm steady"

    return "Arm position looks steady"
