"""
collect_data.py — Exercise data collector for RehaTrack

Usage:
    python collect_data.py

Controls:
    1  — Elbow Flexion
    2  — Biceps Curl
    3  — Forearm Rotation (elbow bent)
    4  — Heel Slide
    5  — Seated Knee Extension
    6  — Mini Squat
    R  — Start / Stop recording for current exercise
    U  — Undo last session (delete it from the CSV)
    Q  — Quit and save

Data is saved to: data/exercises.csv
Each row = one frame, with landmark coordinates + key angles + label.
"""

import csv
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXERCISES = {
    "1": "elbow_flexion",
    "2": "biceps_curl",
    "3": "forearm_rotation",
    "4": "heel_slide",
    "5": "seated_knee_extension",
    "6": "mini_squat",
}

# Which angle to display live + acceptable range for each exercise
# angle_key matches compute_angles() order:
#   0=left_elbow, 1=right_elbow, 2=left_knee, 3=right_knee
EXERCISE_INFO = {
    "elbow_flexion":         {"label": "Elbow angle",  "angle_key": 1, "ref": 120, "lo": 110, "hi": 130},
    "biceps_curl":           {"label": "Elbow angle",  "angle_key": 1, "ref":  90, "lo":  80, "hi": 100},
    "forearm_rotation":      {"label": "Elbow angle",  "angle_key": 1, "ref":  90, "lo":  85, "hi":  95},
    "heel_slide":            {"label": "Knee angle",   "angle_key": 3, "ref": 110, "lo": 100, "hi": 120},
    "seated_knee_extension": {"label": "Knee angle",   "angle_key": 3, "ref":   0, "lo":   0, "hi":  10},
    "mini_squat":            {"label": "Knee angle",   "angle_key": 3, "ref":  45, "lo":  40, "hi":  50},
}

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exercises.csv")

# MediaPipe landmark indices
LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}

CSV_HEADER = (
    ["timestamp", "exercise", "rep"]
    + [f"{name}_{axis}" for name in LANDMARKS for axis in ("x", "y", "z", "vis")]
    + ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle",
       "left_shoulder_angle", "right_shoulder_angle"]
)

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (~3 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

def draw_landmarks(frame, landmarks_list):
    """Draw pose skeleton on frame manually with OpenCV."""
    h, w = frame.shape[:2]
    for pose_landmarks in landmarks_list:
        # Draw connections
        for a, b in POSE_CONNECTIONS:
            if a < len(pose_landmarks) and b < len(pose_landmarks):
                lm_a = pose_landmarks[a]
                lm_b = pose_landmarks[b]
                pt_a = (int(lm_a.x * w), int(lm_a.y * h))
                pt_b = (int(lm_b.x * w), int(lm_b.y * h))
                cv2.line(frame, pt_a, pt_b, (0, 200, 0), 2)
        # Draw joints
        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return None
    cosine = max(-1.0, min(1.0, (ba[0]*bc[0] + ba[1]*bc[1]) / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def lm_xy(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y)


def lm_row(landmarks):
    row = []
    for idx in LANDMARKS.values():
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", 0.0)
        row += [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5), round(vis, 3)]
    return row


def compute_angles(landmarks):
    def angle(a_idx, b_idx, c_idx):
        return calculate_angle(lm_xy(landmarks, a_idx),
                               lm_xy(landmarks, b_idx),
                               lm_xy(landmarks, c_idx))

    angles = [
        angle(11, 13, 15),  # left elbow
        angle(12, 14, 16),  # right elbow
        angle(23, 25, 27),  # left knee
        angle(24, 26, 28),  # right knee
        angle(13, 11, 23),  # left shoulder
        angle(14, 12, 24),  # right shoulder
    ]
    return [round(a, 2) if a is not None else "" for a in angles]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open CSV
    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file = open(OUTPUT_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(CSV_HEADER)

    # MediaPipe Tasks setup
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    current_exercise = "elbow_flexion"
    recording         = False
    rep_count         = 0
    frames_recorded   = 0
    session_start     = None
    session_buffer    = []
    last_session_rows = 0
    undo_message      = ""
    undo_message_timer = 0

    print("\n=== RehaTrack Data Collector ===")
    print("Keys: 1-6 select exercise | R = start/stop | U = undo last session | Q = quit\n")
    for k, ex in EXERCISES.items():
        info = EXERCISE_INFO[ex]
        print(f"  [{k}] {ex.replace('_',' ').title():30s} target {info['ref']}° (range {info['lo']}°–{info['hi']}°)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw skeleton
        if result.pose_landmarks:
            draw_landmarks(frame, result.pose_landmarks)

        # Buffer frame during recording
        if recording and result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            row = (
                [round(time.time(), 3), current_exercise, rep_count]
                + lm_row(landmarks)
                + compute_angles(landmarks)
            )
            session_buffer.append(row)
            frames_recorded += 1

        # HUD
        h, w = frame.shape[:2]
        status_color = (0, 0, 220) if recording else (180, 180, 180)
        status_text  = "RECORDING" if recording else "PAUSED"

        cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
        cv2.putText(frame, f"Exercise: {current_exercise.replace('_', ' ').title()}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Frames saved: {frames_recorded}",
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Live angle feedback
        if result.pose_landmarks:
            info = EXERCISE_INFO[current_exercise]
            angles = compute_angles(result.pose_landmarks[0])
            live_angle = angles[info["angle_key"]]
            if live_angle != "":
                in_range = info["lo"] <= live_angle <= info["hi"]
                angle_color = (0, 220, 0) if in_range else (0, 80, 220)
                range_text = f"Target: {info['ref']}° ({info['lo']}–{info['hi']}°)"
                angle_text = f"{info['label']}: {live_angle:.1f}°  {'IN RANGE' if in_range else 'OUT OF RANGE'}"
                cv2.rectangle(frame, (w - 360, 0), (w, 60), (30, 30, 30), -1)
                cv2.putText(frame, range_text, (w - 355, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.putText(frame, angle_text, (w - 355, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)

        if undo_message and time.time() < undo_message_timer:
            cv2.rectangle(frame, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 180), -1)
            cv2.putText(frame, undo_message, (10, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.rectangle(frame, (0, h - 52), (w, h), (30, 30, 30), -1)
        row1 = "  ".join([f"[{k}]{v.replace('_',' ')}" for k, v in list(EXERCISES.items())[:3]])
        row2 = "  ".join([f"[{k}]{v.replace('_',' ')}" for k, v in list(EXERCISES.items())[3:]])
        cv2.putText(frame, row1, (8, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(frame, row2, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("RehaTrack — Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif chr(key) in EXERCISES:
            current_exercise = EXERCISES[chr(key)]
            recording = False
            print(f"Exercise selected: {current_exercise}")

        elif key == ord("r"):
            recording = not recording
            if recording:
                session_start  = time.time()
                session_buffer = []
                print(f"Recording started — {current_exercise}")
            else:
                for row in session_buffer:
                    writer.writerow(row)
                csv_file.flush()
                last_session_rows = len(session_buffer)
                elapsed = round(time.time() - session_start, 1)
                print(f"Recording stopped — {last_session_rows} frames saved ({elapsed}s)")
                session_buffer = []

        elif key == ord("u"):
            if recording:
                discarded = len(session_buffer)
                session_buffer = []
                recording = False
                frames_recorded -= discarded
                undo_message = f"Recording discarded ({discarded} frames)"
                undo_message_timer = time.time() + 2
                print(f"Undo: discarded in-progress recording ({discarded} frames)")
            elif last_session_rows > 0:
                csv_file.flush()
                csv_file.close()
                with open(OUTPUT_FILE, "r", newline="") as f:
                    all_rows = list(csv.reader(f))
                trimmed = all_rows[:-last_session_rows]
                with open(OUTPUT_FILE, "w", newline="") as f:
                    csv.writer(f).writerows(trimmed)
                csv_file = open(OUTPUT_FILE, "a", newline="")
                writer   = csv.writer(csv_file)
                frames_recorded  -= last_session_rows
                undo_message      = f"Undo: removed last {last_session_rows} frames"
                undo_message_timer = time.time() + 2
                print(f"Undo: removed last session ({last_session_rows} frames)")
                last_session_rows = 0
            else:
                undo_message      = "Nothing to undo"
                undo_message_timer = time.time() + 2

        elif key == ord("+") or key == ord("="):
            rep_count += 1
            print(f"Rep count: {rep_count}")

        elif key == ord("-"):
            rep_count = max(0, rep_count - 1)
            print(f"Rep count: {rep_count}")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    csv_file.close()

    print(f"\nDone. {frames_recorded} frames saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
