
import csv
import time
import sys
from pathlib import Path

import cv2
import mediapipe as mp

# File locations
BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

# Import angle calculation from movement_analysis.py
sys.path.insert(0, str(EXPERIMENTS_DIR))
from bicep_curl import calculate_angle


def main():
    # 1. Ask for the exercise label FIRST
    valid_labels = {"bicep_curl", "shoulder_raise", "squat"}

    label = input(
        "Enter exercise label (bicep_curl, shoulder_raise, squat): "
    ).strip().lower()

    if label not in valid_labels:
        raise ValueError("Invalid exercise label.")

    # 2. Ask for participant and session IDs
    participant_id = input(
        "Enter participant ID (e.g. person_01): "
    ).strip().lower()

    session_id = input(
        "Enter session ID (e.g. session_01): "
    ).strip().lower()

    if not participant_id or not session_id:
        raise ValueError("Participant and session IDs are required.")

    # 3. Check model and create a unique CSV file
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    dataset_dir = BASE_DIR / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / (
        f"{participant_id}_{label}_{session_id}.csv"
    )

    if csv_path.exists():
        raise FileExistsError(
            f"{csv_path.name} already exists. "
            "Choose a new session ID."
        )

    # 4. Prepare CSV columns
    columns = [
        "timestamp_ms",
        "participant_id",
        "session_id",
        "label",
        "elbow_angle",
        "shoulder_angle",
    ]

    for i in range(33):
        columns.extend([
            f"x{i}", f"y{i}", f"z{i}", f"visibility{i}"
        ])

    # 5. Configure MediaPipe Pose Landmarker
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam. Close other apps using the camera."
        )

    recording = False
    saved_rows = 0
    start_time = time.perf_counter()
    last_timestamp = -1

    print("\nControls: R = start/stop recording, Q = quit")
    print(f"Exercise: {label}")
    print(f"Saving to: {csv_path}")

    try:
        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file, fieldnames=columns
            )
            writer.writeheader()

            with mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            ) as landmarker:

                while True:
                    ok, frame = camera.read()

                    if not ok:
                        print("Could not read webcam frame.")
                        break

                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    )

                    timestamp = max(
                        last_timestamp + 1,
                        int((time.perf_counter() - start_time) * 1000),
                    )
                    last_timestamp = timestamp

                    result = landmarker.detect_for_video(
                        image, timestamp
                    )

                    if result.pose_landmarks:
                        points = result.pose_landmarks[0]

                        for point in points:
                            if point.visibility > 0.5:
                                x = int(point.x * width)
                                y = int(point.y * height)
                                cv2.circle(
                                    frame, (x, y), 3, (0, 255, 0), -1
                                )

                        # Right shoulder, elbow, wrist and hip
                        shoulder = points[12]
                        elbow = points[14]
                        wrist = points[16]
                        hip = points[24]

                        elbow_angle = None
                        shoulder_angle = None

                        if min(
                            shoulder.visibility,
                            elbow.visibility,
                            wrist.visibility,
                        ) > 0.5:
                            elbow_angle = calculate_angle(
                                (shoulder.x, shoulder.y),
                                (elbow.x, elbow.y),
                                (wrist.x, wrist.y),
                            )

                        if min(
                            hip.visibility,
                            shoulder.visibility,
                            elbow.visibility,
                        ) > 0.5:
                            shoulder_angle = calculate_angle(
                                (hip.x, hip.y),
                                (shoulder.x, shoulder.y),
                                (elbow.x, elbow.y),
                            )

                        if recording:
                            row = {
                                "timestamp_ms": timestamp,
                                "participant_id": participant_id,
                                "session_id": session_id,
                                "label": label,
                                "elbow_angle": (
                                    round(elbow_angle, 2)
                                    if elbow_angle is not None else ""
                                ),
                                "shoulder_angle": (
                                    round(shoulder_angle, 2)
                                    if shoulder_angle is not None else ""
                                ),
                            }

                            for i, point in enumerate(points):
                                row[f"x{i}"] = point.x
                                row[f"y{i}"] = point.y
                                row[f"z{i}"] = point.z
                                row[f"visibility{i}"] = point.visibility

                            writer.writerow(row)
                            csv_file.flush()
                            saved_rows += 1

                        if elbow_angle is not None:
                            cv2.putText(
                                frame,
                                f"Elbow: {elbow_angle:.1f} deg",
                                (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2,
                            )

                    status = "RECORDING" if recording else "NOT RECORDING"

                    cv2.putText(
                        frame,
                        f"{label} | {status} | Rows: {saved_rows}",
                        (15, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255) if recording else (255, 255, 0),
                        2,
                    )

                    cv2.imshow("RehaTrack Data Collection", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("r"):
                        recording = not recording
                        print(
                            "Recording started."
                            if recording else "Recording paused."
                        )

                    elif key == ord("q") or key == 27:
                        break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nSaved {saved_rows} rows to:")
    print(csv_path)


if __name__ == "__main__":
    main()
