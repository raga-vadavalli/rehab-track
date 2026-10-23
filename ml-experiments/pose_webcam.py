


#v2


import time
from pathlib import Path

import cv2
import mediapipe as mp

from bicep_curl import calculate_angle, BicepCurlCounter,check_bicep_curl_posture

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pose_landmarker_lite.task"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Close other camera apps and try again."
    )

start_time = time.perf_counter()
frame_count = 0
fps = 0.0
curl_counter = BicepCurlCounter()

try:
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            status = "No pose detected"

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Draw skeleton connections
                for a, b in CONNECTIONS:
                    p1 = landmarks[a]
                    p2 = landmarks[b]

                    x1 = int(p1.x * width)
                    y1 = int(p1.y * height)
                    x2 = int(p2.x * width)
                    y2 = int(p2.y * height)

                    cv2.line(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                # Draw landmarks
                for point in landmarks:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame, (x, y), 4,
                        (0, 0, 255), -1
                    )

                status = "Pose detected"

                # Right arm: shoulder 12, elbow 14, wrist 16
                shoulder = landmarks[12]
                elbow = landmarks[14]
                wrist = landmarks[16]

                if min(
                    shoulder.visibility,
                    elbow.visibility,
                    wrist.visibility,
                ) > 0.5:

                    angle = calculate_angle(
                        (shoulder.x * width,
                         shoulder.y * height),
                        (elbow.x * width,
                         elbow.y * height),
                        (wrist.x * width,
                         wrist.y * height),
                    )

                    reps, phase = curl_counter.update(angle)
                    feedback = check_bicep_curl_posture(
                    shoulder,
                     elbow,
                    wrist,
                    )   

                    cv2.putText(
                        frame,
                            f"Feedback: {feedback}",
                            (15, 135),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
    

                    cv2.putText(
                        frame,
                        f"Elbow angle: {angle:.1f} deg",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Curls: {reps} | Phase: {phase}",
                        (15, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                else:
                    cv2.putText(
                        frame,
                        "Keep your right arm visible",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            frame_count += 1
            elapsed = time.perf_counter() - start_time

            if elapsed > 0:
                fps = frame_count / elapsed

            cv2.putText(
                frame,
                f"{status} | FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow(
                "RehaTrack - MediaPipe Baseline", frame
            )

            # Press Q or ESC to exit
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

