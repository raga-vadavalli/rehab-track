"""
server.py -- RehaTrack web dashboard backend.
Run: python -m uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from bicep_curl import BicepCurlCounter, calculate_angle, check_bicep_curl_posture
from squat import SquatCounter, check_squat_posture
from shoulder_press import ShoulderPressCounter, check_shoulder_press_posture
from session_store import get_session, list_sessions, save_session

# -- Paths ------------------------------------------------------------------
MODEL_PATH         = BASE / "models" / "pose_landmarker_lite.task"
PRESCRIPTIONS_PATH = BASE / "prescriptions.json"
STATIC_DIR         = BASE / "static"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")

# -- Skeleton connections ---------------------------------------------------
CONNECTIONS = [
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),(9,10),
]

# -- Pydantic models --------------------------------------------------------
class PrescriptionModel(BaseModel):
    display_name:    str
    exercise:        str   = "bicep_curl"
    arm:             str   = "right"
    target_reps:     int
    extension_angle: float
    flexion_angle:   float

    @field_validator("display_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator("target_reps")
    @classmethod
    def reps_range(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError("Reps must be 1-50")
        return v

    @field_validator("extension_angle")
    @classmethod
    def ext_range(cls, v: float) -> float:
        if not (90 <= v <= 180):
            raise ValueError("Extension angle must be 90-180 deg")
        return v

    @field_validator("flexion_angle")
    @classmethod
    def flex_range(cls, v: float) -> float:
        # Squats need up to 130 deg flexion; bicep curls need 10-90 deg
        if not (10 <= v <= 130):
            raise ValueError("Flexion angle must be 10-130 deg")
        return v


class SessionStartRequest(BaseModel):
    prescription_key: str
    prescription:     PrescriptionModel


# -- Camera / inference worker ----------------------------------------------
class CameraWorker:
    def __init__(self, prescription: dict) -> None:
        self.prescription = prescription
        self.running      = False
        self._thread: Optional[threading.Thread] = None
        self._lock        = threading.Lock()
        self._frame: Optional[bytes] = None
        self._state: dict = {
            "angle":         None,
            "reps":          0,
            "phase":         "waiting",
            "feedback":      "Starting camera...",
            "is_valid_form": False,
            "pose_found":    False,
            "session_done":  False,
            "elapsed":       0.0,
        }
        self._start_ts: float = 0.0

    def start(self) -> None:
        self.running    = True
        self._start_ts  = time.perf_counter()
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    def get_state(self) -> dict:
        with self._lock:
            s = dict(self._state)
        s["elapsed"] = round(time.perf_counter() - self._start_ts, 1)
        return s

    def _run(self) -> None:
        BaseOptions           = mp.tasks.BaseOptions
        PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode           = mp.tasks.vision.RunningMode

        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
        )

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            with self._lock:
                self._state["feedback"] = "Cannot open camera"
            self.running = False
            return

        rx           = self.prescription
        exercise     = rx.get("exercise", "bicep_curl")
        target_reps  = rx["target_reps"]
        session_done = False
        t0           = time.perf_counter()

        if exercise == "squat":
            counter = SquatCounter(
                extension_angle=rx["extension_angle"],
                flexion_angle=rx["flexion_angle"],
            )
        elif exercise == "shoulder_press":
            counter = ShoulderPressCounter(
                extension_angle=rx["extension_angle"],
                flexion_angle=rx["flexion_angle"],
            )
        else:  # bicep_curl (default)
            counter = BicepCurlCounter(
                extension_angle=rx["extension_angle"],
                flexion_angle=rx["flexion_angle"],
            )

        try:
            with PoseLandmarker.create_from_options(opts) as landmarker:
                while self.running:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame    = cv2.flip(frame, 1)
                    h, w     = frame.shape[:2]
                    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms    = int((time.perf_counter() - t0) * 1000)
                    result   = landmarker.detect_for_video(mp_img, ts_ms)

                    pose_found = bool(result.pose_landmarks)
                    angle      = None
                    feedback   = "No pose detected"
                    is_valid   = False
                    reps       = counter.reps
                    phase      = counter.phase

                    if pose_found:
                        lm = result.pose_landmarks[0]

                        # Draw skeleton
                        for a, b in CONNECTIONS:
                            cv2.line(frame,
                                     (int(lm[a].x*w), int(lm[a].y*h)),
                                     (int(lm[b].x*w), int(lm[b].y*h)),
                                     (0, 180, 130), 2, cv2.LINE_AA)
                        for pt in lm:
                            cv2.circle(frame,
                                       (int(pt.x*w), int(pt.y*h)),
                                       4, (240, 240, 240), -1, cv2.LINE_AA)

                        if exercise == "squat":
                            hip      = lm[24]; knee     = lm[26]
                            ankle    = lm[28]; shoulder = lm[12]

                            if min(hip.visibility, knee.visibility,
                                   ankle.visibility) > 0.5:
                                angle = calculate_angle(
                                    (hip.x*w,   hip.y*h),
                                    (knee.x*w,  knee.y*h),
                                    (ankle.x*w, ankle.y*h),
                                )
                                feedback, is_valid = check_squat_posture(
                                    hip, knee, ankle, shoulder
                                )

                                if is_valid and angle is not None:
                                    reps, phase = counter.update(angle)

                                if reps >= target_reps and not session_done:
                                    session_done = True

                                # Angle label near knee
                                if angle is not None:
                                    kx = int(knee.x * w)
                                    ky = int(knee.y * h)
                                    cv2.putText(frame, f"{angle:.0f}\u00b0",
                                                (kx + 10, ky - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (0, 210, 255), 2, cv2.LINE_AA)
                            else:
                                feedback = "Keep your full body visible"

                        elif exercise == "shoulder_press":
                            shoulder = lm[12]; elbow = lm[14]
                            wrist    = lm[16]; hip   = lm[24]

                            if min(shoulder.visibility, elbow.visibility,
                                   wrist.visibility, hip.visibility) > 0.5:

                                angle = calculate_angle(
                                    (shoulder.x*w, shoulder.y*h),
                                    (elbow.x*w,    elbow.y*h),
                                    (wrist.x*w,    wrist.y*h),
                                )
                                feedback, is_valid = check_shoulder_press_posture(
                                    shoulder, elbow, wrist, hip
                                )

                                if is_valid and angle is not None:
                                    reps, phase = counter.update(angle)

                                if reps >= target_reps and not session_done:
                                    session_done = True

                                if angle is not None:
                                    ex = int(elbow.x * w)
                                    ey = int(elbow.y * h)
                                    cv2.putText(frame, f"{angle:.0f}\u00b0",
                                                (ex + 10, ey - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (0, 210, 255), 2, cv2.LINE_AA)
                            else:
                                feedback = "Keep your arm and shoulder visible"

                        else:  # bicep_curl
                            shoulder = lm[12]; elbow = lm[14]
                            wrist    = lm[16]; hip   = lm[24]

                            if min(shoulder.visibility, elbow.visibility,
                                   wrist.visibility, hip.visibility) > 0.5:

                                angle = calculate_angle(
                                    (shoulder.x*w, shoulder.y*h),
                                    (elbow.x*w,    elbow.y*h),
                                    (wrist.x*w,    wrist.y*h),
                                )
                                feedback, is_valid, _ = check_bicep_curl_posture(
                                    shoulder, elbow, wrist, hip
                                )

                                if is_valid and angle is not None:
                                    reps, phase = counter.update(angle)

                                if reps >= target_reps and not session_done:
                                    session_done = True

                                # Angle label on frame
                                if angle is not None:
                                    ex = int(elbow.x * w)
                                    ey = int(elbow.y * h)
                                    cv2.putText(frame, f"{angle:.0f}\u00b0",
                                                (ex + 10, ey - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7, (0, 210, 255), 2, cv2.LINE_AA)
                            else:
                                feedback = "Keep your arm visible"

                    # Mini HUD strip
                    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
                    cv2.putText(
                        frame,
                        f"Reps: {reps}/{target_reps}   {phase.capitalize()}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 180, 130), 2, cv2.LINE_AA,
                    )

                    _, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78]
                    )
                    with self._lock:
                        self._frame = jpg.tobytes()
                        self._state = {
                            "angle":
                                round(angle, 1) if angle is not None else None,
                            "reps":          reps,
                            "phase":         phase,
                            "feedback":      feedback,
                            "is_valid_form": is_valid,
                            "pose_found":    pose_found,
                            "session_done":  session_done,
                            "elapsed":
                                round(time.perf_counter() - self._start_ts, 1),
                        }
        finally:
            cap.release()


# -- Global session state ---------------------------------------------------
_worker:                 Optional[CameraWorker] = None
_active_prescription:    Optional[dict]         = None
_active_prescription_key: Optional[str]         = None


# -- Helpers ----------------------------------------------------------------
def _load_prescriptions() -> dict:
    if PRESCRIPTIONS_PATH.exists():
        with open(PRESCRIPTIONS_PATH) as f:
            return json.load(f)
    return {}


def _save_prescriptions(data: dict) -> None:
    with open(PRESCRIPTIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# -- FastAPI app ------------------------------------------------------------
app = FastAPI(title="RehaTrack")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(
        (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
    )


# -- Prescriptions ----------------------------------------------------------
@app.get("/prescriptions")
async def get_prescriptions():
    return _load_prescriptions()


@app.post("/prescriptions", status_code=201)
async def create_prescription(body: PrescriptionModel):
    data     = _load_prescriptions()
    base_key = body.display_name.lower().replace(" ", "_")
    key, n   = base_key, 2
    while key in data:
        key = f"{base_key}_{n}"; n += 1
    data[key] = body.model_dump()
    _save_prescriptions(data)
    return {"key": key, **body.model_dump()}


@app.put("/prescriptions/{key}")
async def update_prescription(key: str, body: PrescriptionModel):
    data = _load_prescriptions()
    if key not in data:
        raise HTTPException(404, f"Key '{key}' not found")
    data[key] = body.model_dump()
    _save_prescriptions(data)
    return {"key": key, **body.model_dump()}


@app.delete("/prescriptions/{key}")
async def delete_prescription(key: str):
    data = _load_prescriptions()
    if key not in data:
        raise HTTPException(404, f"Key '{key}' not found")
    del data[key]
    _save_prescriptions(data)
    return {"deleted": key}


# -- Session ----------------------------------------------------------------
@app.post("/session/start")
async def start_session(body: SessionStartRequest):
    global _worker, _active_prescription, _active_prescription_key
    if _worker is not None and _worker.running:
        raise HTTPException(409, "A session is already active")
    _active_prescription     = body.prescription.model_dump()
    _active_prescription_key = body.prescription_key
    _worker = CameraWorker(_active_prescription)
    _worker.start()
    return {"status": "started", "prescription": _active_prescription}


@app.post("/session/stop")
async def stop_session():
    global _worker, _active_prescription, _active_prescription_key
    if _worker is None or not _worker.running:
        raise HTTPException(400, "No active session")

    state = _worker.get_state()
    _worker.stop()

    rx      = _active_prescription
    summary = {
        "patient":          rx["display_name"],
        "exercise":         rx["exercise"],
        "arm":              rx["arm"],
        "target_reps":      rx["target_reps"],
        "completed_reps":   state["reps"],
        "duration_s":       state["elapsed"],
        "status":           "complete" if state["session_done"] else "incomplete",
        "prescription_key": _active_prescription_key,
        "extension_angle":  rx["extension_angle"],
        "flexion_angle":    rx["flexion_angle"],
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    session_id    = save_session(summary)
    summary["id"] = session_id

    _worker                  = None
    _active_prescription     = None
    _active_prescription_key = None
    return summary


@app.get("/session/state")
async def session_state():
    if _worker is None:
        return {"active": False}
    return {"active": _worker.running, **_worker.get_state()}


# -- MJPEG video feed -------------------------------------------------------
async def _mjpeg_gen():
    while True:
        if _worker is not None:
            frame = _worker.get_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
        await asyncio.sleep(0.033)


@app.get("/video_feed")
async def video_feed():
    if _worker is None:
        raise HTTPException(400, "No active session")
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# -- Server-Sent Events -----------------------------------------------------
async def _sse_gen():
    while True:
        if _worker is not None:
            data = _worker.get_state()
            data["active"] = True
        else:
            data = {"active": False}
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.1)


@app.get("/events")
async def events():
    return StreamingResponse(
        _sse_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -- History ----------------------------------------------------------------
@app.get("/sessions")
async def get_sessions():
    return list_sessions()


@app.get("/sessions/{session_id}")
async def get_one_session(session_id: str):
    s = get_session(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    return s
