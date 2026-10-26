#v4 — professional UI + quit confirmation

import json
import time
from pathlib import Path

import cv2
import mediapipe as mp

from bicep_curl import calculate_angle, BicepCurlCounter, check_bicep_curl_posture

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH         = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
PRESCRIPTIONS_PATH = Path(__file__).parent / "prescriptions.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# ---------------------------------------------------------------------------
# Colours & style
# ---------------------------------------------------------------------------

C_BG         = (20, 20, 20)
C_PANEL      = (35, 35, 35)
C_ACCENT     = (0, 180, 130)       # teal
C_WHITE      = (240, 240, 240)
C_GREY       = (140, 140, 140)
C_WARN       = (0, 80, 220)        # red-ish (BGR)
C_SUCCESS    = (60, 180, 60)
C_YELLOW     = (0, 210, 255)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD    = cv2.FONT_HERSHEY_DUPLEX

def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_text(img, text, x, y, font=FONT, scale=0.65, color=C_WHITE, thickness=1):
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Prescription helpers
# ---------------------------------------------------------------------------

def load_prescriptions():
    if PRESCRIPTIONS_PATH.exists():
        with open(PRESCRIPTIONS_PATH) as f:
            return json.load(f)
    return {}

def save_prescriptions(data):
    with open(PRESCRIPTIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def prompt_float(prompt, default, lo, hi):
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}, default {default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"    Please enter a number between {lo} and {hi}.")

def prompt_int(prompt, default, lo, hi):
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}, default {default}]: ").strip()
        if raw == "":
            return int(default)
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"    Please enter a whole number between {lo} and {hi}.")

def create_new_profile(prescriptions):
    print("\n  ── New Prescription ──────────────────────────")
    name = input("  Patient name: ").strip() or "Patient"

    print("\n  Prescription angles (ask your physiotherapist):")
    extension_angle = prompt_float("  Full extension angle (arm straight)", 150, 90, 180)
    flexion_angle   = prompt_float("  Full flexion angle  (arm curled)",     60, 10,  90)
    target_reps     = prompt_int  ("  Number of reps prescribed",              5,  1,  50)

    key = name.lower().replace(" ", "_")
    # Make key unique if it already exists
    base_key, n = key, 2
    while key in prescriptions:
        key = f"{base_key}_{n}"; n += 1

    profile = {
        "display_name":    name,
        "exercise":        "bicep_curl",
        "arm":             "right",
        "target_reps":     target_reps,
        "extension_angle": extension_angle,
        "flexion_angle":   flexion_angle,
    }
    prescriptions[key] = profile
    save_prescriptions(prescriptions)
    print(f"\n  Profile saved as '{key}'.")
    return profile

# ---------------------------------------------------------------------------
# Startup menu
# ---------------------------------------------------------------------------

prescriptions = load_prescriptions()

print("\n╔══════════════════════════════════════════╗")
print("║          RehaTrack — Welcome             ║")
print("╚══════════════════════════════════════════╝\n")

keys = list(prescriptions.keys())

if keys:
    print("  Existing profiles:")
    for i, key in enumerate(keys):
        p = prescriptions[key]
        print(
            f"    [{i + 1}]  {p['display_name']:<22}"
            f"  {p['target_reps']} reps   "
            f"  extend>{p['extension_angle']}°  flex<{p['flexion_angle']}°"
        )
    print(f"\n    [N]  Enter new prescription")
    print(f"    [D]  Delete a profile\n")

    while True:
        choice = input(f"  Select option (1–{len(keys)} / N / D): ").strip().lower()
        if choice == "n":
            profile = create_new_profile(prescriptions)
            break
        elif choice == "d":
            del_choice = input(f"  Enter number to delete (1–{len(keys)}): ").strip()
            if del_choice.isdigit() and 1 <= int(del_choice) <= len(keys):
                del_key = keys[int(del_choice) - 1]
                confirm = input(f"  Delete '{prescriptions[del_key]['display_name']}'? (y/n): ").strip().lower()
                if confirm == "y":
                    del prescriptions[del_key]
                    save_prescriptions(prescriptions)
                    print("  Profile deleted.")
                    # Restart menu by reloading
                    prescriptions = load_prescriptions()
                    keys = list(prescriptions.keys())
                    if not keys:
                        profile = create_new_profile(prescriptions)
                        break
                    for i, key in enumerate(keys):
                        p = prescriptions[key]
                        print(f"    [{i+1}]  {p['display_name']:<22}  {p['target_reps']} reps")
            else:
                print("  Invalid choice.")
        elif choice.isdigit() and 1 <= int(choice) <= len(keys):
            profile = prescriptions[keys[int(choice) - 1]]
            break
        else:
            print("  Invalid choice, try again.")
else:
    print("  No profiles found. Please enter a new prescription.\n")
    profile = create_new_profile(prescriptions)

print(f"\n  ── Ready ─────────────────────────────────────")
print(f"  Patient  : {profile['display_name']}")
print(f"  Exercise : {profile['exercise'].replace('_', ' ').title()}")
print(f"  Reps     : {profile['target_reps']}")
print(f"  Extend   : > {profile['extension_angle']}°")
print(f"  Flex     : < {profile['flexion_angle']}°")
print(f"\n  Press Q or ESC to quit.\n")

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------

BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode

CONNECTIONS = [
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),(9,10),
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

start_time      = time.perf_counter()
frame_count     = 0
fps             = 0.0
curl_counter    = BicepCurlCounter(
    extension_angle=profile["extension_angle"],
    flexion_angle=profile["flexion_angle"],
)
target_reps     = profile["target_reps"]
session_done    = False
deviation_msg   = ""
deviation_timer = 0.0

# Quit confirmation state
confirm_quit       = False
confirm_quit_timer = 0.0
CONFIRM_TIMEOUT    = 3.0          # seconds to confirm before it disappears

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

try:
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame  = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int((time.perf_counter() - start_time) * 1000)
            result   = landmarker.detect_for_video(mp_image, ts_ms)

            # ── Skeleton ────────────────────────────────────────────────────
            pose_found = bool(result.pose_landmarks)
            if pose_found:
                lm = result.pose_landmarks[0]
                for a, b in CONNECTIONS:
                    cv2.line(frame,
                             (int(lm[a].x*width), int(lm[a].y*height)),
                             (int(lm[b].x*width), int(lm[b].y*height)),
                             C_ACCENT, 2, cv2.LINE_AA)
                for pt in lm:
                    cv2.circle(frame, (int(pt.x*width), int(pt.y*height)),
                               4, C_WHITE, -1, cv2.LINE_AA)

            # ── Pose analysis ───────────────────────────────────────────────
            angle          = None
            shoulder_angle = None
            feedback       = ""
            is_valid_form  = False
            if pose_found:
                shoulder = lm[12]; elbow = lm[14]; wrist = lm[16]; hip = lm[24]
                if min(shoulder.visibility, elbow.visibility,
                       wrist.visibility, hip.visibility) > 0.5:
                    angle = calculate_angle(
                        (shoulder.x*width, shoulder.y*height),
                        (elbow.x*width,    elbow.y*height),
                        (wrist.x*width,    wrist.y*height),
                    )
                    feedback, is_valid_form, shoulder_angle = check_bicep_curl_posture(
                        shoulder, elbow, wrist, hip
                    )

                    # Only count reps when form is correct
                    if is_valid_form:
                        reps, phase = curl_counter.update(angle)
                    else:
                        reps, phase = curl_counter.reps, curl_counter.phase

                    # Deviation
                    if angle and not (profile["flexion_angle"] - 15 <= angle <= profile["extension_angle"] + 15):
                        deviation_msg   = f"  DEVIATION  {angle:.0f}° outside prescribed range"
                        deviation_timer = time.perf_counter() + 2.5

                    # Session complete (trigger once)
                    if reps >= target_reps and not session_done:
                        session_done = True

            # ── FPS ─────────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.perf_counter() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # ================================================================
            # UI PANELS
            # ================================================================

            # ── Top header bar ───────────────────────────────────────────────
            draw_rounded_rect(frame, 0, 0, width, 48, C_PANEL, alpha=0.88)
            cv2.line(frame, (0, 48), (width, 48), C_ACCENT, 2)

            draw_text(frame, "RehaTrack", 14, 32, FONT_BOLD, 0.9, C_ACCENT, 2)

            user_label = f"{profile['display_name']}   |   {profile['exercise'].replace('_',' ').title()}"
            draw_text(frame, user_label, 170, 32, FONT, 0.65, C_WHITE, 1)

            fps_text = f"FPS {fps:.0f}"
            fps_w, _ = cv2.getTextSize(fps_text, FONT, 0.6, 1)[0]
            draw_text(frame, fps_text, width - fps_w - 14, 32, FONT, 0.6, C_GREY, 1)

            # ── Right side stats panel ───────────────────────────────────────
            px, py = width - 210, 60
            draw_rounded_rect(frame, px, py, width - 8, py + 248, C_PANEL, alpha=0.82)
            cv2.line(frame, (px, py), (px, py + 190), C_ACCENT, 2)

            draw_text(frame, "SESSION",        px+12, py+24,  FONT_BOLD, 0.55, C_ACCENT, 1)
            draw_text(frame, "Reps",           px+12, py+55,  FONT, 0.55, C_GREY, 1)
            rep_str = f"{curl_counter.reps} / {target_reps}"
            draw_text(frame, rep_str,          px+12, py+80,  FONT_BOLD, 0.9, C_WHITE, 2)

            draw_text(frame, "Elbow Angle",    px+12, py+112, FONT, 0.55, C_GREY, 1)
            ang_str = f"{angle:.1f}" + chr(176) if angle else "--"
            ang_col = C_YELLOW if angle else C_GREY
            draw_text(frame, ang_str,          px+12, py+137, FONT_BOLD, 0.9, ang_col, 2)

            draw_text(frame, "Shoulder Angle", px+12, py+165, FONT, 0.55, C_GREY, 1)
            sh_str = f"{shoulder_angle:.1f}" + chr(176) if shoulder_angle else "--"
            sh_col = C_SUCCESS if (shoulder_angle is not None and shoulder_angle <= 50) else C_WARN
            draw_text(frame, sh_str,           px+12, py+188, FONT_BOLD, 0.75, sh_col, 2)

            draw_text(frame, "Phase",          px+12, py+215, FONT, 0.55, C_GREY, 1)
            phase_str = curl_counter.phase.capitalize() if pose_found else "--"
            draw_text(frame, phase_str,        px+12, py+236, FONT, 0.65, C_WHITE, 1)

            # ── Pose status pill ─────────────────────────────────────────────
            pill_col = C_ACCENT if pose_found else (60, 60, 60)
            pill_txt = "Pose Detected" if pose_found else "No Pose"
            draw_rounded_rect(frame, 10, 58, 180, 84, pill_col, alpha=0.75)
            draw_text(frame, pill_txt, 20, 77, FONT, 0.58, C_WHITE, 1)

            # ── Feedback bar ─────────────────────────────────────────────────
            if feedback:
                fb_col = C_SUCCESS if is_valid_form else C_WARN
                fb_bg  = (20, 60, 20) if is_valid_form else (40, 20, 20)
                draw_rounded_rect(frame, 10, 92, 420, 118, fb_bg, alpha=0.85)
                cv2.line(frame, (10, 92), (10, 118), fb_col, 3)
                draw_text(frame, f"Form: {feedback}", 18, 111, FONT, 0.58, fb_col, 1)

            # ── Deviation alert ──────────────────────────────────────────────
            if deviation_msg and time.perf_counter() < deviation_timer:
                draw_rounded_rect(frame, 0, height-52, width, height, C_WARN, alpha=0.88)
                cv2.line(frame, (0, height-52), (width, height-52), (100, 100, 255), 2)
                dw, _ = cv2.getTextSize(deviation_msg, FONT_BOLD, 0.7, 2)[0]
                draw_text(frame, deviation_msg,
                          (width - dw)//2, height-18, FONT_BOLD, 0.7, C_WHITE, 2)

            # ── Session complete overlay ──────────────────────────────────────
            if session_done:
                draw_rounded_rect(frame, width//2 - 220, height//2 - 45,
                                  width//2 + 220, height//2 + 45, C_SUCCESS, alpha=0.9)
                cv2.rectangle(frame, (width//2 - 220, height//2 - 45),
                              (width//2 + 220, height//2 + 45), (80, 220, 80), 2)
                done_txt = f"Session Complete!  {curl_counter.reps} reps"
                dw, _ = cv2.getTextSize(done_txt, FONT_BOLD, 0.85, 2)[0]
                draw_text(frame, done_txt, (width-dw)//2, height//2 + 10,
                          FONT_BOLD, 0.85, C_WHITE, 2)

            # ── Quit confirmation overlay ────────────────────────────────────
            if confirm_quit:
                remaining = CONFIRM_TIMEOUT - (time.perf_counter() - confirm_quit_timer)
                if remaining > 0:
                    draw_rounded_rect(frame, 0, 0, width, height, (0, 0, 0), alpha=0.55)
                    draw_rounded_rect(frame, width//2 - 260, height//2 - 70,
                                      width//2 + 260, height//2 + 70, (30, 30, 30), alpha=0.95)
                    cv2.rectangle(frame, (width//2 - 260, height//2 - 70),
                                  (width//2 + 260, height//2 + 70), C_WARN, 2)

                    draw_text(frame, "Quit RehaTrack?",
                              width//2 - 120, height//2 - 28, FONT_BOLD, 0.85, C_WHITE, 2)
                    draw_text(frame, "Your session progress will not be saved.",
                              width//2 - 195, height//2 + 5, FONT, 0.55, C_GREY, 1)
                    draw_text(frame, f"Press Q again to confirm  ({remaining:.1f}s)",
                              width//2 - 170, height//2 + 42, FONT, 0.6, C_YELLOW, 1)
                else:
                    confirm_quit = False   # timed out — dismiss

            # ── Controls hint ────────────────────────────────────────────────
            hint = "Q / ESC — Quit"
            hw, _ = cv2.getTextSize(hint, FONT, 0.45, 1)[0]
            draw_text(frame, hint, width - hw - 12, height - 10, FONT, 0.45, C_GREY, 1)

            cv2.imshow("RehaTrack", frame)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                if confirm_quit:
                    # Second press — confirmed quit
                    break
                else:
                    # First press — show warning
                    confirm_quit       = True
                    confirm_quit_timer = time.perf_counter()

finally:
    camera.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - start_time
    print("\n╔══════════════════════════════════════╗")
    print("║          Session Summary             ║")
    print("╚══════════════════════════════════════╝")
    print(f"  User     : {profile['display_name']}")
    print(f"  Exercise : {profile['exercise'].replace('_', ' ').title()}")
    print(f"  Reps     : {curl_counter.reps} / {target_reps}")
    print(f"  Duration : {elapsed:.1f}s")
    print(f"  Status   : {'Complete' if session_done else 'Incomplete'}\n")
