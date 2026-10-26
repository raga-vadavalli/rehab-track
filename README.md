# rehab_track

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Learn Flutter](https://docs.flutter.dev/get-started/learn-flutter)
- [Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Flutter learning resources](https://docs.flutter.dev/reference/learning-resources)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

# RehaTrack-ML

Machine learning experiments for RehaTrack, a rehabilitation application focused on exercise tracking and movement analysis.

## Current Work

- **Pose estimation:** MediaPipe Pose Landmarker Lite
- **Exercise tracking:** Bicep curl repetition counting
- **Joint analysis:** Shoulder, elbow, and wrist landmark tracking
- **Angle calculation:** Elbow angle estimation during exercise
- **Data collection:** Scripts for collecting exercise data for future ML experiments

## Project Structure

```text
ml-experiments/
├── bicep_curl.py
├── pose_webcam.py
├── requirements.txt
└── models/
    ├── collect_data.py
    └── pose_landmarker_lite.task
```

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml-experiments\requirements.txt
```

## Running the Experiments

Run the webcam pose detection experiment:

```powershell
python ml-experiments\pose_webcam.py
```

Run the bicep curl experiment:

```powershell
python ml-experiments\bicep_curl.py
```

Run the data collection script:

```powershell
python ml-experiments\models\collect_data.py
```

## Current Goals

- Improve exercise repetition counting.
- Collect labelled exercise movement data.
- Explore posture feedback and movement classification.
- Evaluate suitable ML approaches for rehabilitation tracking.

## Notes

These experiments are part of the ongoing development of RehaTrack. Features and implementation details may change as the project evolves.
