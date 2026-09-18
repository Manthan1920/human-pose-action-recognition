# Human Pose Estimation & Action Recognition in Crowded Scenes

A working, ready-to-run computer vision project that:
1. Detects **every person** in a video frame (even in dense/crowded scenes).
2. Estimates their **17-keypoint skeleton** (pose).
3. **Tracks** each person across frames (consistent ID).
4. Classifies each person's **action** (Standing / Walking / Running / Sitting / Falling / Waving) from their motion over time.
5. Shows the result in a **Streamlit web app** or saves an annotated output video.

---

## 1. Project structure

```
human_pose_action_recognition/
├── app.py                     # Streamlit web demo (easiest way to test)
├── inference.py                # CLI script: process a video file end-to-end
├── config.yaml                 # All tunable settings in one place
├── requirements.txt
├── Dockerfile                  # Containerized deployment
├── src/
│   ├── pose_estimator.py       # YOLOv8-Pose wrapper (multi-person + tracking)
│   ├── action_recognizer.py    # Rule-based classifier + optional LSTM classifier
│   ├── pipeline.py              # Glues pose + tracking + action recognition together
│   ├── dataset.py               # Dataset loader for training the LSTM (optional)
│   ├── train_action_model.py   # Training script for the LSTM (optional)
│   └── utils.py                 # Skeleton drawing, geometry helpers
├── data/sample_videos/          # Put your test videos here
├── models/                      # Trained LSTM checkpoints go here (optional)
└── outputs/                     # Annotated output videos land here
```

---

## 2. How the pipeline works (step by step)

**Step 1 — Pose Estimation (per frame):**
`YOLOv8-Pose` looks at the raw frame and outputs, for every person it finds:
a bounding box, 17 (x, y, confidence) keypoints, and a detection confidence score.
It's a single model that handles many overlapping people in one pass — important
for crowded scenes where a slower "detect-then-crop-then-pose" approach would
struggle with occlusion and speed.

**Step 2 — Tracking:**
Ultralytics' built-in `ByteTrack` tracker matches detections across consecutive
frames so the same physical person keeps the same `track_id`, even through
brief occlusion. Without this, you cannot measure "movement over time" per person.

**Step 3 — Temporal buffering:**
`PoseBuffer` keeps the last ~30 frames (~1 second) of skeleton history for
every `track_id` in a sliding window (a Python `deque`).

**Step 4 — Action Recognition:**
For each person's buffered history, `RuleBasedActionRecognizer` computes a
few explainable signals:
- torso tilt angle (falling / sitting)
- vertical hip velocity (falling)
- ankle displacement per frame (standing vs walking vs running speed)
- wrist height relative to shoulder (waving)

These are combined with simple thresholds (normalized by shoulder width, so
it works whether the person is near or far from the camera) into one of 7
action labels. **This mode needs zero training data and works immediately.**

An optional `LSTMActionClassifier` (in the same file) can replace the rule
engine once you've collected and labeled your own clips — see Section 5.

**Step 5 — Visualization:**
`utils.py` draws the skeleton, a colored bounding box (color = action), the
track ID, and the action label directly onto the frame.

---

## 3. Setup (do this once)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First run only) YOLOv8-pose weights auto-download the first time
#    the model is used — no manual step needed, just make sure you have
#    internet access once.
```

Put a test video (a few people, ideally a crowd/street/plaza clip) at:
`data/sample_videos/sample.mp4`

---

## 4. Running & checking the model works

### Option A — Streamlit web app (recommended, visual, easiest to demo)
```bash
streamlit run app.py
```
Open the printed URL (usually `http://localhost:8501`). Upload a video, click
**Run**, and you'll see live annotated frames, a running action tally, an FPS
readout, and a download button for the output video — this is your "is it
working" check.

### Option B — Command line (batch process a video file)
```bash
python inference.py --source data/sample_videos/sample.mp4 --output outputs/result.mp4
```
This prints per-30-frame FPS progress and a final summary (frame count, total
time, action counts). Open `outputs/result.mp4` to view the annotated result.

### Option C — Webcam, live preview window
```bash
python inference.py --source 0 --output outputs/webcam_result.mp4 --show
```
A window pops up with live annotated video; press `q` to stop.

### Sanity check without any video
```python
from src.pose_estimator import PoseEstimator
import numpy as np
pe = PoseEstimator(weights="yolov8n-pose.pt", device="cpu")
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
print(pe.infer(dummy))   # should return [] (no error) confirming the model loads & runs
```

---

## 5. (Optional) Training the LSTM for higher accuracy

The rule-based recognizer works out of the box, but it's heuristic. To get a
learned classifier:

1. Collect short labeled clips per action (e.g. 3–5 sec each of walking,
   running, falling, etc.) — you can film your own or use public datasets
   (e.g. NTU RGB+D, UCF101 subsets, Kinetics-Skeleton).
2. For each clip, run `PoseEstimator` and save the keypoint sequence for one
   person as a `.npy` file of shape `(T, 17, 3)`.
3. Build `data/action_dataset/labels.csv` with columns `sequence_file,label`.
4. Train:
   ```bash
   python -m src.train_action_model \
       --csv data/action_dataset/labels.csv \
       --seq_dir data/action_dataset/sequences \
       --epochs 30 --out models/action_lstm.pt
   ```
5. In `config.yaml`, set `action_recognition.mode: "lstm"`. The pipeline will
   automatically pick up `models/action_lstm.pt`.

---

## 6. Deployment options

| Option | Best for | Steps |
|---|---|---|
| **Local run** | Development / class demo | `streamlit run app.py` |
| **Docker** | Reproducible deployment anywhere | `docker build -t crowd-pose .` then `docker run -p 8501:8501 crowd-pose` |
| **Streamlit Community Cloud** | Free public link to share/submit | Push repo to GitHub → streamlit.io/cloud → "New app" → pick repo/branch/`app.py` |
| **Hugging Face Spaces** | Free public link, GPU option | Create a Space (SDK: Streamlit or Docker) → push this repo → auto-builds |
| **Cloud VM (AWS EC2 / GCP / Azure)** | Needs GPU / production-like | Provision VM → clone repo → `pip install -r requirements.txt` → run with `nohup streamlit run app.py &` or the Docker image, open the security-group port |
| **REST API** | Integrating into another app | Wrap `CrowdPoseActionPipeline.process_frame` in a FastAPI/Flask endpoint that accepts an image and returns JSON results |

**Fastest path to a shareable link (recommended for a college project submission):**
GitHub → Streamlit Community Cloud (free, ~5 minutes, gives you a public URL
your instructor can open directly).

---

## 7. Notes on "crowded scenes" specifically

- Lower `iou_threshold` in `config.yaml` if overlapping people get merged into
  one detection.
- Use `yolov8m-pose.pt` or larger if small/far-away people are missed (trade-off:
  slower).
- `resize_width` in config controls speed vs. accuracy trade-off — smaller
  frames run faster but may miss small people.
- If IDs "switch" between two people who cross paths, try `botsort.yaml` as the
  tracker (set in `config.yaml`) — it uses appearance features and is more
  robust in dense crowds than ByteTrack, at some speed cost.
