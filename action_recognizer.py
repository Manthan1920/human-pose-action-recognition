"""
action_recognizer.py
---------------------
Turns a TIME SEQUENCE of skeletons (per tracked person) into an action label.

Two modes are provided:

1. RuleBasedActionRecognizer  (default, works instantly, no training data needed)
   Uses simple, explainable geometry + motion rules:
     - torso tilt angle       -> falling detection
     - vertical hip velocity  -> falling / sitting
     - ankle displacement/sec -> standing vs walking vs running
     - wrist-above-shoulder   -> waving

2. LSTMActionClassifier (optional, higher accuracy, needs a trained checkpoint)
   A small LSTM over normalized keypoint sequences -> softmax over action classes.
   Train it with src/train_action_model.py on your own labeled clips.

Both share a PoseBuffer that stores each tracked person's recent keypoint
history (a sliding window), because you cannot recognize an "action"
from a single frame -- you need motion over time.
"""

from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn

from .utils import KP, normalize_pose, euclidean, angle_between

ACTIONS = ["Standing", "Walking", "Running", "Sitting", "Falling", "Waving", "Unknown"]


# ------------------------------------------------------------------
# 1. Per-person temporal buffer
# ------------------------------------------------------------------
class PoseBuffer:
    """Keeps the last N frames of raw keypoints for every track_id."""

    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.buffers = defaultdict(lambda: deque(maxlen=buffer_size))

    def update(self, track_id, keypoints):
        self.buffers[track_id].append(np.array(keypoints, dtype=np.float32))

    def get(self, track_id):
        return list(self.buffers[track_id])

    def cleanup(self, active_ids):
        """Drop buffers for people who left the frame (saves memory)."""
        stale = [tid for tid in self.buffers if tid not in active_ids]
        for tid in stale:
            del self.buffers[tid]


# ------------------------------------------------------------------
# 2. Rule-based classifier (works out-of-the-box)
# ------------------------------------------------------------------
class RuleBasedActionRecognizer:
    def __init__(self, min_frames=10, conf_thresh=0.2):
        self.min_frames = min_frames
        self.conf_thresh = conf_thresh

    def classify(self, history):
        """
        history: list of keypoint arrays (each shape (17,3)), oldest -> newest
        Returns one of ACTIONS.
        """
        if len(history) < self.min_frames:
            return "Unknown"

        first, last = history[0], history[-1]

        def valid(kp, idx):
            return kp[idx][2] > self.conf_thresh

        # ---- Hip center vertical velocity (falling/sitting signal) ----
        hip_ys = []
        for kp in history:
            if valid(kp, KP["l_hip"]) and valid(kp, KP["r_hip"]):
                hip_ys.append((kp[KP["l_hip"]][1] + kp[KP["r_hip"]][1]) / 2.0)
        vertical_drop = (hip_ys[-1] - hip_ys[0]) if len(hip_ys) >= 2 else 0.0

        # ---- Torso tilt angle (shoulder-hip vs vertical) ----
        torso_angle = 90.0
        if all(valid(last, k) for k in [KP["l_shoulder"], KP["r_shoulder"], KP["l_hip"], KP["r_hip"]]):
            sh_c = (np.array(last[KP["l_shoulder"]][:2]) + np.array(last[KP["r_shoulder"]][:2])) / 2
            hip_c = (np.array(last[KP["l_hip"]][:2]) + np.array(last[KP["r_hip"]][:2])) / 2
            vertical_ref = hip_c + np.array([0, -100])  # a point straight above hip
            torso_angle = angle_between(sh_c, hip_c, vertical_ref)

        # ---- Ankle motion magnitude per frame (walking/running speed) ----
        ankle_speeds = []
        for a, b in zip(history[:-1], history[1:]):
            if valid(a, KP["l_ankle"]) and valid(b, KP["l_ankle"]):
                ankle_speeds.append(euclidean(a[KP["l_ankle"]][:2], b[KP["l_ankle"]][:2]))
            if valid(a, KP["r_ankle"]) and valid(b, KP["r_ankle"]):
                ankle_speeds.append(euclidean(a[KP["r_ankle"]][:2], b[KP["r_ankle"]][:2]))
        avg_ankle_speed = float(np.mean(ankle_speeds)) if ankle_speeds else 0.0

        # ---- Wrist above shoulder (waving signal) ----
        waving = False
        if valid(last, KP["l_wrist"]) and valid(last, KP["l_shoulder"]):
            waving = waving or (last[KP["l_wrist"]][1] < last[KP["l_shoulder"]][1] - 15)
        if valid(last, KP["r_wrist"]) and valid(last, KP["r_shoulder"]):
            waving = waving or (last[KP["r_wrist"]][1] < last[KP["r_shoulder"]][1] - 15)

        # Rough scale reference (shoulder width) to normalize thresholds across
        # people at different distances from the camera.
        scale = 60.0
        if all(valid(last, k) for k in [KP["l_shoulder"], KP["r_shoulder"]]):
            scale = max(euclidean(last[KP["l_shoulder"]][:2], last[KP["r_shoulder"]][:2]), 20.0)

        # ---------------- Decision rules (checked in priority order) ----------------
        if vertical_drop > 0.9 * scale and torso_angle > 45:
            return "Falling"
        if torso_angle > 55:
            return "Sitting"
        if waving and avg_ankle_speed < 0.15 * scale:
            return "Waving"
        if avg_ankle_speed > 0.35 * scale:
            return "Running"
        if avg_ankle_speed > 0.08 * scale:
            return "Walking"
        return "Standing"


# ------------------------------------------------------------------
# 3. Optional deep-learning classifier (train on your own data)
# ------------------------------------------------------------------
class LSTMActionClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=len(ACTIONS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=0.3 if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, 34)  -- 17 keypoints * (x, y)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


class LSTMActionRecognizer:
    """Loads a trained LSTMActionClassifier checkpoint and runs inference."""

    def __init__(self, checkpoint_path, device="cpu", min_frames=10):
        self.device = device
        self.min_frames = min_frames
        self.model = LSTMActionClassifier().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def classify(self, history):
        if len(history) < self.min_frames:
            return "Unknown"
        seq = np.stack([normalize_pose(kp).flatten() for kp in history])  # (T, 34)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        pred = int(torch.argmax(logits, dim=1).item())
        return ACTIONS[pred]
