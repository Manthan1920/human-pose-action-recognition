"""
pipeline.py
-----------
The full pipeline for ONE frame:
   frame -> PoseEstimator (detect + track all people)
         -> PoseBuffer (store history per track ID)
         -> ActionRecognizer (classify action per person)
         -> draw results on frame

This is imported by both inference.py (CLI/video file) and app.py (Streamlit UI),
so the logic lives in exactly one place.
"""

import os
import yaml
import cv2

from .pose_estimator import PoseEstimator
from .action_recognizer import PoseBuffer, RuleBasedActionRecognizer, LSTMActionRecognizer
from .utils import draw_skeleton, draw_label, ACTION_COLORS


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class CrowdPoseActionPipeline:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)

        pm = self.cfg["pose_model"]
        self.pose_estimator = PoseEstimator(
            weights=pm["weights"],
            conf_threshold=pm["conf_threshold"],
            iou_threshold=pm["iou_threshold"],
            device=pm["device"],
            tracker=f"{self.cfg['tracker']['type']}.yaml",
        )

        ar = self.cfg["action_recognition"]
        self.buffer = PoseBuffer(buffer_size=ar["buffer_size"])

        if ar["mode"] == "lstm" and os.path.exists(ar["lstm_checkpoint"]):
            self.action_recognizer = LSTMActionRecognizer(
                ar["lstm_checkpoint"], min_frames=ar["min_frames_for_decision"]
            )
            print("[Pipeline] Using trained LSTM action recognizer.")
        else:
            self.action_recognizer = RuleBasedActionRecognizer(
                min_frames=ar["min_frames_for_decision"]
            )
            print("[Pipeline] Using rule-based action recognizer (no training needed).")

        self.video_cfg = self.cfg["video"]

    def process_frame(self, frame):
        """
        Runs the full pipeline on a single frame.
        Returns: (annotated_frame, list_of_dicts)
        Each dict: {track_id, bbox, action, conf}
        """
        h, w = frame.shape[:2]
        resize_w = self.video_cfg.get("resize_width")
        if resize_w and w > resize_w:
            scale = resize_w / w
            frame = cv2.resize(frame, (resize_w, int(h * scale)))

        detections = self.pose_estimator.infer(frame)
        active_ids = set()
        results = []

        for det in detections:
            if det.track_id == -1:
                continue  # tracker hasn't assigned an ID yet, skip for temporal logic
            active_ids.add(det.track_id)
            self.buffer.update(det.track_id, det.keypoints)
            history = self.buffer.get(det.track_id)
            action = self.action_recognizer.classify(history)

            if self.video_cfg.get("draw_skeleton", True):
                color = ACTION_COLORS.get(action, (0, 255, 0))
                draw_skeleton(frame, det.keypoints, color=color)
            if self.video_cfg.get("draw_ids", True) or self.video_cfg.get("draw_action_label", True):
                draw_label(frame, det.bbox, det.track_id, action)

            results.append({
                "track_id": det.track_id,
                "bbox": det.bbox,
                "action": action,
                "conf": det.conf,
            })

        self.buffer.cleanup(active_ids)
        return frame, results
