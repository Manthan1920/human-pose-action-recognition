"""
pose_estimator.py
-----------------
Wraps YOLOv8-Pose (Ultralytics) for MULTI-PERSON pose estimation with
built-in tracking (ByteTrack/BoT-SORT). This is the "detector" stage of
the pipeline.

Why YOLOv8-Pose for crowded scenes?
  - It's a single-shot, top-down-free (bottom-up + detection) model that
    scales well to many people in one frame, which is exactly what
    dense/crowded scenes need.
  - Ultralytics ships built-in multi-object tracking, so we get consistent
    person IDs across frames for free (needed for temporal action recognition).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Ultralytics not installed. Run: pip install ultralytics"
    ) from e


@dataclass
class PersonDetection:
    track_id: int
    bbox: tuple            # (x1, y1, x2, y2)
    keypoints: np.ndarray  # shape (17, 3) -> x, y, confidence
    conf: float


class PoseEstimator:
    def __init__(self, weights="yolov8s-pose.pt", conf_threshold=0.35,
                 iou_threshold=0.45, device="cpu", tracker="bytetrack.yaml"):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.tracker = tracker

    def infer(self, frame) -> List[PersonDetection]:
        """
        Run pose estimation + tracking on a single BGR frame.
        Returns a list of PersonDetection, one per detected/tracked person.
        """
        results = self.model.track(
            source=frame,
            persist=True,                 # keep track state across calls
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            tracker=self.tracker,
            verbose=False,
        )

        detections: List[PersonDetection] = []
        if not results:
            return detections

        r = results[0]
        if r.keypoints is None or r.boxes is None:
            return detections

        boxes = r.boxes
        kpts = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
        xyxy = boxes.xyxy.cpu().numpy()         # (N, 4)
        confs = boxes.conf.cpu().numpy()        # (N,)
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i in range(len(xyxy)):
            track_id = int(ids[i]) if ids is not None else -1
            detections.append(PersonDetection(
                track_id=track_id,
                bbox=tuple(xyxy[i].tolist()),
                keypoints=kpts[i],
                conf=float(confs[i]),
            ))
        return detections
