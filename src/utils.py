"""
utils.py
--------
Shared helper functions:
  - COCO 17-keypoint skeleton definition
  - Drawing functions (skeleton, bbox, ID, action label)
  - Simple geometry helpers (angle between 3 points, distance, normalization)

COCO keypoint order (used by YOLOv8-pose):
 0 Nose        1 L-Eye       2 R-Eye       3 L-Ear       4 R-Ear
 5 L-Shoulder  6 R-Shoulder  7 L-Elbow     8 R-Elbow     9 L-Wrist
10 R-Wrist    11 L-Hip      12 R-Hip      13 L-Knee     14 R-Knee
15 L-Ankle    16 R-Ankle
"""

import cv2
import numpy as np

# Pairs of keypoint indices that form the skeleton "bones"
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

ACTION_COLORS = {
    "Standing": (80, 200, 80),
    "Walking": (255, 191, 0),
    "Running": (0, 128, 255),
    "Sitting": (200, 160, 40),
    "Falling": (0, 0, 255),
    "Waving": (200, 0, 200),
    "Unknown": (160, 160, 160),
}


def angle_between(p1, p2, p3):
    """Angle (degrees) at point p2, formed by p1-p2-p3."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    cos_angle = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def bbox_from_keypoints(keypoints, conf_thresh=0.1):
    """Return (x1, y1, x2, y2) bounding box from valid keypoints."""
    valid = [(x, y) for x, y, c in keypoints if c > conf_thresh]
    if not valid:
        return None
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    return min(xs), min(ys), max(xs), max(ys)


def draw_skeleton(frame, keypoints, color=(0, 255, 0), conf_thresh=0.25):
    """Draw skeleton joints + bones on a frame. keypoints: list of (x, y, conf)."""
    for (i, j) in SKELETON_EDGES:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci > conf_thresh and cj > conf_thresh:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2)
    for (x, y, c) in keypoints:
        if c > conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
    return frame


def draw_label(frame, bbox, track_id, action, color=None):
    """Draw bounding box + track ID + action label above the person."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = color or ACTION_COLORS.get(action, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}: {action}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def normalize_pose(keypoints):
    """
    Normalize a keypoint array relative to hip-center and torso scale.
    Makes the pose invariant to the person's position and distance from camera.
    keypoints: np.array shape (17, 3) -> (x, y, conf)
    Returns np.array shape (17, 2) normalized (x, y).
    """
    kps = np.array(keypoints, dtype=np.float32)
    l_hip, r_hip = kps[KP["l_hip"]][:2], kps[KP["r_hip"]][:2]
    l_sh, r_sh = kps[KP["l_shoulder"]][:2], kps[KP["r_shoulder"]][:2]
    hip_center = (l_hip + r_hip) / 2.0
    shoulder_center = (l_sh + r_sh) / 2.0
    scale = euclidean(hip_center, shoulder_center)
    scale = scale if scale > 1e-3 else 1.0
    normalized = (kps[:, :2] - hip_center) / scale
    return normalized
