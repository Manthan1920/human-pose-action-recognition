"""
dataset.py
----------
Dataset loader for training the optional LSTM action classifier.

Expected data format (create this yourself by running pose estimation on
labeled clips and saving the keypoint sequences):

  data/action_dataset/
    ├── labels.csv                  # columns: sequence_file, label
    └── sequences/
          ├── seq_0001.npy          # shape (T, 17, 3)
          ├── seq_0002.npy
          └── ...

You can generate these .npy files with `src/pose_estimator.py`:
run it over a short labeled video clip, collect the keypoints for one
person across frames, and save with np.save(). This is the standard
way to build a skeleton-action dataset (similar in spirit to NTU RGB+D).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .action_recognizer import ACTIONS
from .utils import normalize_pose


class PoseSequenceDataset(Dataset):
    def __init__(self, csv_path, sequences_dir, seq_len=30):
        self.df = pd.read_csv(csv_path)
        self.sequences_dir = sequences_dir
        self.seq_len = seq_len
        self.label_to_idx = {label: i for i, label in enumerate(ACTIONS)}

    def __len__(self):
        return len(self.df)

    def _pad_or_trim(self, seq):
        t = seq.shape[0]
        if t >= self.seq_len:
            return seq[-self.seq_len:]
        pad = np.repeat(seq[0:1], self.seq_len - t, axis=0)
        return np.concatenate([pad, seq], axis=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.sequences_dir, row["sequence_file"])
        raw = np.load(path)                       # (T, 17, 3)
        raw = self._pad_or_trim(raw)
        normalized = np.stack([normalize_pose(kp) for kp in raw])  # (T, 17, 2)
        flattened = normalized.reshape(self.seq_len, -1)           # (T, 34)
        label_idx = self.label_to_idx[row["label"]]
        return (
            torch.tensor(flattened, dtype=torch.float32),
            torch.tensor(label_idx, dtype=torch.long),
        )
