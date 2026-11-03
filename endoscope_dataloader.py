"""Small video helpers for stacked stereo iMED endoscope videos."""

from __future__ import annotations

from pathlib import Path
from numbers import Integral

import cv2


def read_video_frame(video_path: str | Path, frame_idx: int):
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not isinstance(frame_idx, Integral) or frame_idx < 0:
        raise ValueError("frame_idx must be a non-negative integer")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
            raise RuntimeError(f"Could not seek to frame {frame_idx} in {path}")
        ok, frame = cap.read()
        if not ok:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raise IndexError(f"Could not read frame {frame_idx} from {path}; frame_count={count}")
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_idx + 1:
            raise RuntimeError(f"Decoder returned the wrong frame position for {frame_idx} in {path}")
        return frame
    finally:
        cap.release()


def split_stacked_stereo(frame, convention: str = "vertical"):
    """Split a stacked stereo frame into first and second views."""
    if convention not in {"vertical", "horizontal"}:
        raise ValueError("convention must be 'vertical' or 'horizontal'")
    height, width = frame.shape[:2]
    if convention == "vertical":
        if height % 2:
            raise ValueError(f"Cannot vertically split odd frame height {height}")
        midpoint = height // 2
        return frame[:midpoint], frame[midpoint:]
    if width % 2:
        raise ValueError(f"Cannot horizontally split odd frame width {width}")
    midpoint = width // 2
    return frame[:, :midpoint], frame[:, midpoint:]
