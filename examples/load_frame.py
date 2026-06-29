#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imed_dataset_loader import IMEDDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("sequence_id")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--camera", default="endoscope1")
    parser.add_argument("--raw", action="store_true", help="read endoscope*_raw.mp4 instead of processed endoscope*.mp4")
    args = parser.parse_args()
    dataset = IMEDDataset(args.root)
    top, bottom = dataset.get_frame(args.sequence_id, args.frame_idx, camera=args.camera, raw=args.raw)
    kind = "raw" if args.raw else "processed"
    video_path = dataset.get_video_path(args.sequence_id, camera=args.camera, raw=args.raw)
    print(
        f"{args.sequence_id} {args.camera} kind={kind} frame={args.frame_idx} "
        f"left_top_shape={top.shape} right_bottom_shape={bottom.shape} video={video_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
