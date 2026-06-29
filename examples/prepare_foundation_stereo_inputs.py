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
    parser.add_argument("--camera", default="endoscope1", choices=["endoscope1", "endoscope2"])
    parser.add_argument("--raw", action="store_true", help="read endoscope*_raw.mp4 instead of processed video")
    args = parser.parse_args()

    dataset = IMEDDataset(args.root)
    inputs = dataset.get_foundation_stereo_inputs(
        args.sequence_id,
        args.frame_idx,
        camera=args.camera,
        raw=args.raw,
    )
    config = inputs["foundation_stereo_depth_config"]
    source = "raw" if args.raw else "processed"
    print(
        f"{args.sequence_id} {args.camera} source={source} frame={args.frame_idx} "
        f"pair={inputs['camera_pair']} left_shape={inputs['left'].shape} right_shape={inputs['right'].shape} "
        f"baseline_m={inputs['baseline_m']:.6f} focal_px={inputs['focal_length_px']:.3f} "
        f"method={config['method']} scale={config['scale']} pad_px={config['pad_amount_px']} "
        f"depth_clip_m={config['depth_clip_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
