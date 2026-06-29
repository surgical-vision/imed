#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imed_dataset_loader import IMEDDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("sequence_id")
    args = parser.parse_args()
    dataset = IMEDDataset(args.root)
    loaded = dataset.load_trajectory(args.sequence_id)
    traj = loaded["trajectory"]
    metadata = loaded["metadata"]
    processing_info = metadata.get("processing_info")
    if not isinstance(processing_info, dict):
        processing_info = {}
    pose_source = _format_metadata_value(metadata.get("pose_source") or processing_info.get("pose_source"))
    aruco_dict = (
        metadata.get("aruco_tag_dictionary")
        or metadata.get("aruco_dictionary")
        or processing_info.get("aruco_tag_dictionary")
    )
    aruco_dict = _format_metadata_value(aruco_dict)
    print(
        f"{args.sequence_id} rows={traj.shape[0] if traj.shape else 1} "
        f"fields={len(traj.dtype.names or [])} frame_min={traj['frame_idx'].min():.0f} "
        f"frame_max={traj['frame_idx'].max():.0f} pose_source={pose_source} aruco_dictionary={aruco_dict}"
    )
    return 0


def _format_metadata_value(value: object) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
