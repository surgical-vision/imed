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
    args = parser.parse_args()
    dataset = IMEDDataset(args.root)
    print(f"sessions: {len(dataset.sessions)}")
    for session_id in dataset.sessions:
        print(f"  {session_id}")
    print(f"scenes: {len(dataset.scenes)}")
    print(f"sequences: {len(dataset.sequences)}")
    for sequence_id in sorted(dataset.sequences):
        sequence = dataset.sequences[sequence_id]
        processed = ",".join(sorted(sequence.videos)) or "-"
        raw = ",".join(sorted(sequence.raw_videos)) or "-"
        tools = ",".join(sorted(sequence.tool_masks)) or "-"
        pose_source = _format_metadata_value(sequence.pose_source)
        aruco_dict = _format_metadata_value(sequence.aruco_tag_dictionary)
        print(
            f"  {sequence_id} processed={processed} raw={raw} "
            f"tool_masks={tools} pose_source={pose_source} aruco_dictionary={aruco_dict}"
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
