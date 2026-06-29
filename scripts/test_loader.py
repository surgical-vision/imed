#!/usr/bin/env python3
"""Test the public iMED loader API against an extracted release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from imed_dataset_loader import IMEDDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Path to imed_v1_0")
    parser.add_argument("--sequence-id", default="session_007/scene_6/tool_2")
    parser.add_argument("--frame-idx", type=int, default=0)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    dataset = IMEDDataset(args.root)

    require(len(dataset.sessions) == 7, "expected 7 released sessions")
    require(len(dataset.scenes) == 7, "expected 7 released scenes")
    require(len(dataset.sequences) == 51, "expected 51 released sequences")
    require(dataset.get_scope_baseline("mm") == 4.1, "scope baseline mismatch")
    require(abs(dataset.get_scope_baseline("m") - 0.0041) < 1e-12, "scope baseline mismatch")

    dataset_metadata = dataset.get_dataset_metadata()
    require("public_release_includes" in dataset_metadata, "dataset metadata missing release fields")
    for session_id in dataset.sessions:
        session_metadata = dataset.get_session_metadata(session_id)
        require("frame_rate" in session_metadata, f"{session_id} missing frame_rate")
        require(dataset.get_frame_rate(session_id) in {30.0, 60.0}, f"{session_id} unexpected frame_rate")
        require(session_metadata.get("anatomical_regions"), f"{session_id} missing anatomical regions")
        dataset.get_intrinsics(session_id)
        dataset.get_calibration(session_id)
        dataset.get_stereo_intrinsics(session_id, camera="endoscope1")
        dataset.get_stereo_intrinsics(session_id, camera="endoscope2")

    sequence = dataset.get_sequence(args.sequence_id)
    require(sequence.metadata, "sequence metadata missing")
    require(sequence.region_metadata, "region metadata missing")
    require(sequence.session_metadata, "session metadata missing")
    require(dataset.get_sequence_metadata(args.sequence_id)["movement_type"] == sequence.metadata["movement_type"], "metadata API mismatch")
    dataset.get_scene_metadata(sequence.scene_id if "/" in sequence.scene_id else f"{sequence.session_id}/{sequence.scene_id}")
    dataset.read_timestamps(args.sequence_id)

    for raw in (False, True):
        frame = dataset.get_frame(args.sequence_id, args.frame_idx, raw=raw, split=False)
        require(frame.ndim == 3, "stacked frame should be HWC")
        left, right = dataset.get_frame(args.sequence_id, args.frame_idx, raw=raw)
        require(left.shape == right.shape, "stereo split shapes differ")
        frame_dict = dataset.get_frame_from_video(args.sequence_id, args.frame_idx, raw=raw)
        require({"L", "R", "SL", "SR"}.issubset(frame_dict), "frame dict missing views")
        inputs = dataset.get_foundation_stereo_inputs(args.sequence_id, args.frame_idx, raw=raw)
        require(inputs["left"].shape == inputs["right"].shape, "FoundationStereo input shapes differ")

    dataset.get_video_path(args.sequence_id, camera="endoscope1")
    dataset.get_raw_video_path(args.sequence_id, camera="endoscope1")
    dataset.get_depth_benchmark_config()
    dataset.load_trajectory(args.sequence_id)

    camera_motion = ["circular_camera_motion", "zoom_camera_motion", "lateral_camera_motion"]
    checks = {
        "ex_vivo_rigid_nvs": dataset.filter(session_types=["ex_vivo"], movement_types=camera_motion),
        "postmortem_rigid_nvs": dataset.filter(session_types=["in_vivo_postmortem"], movement_types=camera_motion),
        "live_camera_motion": dataset.filter(session_types=["in_vivo_live"], movement_types=camera_motion),
        "postmortem_deformable": dataset.filter(
            session_types=["in_vivo_postmortem"],
            movement_types=["off_camera_manipulation", "tool_manipulation"],
        ),
        "live_marker_free": dataset.filter(session_types=["in_vivo_live"], aruco_present=False),
        "tool_sequences": dataset.filter(tools_present=True),
        "energy_sequences": dataset.filter(energy_used=True),
        "cautery_sequences": dataset.filter(tool_combinations=["cautery_hook"]),
        "trajectory_sequences": dataset.filter(has_trajectory=True),
        "tool_mask_sequences": dataset.filter(has_tool_masks=True),
        "kidney_sequences": dataset.filter(organ_types=["kidney"]),
        "task_rigid_nvs": dataset.get_task_sequences("rigid_nvs"),
        "task_deformable_nvs": dataset.get_task_sequences("deformable_nvs"),
        "task_pose_estimation": dataset.get_task_sequences("pose_estimation"),
        "task_monocular_depth": dataset.get_task_sequences("monocular_depth"),
        "task_temporal_stereo_pose": dataset.get_task_sequences("temporal_stereo_pose"),
    }
    for name, results in checks.items():
        require(results, f"filter returned no sequences: {name}")

    print("Loader test passed")
    print(f"sessions={len(dataset.sessions)} scenes={len(dataset.scenes)} sequences={len(dataset.sequences)}")
    for name, results in sorted(checks.items()):
        print(f"{name}={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
