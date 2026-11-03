#!/usr/bin/env python3
"""Validate an extracted iMED public dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


TEXT_SUFFIXES = {".json", ".md", ".txt", ".csv", ".py", ".sh", ".yaml", ".yml"}
FORBIDDEN_TEXT = (
    "/" + "raid" + "/",
    "/" + "home" + "/",
    "file" + "://",
    "~" + "/",
    "hidden" + "_test",
    "private" + "_test",
    "train" + "_test" + "_split",
    "benchmark" + "_test",
    "with" + "held",
    "check" + "point",
    "diag" + "nostics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Path to an extracted iMED release")
    parser.add_argument("--quick", action="store_true", help="Skip opening every video with OpenCV.")
    parser.add_argument("--verify-checksums", action="store_true", help="Check every file against the v2 release manifest.")
    return parser.parse_args()


def error(errors: list[str], message: str) -> None:
    errors.append(message)


def scan_text_files(root: Path, errors: list[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in FORBIDDEN_TEXT:
            if token.lower() in text:
                error(errors, f"forbidden text {token!r} found in {path}")


def validate_metadata(root: Path, errors: list[str]) -> None:
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        error(errors, f"missing {metadata_path}")
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        error(errors, f"could not parse {metadata_path}: {exc}")
        return
    try:
        baseline = float(metadata.get("scope_baseline_mm"))
    except (TypeError, ValueError):
        error(errors, "dataset_metadata.json must contain numeric scope_baseline_mm")
        return
    if abs(baseline - 4.1) > 1e-6:
        error(errors, "scope_baseline_mm must be 4.1")
    sessions = metadata.get("sessions")
    if isinstance(sessions, list):
        for session_id in sessions:
            path = root / str(session_id) / "session_metadata.json"
            if not path.is_file():
                error(errors, f"missing session metadata {path}")
                continue
            try:
                session_metadata = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                error(errors, f"could not parse {path}: {exc}")
                continue
            try:
                float(session_metadata.get("frame_rate"))
            except (TypeError, ValueError):
                error(errors, f"{path} must contain numeric frame_rate")
            if not session_metadata.get("anatomical_regions"):
                error(errors, f"{path} must contain anatomical_regions")


def validate_layout(root: Path, errors: list[str]) -> int:
    calibration_sequences = set()
    for path in root.glob('session_*/session_metadata.json'):
        session = json.loads(path.read_text())
        for region in session.get('anatomical_regions', []):
            for sequence in region.get('sequences', []):
                if sequence.get('movement_type') == 'aruco_marker_tracking':
                    calibration_sequences.add(sequence['sequence_id'])
    sequence_dirs = sorted(path for path in root.glob("session_*/*/*") if path.is_dir())
    if not sequence_dirs:
        error(errors, "no sequence directories found under session_*/*/*")
        return 0
    for seq_dir in sequence_dirs:
        moving_aruco = ("moving_aruco" in seq_dir.name
                       or str(seq_dir.relative_to(root)) in calibration_sequences)
        required = ["endoscope1_raw.mp4", "endoscope2_raw.mp4"]
        if not moving_aruco:
            required.extend(["endoscope1.mp4", "endoscope2.mp4"])
        for name in required:
            if not (seq_dir / name).is_file():
                error(errors, f"missing required file {seq_dir / name}")
        if moving_aruco:
            for name in ("endoscope1.mp4", "endoscope2.mp4"):
                if (seq_dir / name).exists():
                    error(errors, f"moving-ArUco calibration sequence should be raw-only: {seq_dir / name}")
            for name in ("endoscope1_tools.mp4", "endoscope2_tools.mp4"):
                if (seq_dir / name).exists():
                    error(errors, f"moving-ArUco calibration sequence should not include tool masks: {seq_dir / name}")
        if not list(seq_dir.glob("*_filenames.csv")):
            error(errors, f"missing timestamp CSV in {seq_dir}")
        if (seq_dir / "endoscope1_aruco.mp4").exists() or (seq_dir / "endoscope2_aruco.mp4").exists():
            error(errors, f"ArUco mask videos must not be included in {seq_dir}")
    return len(sequence_dirs)


def validate_trajectories(root: Path, errors: list[str]) -> int:
    count = 0
    for pose_json in root.rglob("relative_poses.json"):
        count += 1
        try:
            data = json.loads(pose_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            error(errors, f"could not parse {pose_json}: {exc}")
            continue
        trajectory_file = data.get("trajectory_file")
        if not isinstance(trajectory_file, str):
            error(errors, f"{pose_json} missing string trajectory_file")
            continue
        target = pose_json.parent / trajectory_file
        if not target.is_file():
            error(errors, f"trajectory file referenced by {pose_json} is missing: {target}")
    return count


def validate_videos(root: Path, errors: list[str], quick: bool) -> int:
    videos = sorted(root.rglob("*.mp4"))
    if quick:
        return len(videos)
    import cv2
    session_fps = {}
    frame_counts = {}
    for metadata_path in root.glob("session_*/session_metadata.json"):
        try:
            session = json.loads(metadata_path.read_text(encoding="utf-8"))
            session_fps[metadata_path.parent.name] = float(session['frame_rate'])
            for region in session.get('anatomical_regions', []):
                for sequence in region.get('sequences', []):
                    if 'frame_count' in sequence:
                        frame_counts[sequence['sequence_id']] = sequence['frame_count']
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error(errors, f"could not read frame_rate from {metadata_path}: {exc}")
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                error(errors, f"OpenCV could not open {video}")
                continue
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if min(width, height, frames, fps) <= 0:
                error(errors, f"invalid video metadata for {video}")
            sequence_id = str(video.parent.relative_to(root))
            if sequence_id in frame_counts and (frames != frame_counts[sequence_id] or (width, height) != (1280, 2048)):
                error(errors, f"frame count or native resolution mismatch: {video}")
            try:
                session_id = video.relative_to(root).parts[0]
            except ValueError:
                session_id = ""
            expected_fps = session_fps.get(session_id)
            if expected_fps is not None and abs(fps - expected_fps) > 0.01:
                error(errors, f"video FPS {fps:.6f} does not match {session_id} frame_rate {expected_fps:.6f}: {video}")
        finally:
            cap.release()
    return len(videos)


def validate_checksums(root: Path, errors: list[str]) -> None:
    path = root / 'release_manifest.json'
    if not path.is_file():
        error(errors, 'No release_manifest.json available for checksum verification')
        return
    entries = json.loads(path.read_text())['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(entries) | {'release_manifest.json'}:
        error(errors, 'Release file list differs from the manifest')
    for name, record in entries.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            error(errors, f'Invalid manifest path: {name}')
            continue
        target = root / relative
        if not target.is_file() or target.is_symlink():
            error(errors, f'Missing or linked manifest file: {name}')
            continue
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if target.stat().st_size != record['bytes'] or digest != record['sha256']:
            error(errors, f'Checksum or size mismatch: {name}')


def main() -> int:
    args = parse_args()
    root = args.root
    errors: list[str] = []
    if not root.is_dir():
        print(f"dataset root not found: {root}", file=sys.stderr)
        return 2
    scan_text_files(root, errors)
    validate_metadata(root, errors)
    sequence_count = validate_layout(root, errors)
    trajectory_count = validate_trajectories(root, errors)
    video_count = validate_videos(root, errors, args.quick)
    if args.verify_checksums:
        validate_checksums(root, errors)
    if errors:
        print("Validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1
    print("Validation passed")
    print(f"sequences={sequence_count} videos={video_count} trajectory_json={trajectory_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
