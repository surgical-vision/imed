"""Minimal public loader for the iMED first public subset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from endoscope_dataloader import read_video_frame, split_stacked_stereo


SCOPE_BASELINE_MM = 4.1
SCOPE_BASELINE_M = SCOPE_BASELINE_MM / 1000.0
STEREO_CAMERA_PAIRS = {
    "endoscope1": ("L", "R"),
    "endoscope2": ("SL", "SR"),
}
FOUNDATION_STEREO_DEPTH_CONFIG = {
    "method": "FoundationStereo",
    "baseline_m": SCOPE_BASELINE_M,
    "scale": 0.5,
    "pad_amount_px": 90,
    "valid_iters": 32,
    "depth_clip_m": [0.002, 0.2],
    "input_color": "BGR arrays from OpenCV; benchmark script converts BGR to RGB before model inference",
}


@dataclass(frozen=True)
class IMEDSequence:
    sequence_id: str
    session_id: str
    scene_id: str
    sequence_name: str
    path: Path
    timestamp_csv: Path | None
    videos: dict[str, Path]
    raw_videos: dict[str, Path]
    tool_masks: dict[str, Path]
    masks: dict[str, Path]
    relative_pose_file: Path | None
    trajectory_pose_file: Path | None
    metadata: dict[str, object]
    region_metadata: dict[str, object]
    session_metadata: dict[str, object]
    pose_source: object | None
    aruco_tag_dictionary: object | None

    def processed_video_path(self, camera: str = "endoscope1") -> Path:
        """Return the processed public video path for an endoscope camera."""
        return _lookup_path(self.videos, camera, f"{camera}.mp4", self.sequence_id)

    def raw_video_path(self, camera: str = "endoscope1") -> Path:
        """Return the unprocessed public video path for an endoscope camera."""
        return _lookup_path(self.raw_videos, camera, f"{camera}_raw.mp4", self.sequence_id)

    def video_path(self, camera: str = "endoscope1", raw: bool = False) -> Path:
        return self.raw_video_path(camera) if raw else self.processed_video_path(camera)

    def tool_mask_path(self, camera: str = "endoscope1") -> Path:
        return _lookup_path(self.tool_masks, camera, f"{camera}_tools.mp4", self.sequence_id)


class IMEDDataset:
    """Discover and load a public iMED release tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.dataset_metadata = _read_dataset_metadata(self.root / "dataset_metadata.json")
        self.sessions = self._discover_sessions()
        self.session_metadata = self._read_session_metadata()
        self.scenes = self._discover_scenes()
        self.sequences = self._discover_sequences()

    def _discover_sessions(self) -> list[str]:
        return sorted(path.name for path in self.root.glob("session_*") if path.is_dir())

    def _discover_scenes(self) -> list[str]:
        scenes = set()
        for session_id in self.sessions:
            session_metadata = self.session_metadata.get(session_id, {})
            for region in session_metadata.get("anatomical_regions", []) if isinstance(session_metadata, dict) else []:
                if isinstance(region, dict) and isinstance(region.get("region_name"), str):
                    scenes.add(f"{session_id}/{region['region_name']}")
            for scene_dir in sorted((self.root / session_id).iterdir()):
                if scene_dir.is_dir():
                    scenes.add(f"{session_id}/{scene_dir.name}")
        return sorted(scenes)

    def _read_session_metadata(self) -> dict[str, dict[str, object]]:
        metadata = {}
        for session_id in self.sessions:
            path = self.root / session_id / "session_metadata.json"
            if path.is_file():
                metadata[session_id] = _read_json_object(path, "session metadata")
            else:
                metadata[session_id] = {}
        return metadata

    def _discover_sequences(self) -> dict[str, IMEDSequence]:
        sequences: dict[str, IMEDSequence] = {}
        metadata_by_sequence = self._sequence_metadata_index()
        for scene_id in self.scenes:
            session_id, scene_name = scene_id.split("/", 1)
            scene_dir = self.root / session_id / scene_name
            if not scene_dir.is_dir():
                continue
            for seq_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
                sequence_id = f"{session_id}/{scene_name}/{seq_dir.name}"
                sequence_metadata, region_metadata = metadata_by_sequence.get(sequence_id, ({}, {}))
                csvs = _find_timestamp_csvs(seq_dir)
                videos = _find_named_videos(seq_dir, suffix="")
                raw_videos = _find_named_videos(seq_dir, suffix="_raw")
                tool_masks = _find_named_videos(seq_dir, suffix="_tools")
                rel_pose = seq_dir / "relative_poses.json"
                trajectory_pose_file = seq_dir / "trajectory_poses.txt"
                pose_metadata = _read_pose_metadata(rel_pose if rel_pose.is_file() else None)
                sequences[sequence_id] = IMEDSequence(
                    sequence_id=sequence_id,
                    session_id=session_id,
                    scene_id=scene_name,
                    sequence_name=seq_dir.name,
                    path=seq_dir,
                    timestamp_csv=csvs[0] if csvs else None,
                    videos=videos,
                    raw_videos=raw_videos,
                    tool_masks=tool_masks,
                    masks=tool_masks,
                    relative_pose_file=rel_pose if rel_pose.is_file() else None,
                    trajectory_pose_file=trajectory_pose_file if trajectory_pose_file.is_file() else None,
                    metadata=sequence_metadata,
                    region_metadata=region_metadata,
                    session_metadata=self.session_metadata.get(session_id, {}),
                    pose_source=_metadata_value(pose_metadata, "pose_source"),
                    aruco_tag_dictionary=_metadata_value(
                        pose_metadata,
                        "aruco_marker_dictionary",
                        "aruco_tag_dictionary",
                        "aruco_dictionary",
                        "tag_dictionary",
                    ),
                )
        return sequences

    def _sequence_metadata_index(self) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
        index = {}
        for session_id, session_metadata in self.session_metadata.items():
            for region in session_metadata.get("anatomical_regions", []) if isinstance(session_metadata, dict) else []:
                if not isinstance(region, dict):
                    continue
                scene_id = region.get("region_name")
                if not isinstance(scene_id, str):
                    continue
                region_metadata = {key: value for key, value in region.items() if key != "sequences"}
                for sequence in region.get("sequences", []):
                    if not isinstance(sequence, dict) or not isinstance(sequence.get("sequence_name"), str):
                        continue
                    sequence_id = sequence.get("sequence_id") or f"{session_id}/{scene_id}/{sequence['sequence_name']}"
                    if isinstance(sequence_id, str):
                        index[sequence_id] = (sequence, region_metadata)
        return index

    def get_sequence(self, sequence_id: str) -> IMEDSequence:
        try:
            return self.sequences[sequence_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.sequences)[:5])
            raise KeyError(f"Unknown sequence_id {sequence_id!r}. First available: {available}") from exc

    def get_dataset_metadata(self) -> dict[str, object]:
        return self.dataset_metadata

    def get_session_metadata(self, session_id: str) -> dict[str, object]:
        try:
            return self.session_metadata[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session_id {session_id!r}") from exc

    def get_scene_metadata(self, scene_id: str) -> dict[str, object]:
        session_id, region_name = scene_id.split("/", 1)
        session_metadata = self.get_session_metadata(session_id)
        for region in session_metadata.get("anatomical_regions", []):
            if isinstance(region, dict) and region.get("region_name") == region_name:
                return region
        raise KeyError(f"Unknown scene_id {scene_id!r}")

    def get_sequence_metadata(self, sequence_id: str) -> dict[str, object]:
        return self.get_sequence(sequence_id).metadata

    def get_frame_rate(self, session_id: str) -> float:
        metadata = self.get_session_metadata(session_id)
        value = metadata.get("frame_rate")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frame_rate must be numeric for {session_id}, got {value!r}") from exc

    def filter(
        self,
        session_ids: list[str] | tuple[str, ...] | set[str] | None = None,
        scene_ids: list[str] | tuple[str, ...] | set[str] | None = None,
        sequence_names: list[str] | tuple[str, ...] | set[str] | None = None,
        session_types: list[str] | tuple[str, ...] | set[str] | None = None,
        tissue_types: list[str] | tuple[str, ...] | set[str] | None = None,
        organ_types: list[str] | tuple[str, ...] | set[str] | None = None,
        movement_types: list[str] | tuple[str, ...] | set[str] | None = None,
        tools_present: bool | None = None,
        aruco_present: bool | None = None,
        arucos_present: bool | None = None,
        energy_used: bool | None = None,
        rigidity: bool | None = None,
        tissue_deformation: bool | None = None,
        tool_combinations: list[str] | tuple[str, ...] | set[str] | None = None,
        require_all_tools: bool = False,
        has_trajectory: bool | None = None,
        has_tool_masks: bool | None = None,
        has_processed_videos: bool | None = None,
        has_raw_videos: bool | None = None,
        predicate: Callable[[IMEDSequence], bool] | None = None,
    ) -> list[IMEDSequence]:
        """Return sequences matching released metadata and file-availability filters."""
        if aruco_present is None:
            aruco_present = arucos_present
        matches = []
        for sequence in self.sequences.values():
            metadata = sequence.metadata
            session_metadata = sequence.session_metadata
            region_metadata = sequence.region_metadata
            if session_ids is not None and sequence.session_id not in set(session_ids):
                continue
            if scene_ids is not None and sequence.scene_id not in set(scene_ids) and f"{sequence.session_id}/{sequence.scene_id}" not in set(scene_ids):
                continue
            if sequence_names is not None and sequence.sequence_name not in set(sequence_names):
                continue
            if session_types is not None and session_metadata.get("tissue_type") not in set(session_types):
                continue
            if tissue_types is not None and session_metadata.get("tissue_type") not in set(tissue_types):
                continue
            if organ_types is not None and region_metadata.get("organ_type") not in set(organ_types):
                continue
            if movement_types is not None and metadata.get("movement_type") not in set(movement_types):
                continue
            if tools_present is not None and _metadata_bool(metadata.get("tools_present")) is not tools_present:
                continue
            if aruco_present is not None and _metadata_bool(metadata.get("arucos_present")) is not aruco_present:
                continue
            if energy_used is not None and _metadata_bool(metadata.get("energy_used")) is not energy_used:
                continue
            if rigidity is not None and _metadata_bool(metadata.get("rigidity")) is not rigidity:
                continue
            if tissue_deformation is not None and _metadata_bool(metadata.get("tissue_deformation")) is not tissue_deformation:
                continue
            if tool_combinations is not None:
                wanted = set(tool_combinations)
                actual = set(metadata.get("tool_combination") or [])
                if require_all_tools:
                    if not wanted.issubset(actual):
                        continue
                elif not wanted.intersection(actual):
                    continue
            if has_trajectory is not None and (sequence.relative_pose_file is not None and sequence.trajectory_pose_file is not None) is not has_trajectory:
                continue
            if has_tool_masks is not None and bool(sequence.tool_masks) is not has_tool_masks:
                continue
            if has_processed_videos is not None and (len(sequence.videos) == 2) is not has_processed_videos:
                continue
            if has_raw_videos is not None and (len(sequence.raw_videos) == 2) is not has_raw_videos:
                continue
            if predicate is not None and not predicate(sequence):
                continue
            matches.append(sequence)
        return sorted(matches, key=lambda item: item.sequence_id)

    def get_task_sequences(self, task: str, session_types: list[str] | tuple[str, ...] | set[str] | None = None) -> list[IMEDSequence]:
        """Return common paper-task subsets using released metadata filters."""
        camera_motion = ["circular_camera_motion", "zoom_camera_motion", "lateral_camera_motion"]
        task = task.lower().replace("-", "_").replace(" ", "_")
        if task in {"rigid_nvs", "temporal_stereo_pose", "temporal_stereo_pose_estimation"}:
            return self.filter(session_types=session_types, movement_types=camera_motion, has_trajectory=True)
        if task in {"deformable_nvs", "online_deformable_nvs"}:
            return self.filter(
                session_types=session_types,
                movement_types=["off_camera_manipulation", "tool_manipulation"],
            )
        if task in {"pose_estimation", "feature_matching", "optical_flow", "monocular_depth"}:
            return self.filter(session_types=session_types, aruco_present=False)
        raise ValueError(
            "Unknown task. Expected one of rigid_nvs, deformable_nvs, online_deformable_nvs, "
            "pose_estimation, feature_matching, optical_flow, monocular_depth, temporal_stereo_pose."
        )

    def read_timestamps(self, sequence_id: str) -> list[dict[str, str]]:
        sequence = self.get_sequence(sequence_id)
        if sequence.timestamp_csv is None:
            raise FileNotFoundError(f"No timestamp CSV found for {sequence_id}")
        with sequence.timestamp_csv.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def get_video_path(self, sequence_id: str, camera: str = "endoscope1", raw: bool = False) -> Path:
        sequence = self.get_sequence(sequence_id)
        return sequence.video_path(camera=camera, raw=raw)

    def get_raw_video_path(self, sequence_id: str, camera: str = "endoscope1") -> Path:
        return self.get_video_path(sequence_id, camera=camera, raw=True)

    def get_scope_baseline(self, unit: str = "m") -> float:
        """Return the public stereo baseline shared by the endoscope scopes."""
        baseline_mm = _metadata_float(self.dataset_metadata, "scope_baseline_mm", SCOPE_BASELINE_MM)
        if unit == "mm":
            return baseline_mm
        if unit == "m":
            return baseline_mm / 1000.0
        raise ValueError("unit must be 'm' or 'mm'")

    def get_frame(
        self,
        sequence_id: str,
        frame_idx: int,
        camera: str = "endoscope1",
        split: bool = True,
        raw: bool = False,
    ):
        sequence = self.get_sequence(sequence_id)
        video = sequence.video_path(camera=camera, raw=raw)
        frame = read_video_frame(video, frame_idx)
        if not split:
            return frame
        return split_stacked_stereo(frame, "vertical")

    def get_stereo_pair(
        self,
        sequence_id: str,
        frame_idx: int,
        camera: str = "endoscope1",
        raw: bool = False,
    ):
        """Return the two per-scope stereo views for a processed or raw video frame."""
        return self.get_frame(sequence_id, frame_idx, camera=camera, split=True, raw=raw)

    def get_frame_from_video(self, sequence_id: str, frame_idx: int, raw: bool = False) -> dict[str, object]:
        """Return one frame in the original loader's L/R/SL/SR convention.

        This mirrors the internal loader contract used by the benchmarking
        scripts: endoscope1 contains L/R and endoscope2 contains SL/SR.
        """
        l_img, r_img = self.get_stereo_pair(sequence_id, frame_idx, camera="endoscope1", raw=raw)
        sl_img, sr_img = self.get_stereo_pair(sequence_id, frame_idx, camera="endoscope2", raw=raw)
        sequence = self.get_sequence(sequence_id)
        return {
            "L": l_img,
            "R": r_img,
            "SL": sl_img,
            "SR": sr_img,
            "sequence_id": sequence_id,
            "session_id": sequence.session_id,
            "frame_index": frame_idx,
            "raw": raw,
            "calibration": self.get_calibration(sequence.session_id),
            "foundation_stereo_depth_config": self.get_depth_benchmark_config(),
        }

    def get_intrinsics(self, session_id: str) -> dict[str, dict[str, np.ndarray]]:
        calib_dir = self.root / "calibrations" / session_id
        if not calib_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
        intrinsics = {}
        for name in ("L", "R", "SL", "SR"):
            path = calib_dir / f"{name}_calibration_chessboard.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"Calibration file not found: {path}")
            intrinsics[name] = _read_opencv_calibration(path)
        return intrinsics

    def get_calibration(self, session_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return calibration in the original loader shape: camera -> (K, D)."""
        intrinsics = self.get_intrinsics(session_id)
        return {name: (record["K"], record["D"]) for name, record in intrinsics.items()}

    def get_stereo_intrinsics(self, session_id: str, camera: str = "endoscope1") -> dict[str, dict[str, np.ndarray]]:
        """Return calibration records for the two views inside one stacked endoscope video."""
        left_key, right_key = _camera_pair(camera)
        intrinsics = self.get_intrinsics(session_id)
        return {"left": intrinsics[left_key], "right": intrinsics[right_key]}

    def get_focal_length_px(self, session_id: str, camera: str = "endoscope1") -> float:
        """Return the left-view focal length used by the benchmark depth script."""
        left_key, _ = _camera_pair(camera)
        k = self.get_intrinsics(session_id)[left_key]["K"]
        return float(k[0, 0])

    def get_depth_benchmark_config(self) -> dict[str, object]:
        """Return the FoundationStereo settings used by the benchmark scripts."""
        config = FOUNDATION_STEREO_DEPTH_CONFIG.copy()
        config["baseline_m"] = self.get_scope_baseline("m")
        return config

    def get_foundation_stereo_inputs(
        self,
        sequence_id: str,
        frame_idx: int,
        camera: str = "endoscope1",
        raw: bool = False,
    ) -> dict[str, object]:
        """Return the inputs expected by the benchmark FoundationStereo path.

        The public loader does not ship FoundationStereo weights or run dense
        depth inference. It returns the same stereo pair, calibration, baseline,
        and benchmark settings used by the internal depth scripts so users can
        run an aligned depth pipeline externally.
        """
        sequence = self.get_sequence(sequence_id)
        left, right = self.get_stereo_pair(sequence_id, frame_idx, camera=camera, raw=raw)
        left_key, right_key = _camera_pair(camera)
        calibration = self.get_calibration(sequence.session_id)
        return {
            "left": left,
            "right": right,
            "K_left": calibration[left_key][0],
            "D_left": calibration[left_key][1],
            "K_right": calibration[right_key][0],
            "D_right": calibration[right_key][1],
            "baseline_m": self.get_scope_baseline("m"),
            "baseline_mm": self.get_scope_baseline("mm"),
            "focal_length_px": self.get_focal_length_px(sequence.session_id, camera=camera),
            "camera": camera,
            "camera_pair": (left_key, right_key),
            "foundation_stereo_depth_config": self.get_depth_benchmark_config(),
            "raw": raw,
            "sequence_id": sequence_id,
            "frame_idx": frame_idx,
        }

    def load_trajectory(self, sequence_id: str) -> dict[str, object]:
        sequence = self.get_sequence(sequence_id)
        if sequence.relative_pose_file is None:
            raise FileNotFoundError(f"No public relative_poses.json found for {sequence_id}")
        with sequence.relative_pose_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        trajectory_name = metadata.get("trajectory_file", "trajectory_poses.txt")
        traj_path = _resolve_sequence_file(sequence.path, trajectory_name)
        if not traj_path.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        data = np.genfromtxt(traj_path, names=True, dtype=float)
        if data.size == 0:
            raise ValueError(f"Trajectory file has no pose rows: {traj_path}")
        return {"metadata": metadata, "trajectory": data, "path": traj_path}


def _lookup_path(paths: dict[str, Path], key: str, expected_name: str, sequence_id: str) -> Path:
    path = paths.get(key)
    if path is None:
        raise FileNotFoundError(f"{expected_name} not found for {sequence_id}")
    return path


def _find_timestamp_csvs(sequence_dir: Path) -> list[Path]:
    csvs = []
    for pattern in ("filenames.csv", "*_filenames.csv", "*filenames*.csv"):
        csvs.extend(path for path in sequence_dir.glob(pattern) if path.is_file())
    return sorted(set(csvs))


def _find_named_videos(sequence_dir: Path, suffix: str) -> dict[str, Path]:
    videos = {}
    for camera in ("endoscope1", "endoscope2"):
        path = sequence_dir / f"{camera}{suffix}.mp4"
        if path.is_file():
            videos[camera] = path
    return videos


def _read_pose_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return _read_json_object(path, "public pose metadata")


def _read_dataset_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"scope_baseline_mm": SCOPE_BASELINE_MM}
    metadata = _read_json_object(path, "dataset metadata")
    metadata.setdefault("scope_baseline_mm", SCOPE_BASELINE_MM)
    return metadata


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {label}: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return metadata


def _metadata_value(metadata: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    processing_info = metadata.get("processing_info")
    if isinstance(processing_info, dict):
        for key in keys:
            value = processing_info.get(key)
            if value not in (None, ""):
                return value
    return None


def _metadata_float(metadata: dict[str, object], key: str, default: float) -> float:
    value = metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc


def _metadata_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    return bool(value)


def _camera_pair(camera: str) -> tuple[str, str]:
    try:
        return STEREO_CAMERA_PAIRS[camera]
    except KeyError as exc:
        raise KeyError(f"Unknown camera {camera!r}; expected one of {sorted(STEREO_CAMERA_PAIRS)}") from exc


def _resolve_sequence_file(sequence_dir: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"trajectory_file must be a relative filename in {sequence_dir}")
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"trajectory_file must stay inside the sequence directory: {filename}")
    return sequence_dir / path


def _read_opencv_calibration(path: Path) -> dict[str, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise RuntimeError(f"Could not open calibration YAML: {path}")
        k = fs.getNode("K").mat()
        d = fs.getNode("D").mat()
    finally:
        fs.release()
    if k is None or d is None:
        raise ValueError(f"Calibration YAML missing K or D matrix: {path}")
    return {"K": k, "D": d}
