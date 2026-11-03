# iMED Scripts

This repository contains the scripts supporting the iMED paper: the dataset
loader, examples, validation tools, pose-estimation and novel-view-synthesis
converters, and optional stereo depth generation.

Following the completion of the iMED challenge, **iMED v2.0** is the full dataset
release, containing 341 sequences across 43 scenes and seven sessions (~48 GB).
The earlier v1.0 sample release contains one scene per session (~6 GB).

**Dataset:** [Download iMED](https://doi.org/10.5522/04/32764638)

**Challenge pages:**

- [iMED Challenge](https://imed-challenge.github.io/)
- [Synapse challenge page](https://www.synapse.org/Synapse:syn74277461/wiki/639538)

The loader supports both v1 and v2. After downloading and extracting the dataset
archive, the examples below assume this layout:

```text
imed_v2_0/
imed_scripts/
```

Use `IMEDDataset("../imed_v2_0")` for v2 or `IMEDDataset("../imed_v1_0")` for v1.

## Release Layout

iMED v2.0:

- `imed_v2_0/`: data and primary loader target. Each released sequence contains processed videos when applicable, raw videos for reprocessing, timestamps, calibration/intrinsics, tool masks where present, metadata, and reference trajectories.

Sequence directories use these names:

- `endoscope1.mp4`, `endoscope2.mp4`: processed endoscopic videos. For sequences with ArUcos present, these are the current marker-inpainted outputs.
- `endoscope1_raw.mp4`, `endoscope2_raw.mp4`: unprocessed/raw videos for users who want to inspect or recompute processing.
- `endoscope1_tools.mp4`, `endoscope2_tools.mp4`: tool masks when tools are present.
- `*_filenames.csv`: timestamp table.
- `relative_poses.json` and `trajectory_poses.txt`: public pose metadata and trajectory values.

The release provides processed and raw videos, plus ArUco dictionary metadata.

The release includes:

- processed stereo videos
- raw stereo videos in the same sequence directories
- timestamp CSV files
- per-session intrinsics and calibration files
- calibration sequences
- dataset, session, scene, and sequence metadata
- instrument masks when tools are present
- reference trajectories
- dataloader, examples, validation script, and loader test

Stereo depth maps can be generated with the included FoundationStereo helper.

## Video Layout

Each `endoscope1.mp4` and `endoscope2.mp4` file is a stacked stereo video for one physical endoscope.

- The top half of each frame is the left cam.
- The bottom half of each frame is the right cam.

Frame indices are zero-based video positions. Both eyes are split from the same
decoded frame: endoscope1 supplies L/R and endoscope2 supplies SL/SR. Capture
timestamps are provided for each endoscope.

## Install

Requires Python 3.11 or newer.

```bash
cd imed_scripts
pip install -r requirements.txt
```

Validate an extracted dataset:

```bash
python scripts/validate_dataset.py ../imed_v2_0
```

## Expected Layout

```text
imed_scripts/
  README.md
  requirements.txt
  imed_dataset_loader.py
  endoscope_dataloader.py
  examples/
    list_dataset.py
    load_frame.py
    load_trajectory.py
    prepare_foundation_stereo_inputs.py
  scripts/
    convert_to_imed_pe.py
    convert_to_imed_nvs.py
    convert_task.py
    run_foundation_stereo_depths.py
    test_loader.py
    test_pose_api.py
    test_video_indexing.py
    validate_dataset.py
```

## Loader API

The loader exposes:

- `IMEDDataset(root)`
- `dataset.sessions`
- `dataset.scenes`
- `dataset.sequences`
- `dataset.get_sequence(sequence_id)`
- `dataset.get_dataset_metadata()`
- `dataset.get_session_metadata(session_id)`
- `dataset.get_scene_metadata(scene_id)`
- `dataset.get_sequence_metadata(sequence_id)`
- `dataset.get_frame_rate(session_id)`
- `dataset.filter(...)`
- `dataset.get_frame(sequence_id, frame_idx)`
- `dataset.get_frame(sequence_id, frame_idx, raw=True)`
- `dataset.get_video_path(sequence_id, camera="endoscope1", raw=False)`
- `dataset.get_raw_video_path(sequence_id, camera="endoscope1")`
- `dataset.get_frame_from_video(sequence_id, frame_idx, raw=False)`
- `dataset.get_intrinsics(session_id)`
- `dataset.get_calibration(session_id)`
- `dataset.get_stereo_intrinsics(session_id, camera="endoscope1")`
- `dataset.get_scope_baseline(unit="m")`
- `dataset.get_depth_benchmark_config()`
- `dataset.get_foundation_stereo_inputs(sequence_id, frame_idx, raw=False)`
- `dataset.load_trajectory(sequence_id)`

Run the full loader test:

```bash
python scripts/test_loader.py ../imed_v2_0
```

Run the pixel-level indexing regression without downloading the dataset:

```bash
python scripts/test_video_indexing.py
```

It checks all four eyes in raw and processed videos against independent
sequential decoding, including repeated/backward access and invalid indices.

Each `IMEDSequence` exposes:

- `sequence.videos`: processed videos, keyed by `endoscope1` and `endoscope2`
- `sequence.raw_videos`: unprocessed videos, keyed by `endoscope1` and `endoscope2`
- `sequence.tool_masks`: tool-mask videos, keyed by camera when present
- `sequence.masks`: compatibility alias for `sequence.tool_masks`
- `sequence.metadata`: released per-sequence metadata, including motion category, tool flags/types, ArUco flags, energy usage, tissue deformation, and file references
- `sequence.region_metadata`: released anatomical region metadata, including organ type and organ description
- `sequence.session_metadata`: released per-session metadata, including tissue type, camera system, frame rate, sequence duration, and calibration references
- `sequence.pose_source`: optional source sequence association for released poses, preserved as the JSON value from `relative_poses.json`
- `sequence.aruco_tag_dictionary`: optional ArUco dictionary metadata used for pose generation
- `sequence.processed_video_path(camera)`, `sequence.raw_video_path(camera)`, and `sequence.tool_mask_path(camera)`

Moving-ArUco calibration sequences provide raw videos and are linked through
pose-source metadata.

## Metadata Filtering

The loader supports the task-specific filtering shown in the paper:

```python
from imed_dataset_loader import IMEDDataset

dataset = IMEDDataset("../imed_v2_0")

camera_motion = [
    "circular_camera_motion",
    "zoom_camera_motion",
    "lateral_camera_motion",
]

rigid_nvs = dataset.filter(
    session_types=["ex_vivo"],
    movement_types=camera_motion,
)

deformable_nvs = dataset.filter(
    session_types=["in_vivo_postmortem"],
    movement_types=["off_camera_manipulation", "tool_manipulation"],
)

pose_or_depth = dataset.filter(
    session_types=["in_vivo_live"],
    aruco_present=False,
)

tool_energy = dataset.filter(
    tools_present=True,
    energy_used=True,
    tool_combinations=["cautery_hook"],
)
```

## Stereo Depths

Prepare stereo depth inputs:

```python
inputs = dataset.get_foundation_stereo_inputs(
    sequence_id,
    frame_idx=0,
    camera="endoscope1"
)
```

Run one frame:

```bash
python scripts/run_foundation_stereo_depths.py \
  /path/to/imed_v2_0 \
  --sequence-id session_001/chickenbreast_10deg/circular \
  --foundation-stereo-dir /path/to/FoundationStereo \
  --frame-idx 0 \
  --camera endoscope1 \
  --device cuda \
  --save-image
```

Run a full sequence and save depth videos inside the sequence directory:

```bash
python scripts/run_foundation_stereo_depths.py \
  /path/to/imed_v2_0 \
  --sequence-id session_001/chickenbreast_10deg/circular \
  --foundation-stereo-dir /path/to/FoundationStereo \
  --camera both \
  --device cuda \
  --save-video \
  --no-save-arrays
```

By default, outputs are written inside the sequence folder as `endoscope1_depth/`, `endoscope2_depth/`, `endoscope1_depth.mp4`, and `endoscope2_depth.mp4`.

FoundationStereo model weights/config are expected at:

```text
FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
FoundationStereo/pretrained_models/23-51-11/cfg.yaml
```

For depth generation, install FoundationStereo's dependencies, including a
CUDA-compatible PyTorch build and `omegaconf`.


## Poses

`trajectory_poses.txt` is a whitespace-separated table with a header:

```text
frame_idx T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23 T30 T31 T32 T33 trans_uncertainty_m rot_uncertainty_deg combined_uncertainty
```

`T00`–`T33` form a row-major 4×4 transform. In v2, this is `T_SL_from_L`,
mapping endoscope1 left coordinates to endoscope2 left coordinates:
`X_SL = R @ X_L + t`, with translation in **metres**. `frame_idx` is the
zero-based endoscope1 video/CSV row. Dynamic trajectories contain one transform
per labelled frame; static trajectories contain one transform with a
`valid_frame_range` recorded in `relative_poses.json`.

Use `dataset.load_trajectory(sequence_id)` to read trajectories in either
version. V2 also supports frame-specific pose queries:

```python
dataset.get_pose("session_007/scene_9/circular", 561)
dataset.get_poses("session_007/scene_9/circular", [0, 561])
dataset.get_image_calibration("session_007")
```

`get_pose` and `get_poses` check for coverage and validity. `get_calibration` retains
the original lens coefficients; `get_image_calibration` returns zero distortion
for v2's already-undistorted pixels.

Uncertainty columns describe independent channel disagreement using a nominal
parallel stereo baseline.

## Task conversion

The converters read the public v2 release.
Use `--dry-run` to list eligible recordings before extraction:

```sh
python imed_scripts/scripts/convert_to_imed_pe.py imed_v2_0 iMED_pe_v2 --dry-run
python imed_scripts/scripts/convert_to_imed_pe.py imed_v2_0 iMED_pe_v2 --sequence-id session_007/scene_9/circular --split train
python imed_scripts/scripts/convert_to_imed_nvs.py imed_v2_0 iMED_NVS_v2 --sequence-id session_007/scene_9/tool_1
```

Both formats have `session_NNN_scene_sequence/`, `K.txt`, `pose.txt`, and
`endoscope{1,2}/{L,R}/frame_NNNNNN.png`. Images are 1280×1024, with matching
intrinsics in `K.txt`. Task pose translations are in **millimetres**, quaternions in
**xyzw** order. PE uses the normalization
`inverse(T_first_exported) @ T_frame`, with source video indices in column 0.
NVS uses two pose rows: row 0 is endoscope2 left at identity; row 1 is
endoscope1 left in the endoscope2-left coordinate system.

NVS exports masks into `toolL/` and `toolR/`. To generate depth in millimetres,
add `--foundation-stereo-dir /path/to/FoundationStereo --device cuda:0`.

To verify the downloaded dataset against its manifest, run
`python imed_scripts/scripts/validate_dataset.py imed_v2_0 --verify-checksums`.
