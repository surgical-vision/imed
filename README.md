# iMED v1.0 Scripts

iMED v1.0 is the first iMED sample release. It is not the full iMED dataset. The full dataset is deferred while the MICCAI challenge is active to reduce data leakage. Challenge solutions are due by Sept. 1, with the planned full iMED release targeted for the ECCV conference date, Sept. 8. 

Version 1 contains one scene per session, so users can inspect the structure of iMED (\~6 GB). The full dataset contains contains \~7 scenes per session (\~45 GB). 

The dataset can be found at: [LINK HERE]

Challenge pages:

- https://imed-challenge.github.io/
- https://www.synapse.org/Synapse:syn74277461/wiki/639538

This repository contains the scripts for iMED v1.0: the loader, examples, validation script, loader test, and optional depth-generation helper.

After downloading and extracting the dataset archive, examples below assume this layout:

```text
imed_v1_0/
imed_scripts/
```

## Release Layout

iMED v1.0:

- `imed_v1_0/`: data and primary loader target. Each released sequence contains processed videos when applicable, raw videos for reprocessing, timestamps, calibration/intrinsics, tool masks where present, metadata, and reference trajectories.

Sequence directories use these names:

- `endoscope1.mp4`, `endoscope2.mp4`: processed endoscopic videos. For sequences with ArUcos present, these are photometrically matched and inpainted outputs.
- `endoscope1_raw.mp4`, `endoscope2_raw.mp4`: unprocessed/raw videos for users who want to inspect or recompute processing.
- `endoscope1_tools.mp4`, `endoscope2_tools.mp4`: tool masks when tools are present.
- `*_filenames.csv`: timestamp table.
- `relative_poses.json` and `trajectory_poses.txt`: public pose metadata and trajectory values.

The release provides processed and raw videos, plus ArUco dictionary metadata.

## Included Signals

The subset includes:  

- processed stereo videos
- raw stereo videos in the same sequence directories
- timestamp CSV files
- per-session intrinsics and calibration files
- approved calibration sequences needed for recomputation workflows
- dataset, session, scene, and sequence metadata
- instrument masks where tools are present
- reference trajectories where available
- dataloader, examples, validation script, and loader test

Stereo depth maps are omitted for size reasons. They are intended to be recomputable from the released data and code.

## Video Layout

Each `endoscope1.mp4` and `endoscope2.mp4` file is a stacked stereo video for one physical endoscope.

- The top half of each frame is the left cam.
- The bottom half of each frame is the right cam.

## Install

```bash
cd imed_scripts
pip install -r requirements.txt
```

Validate an extracted dataset:

```bash
python scripts/validate_dataset.py ../imed_v1_0
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
    run_foundation_stereo_depths.py
    test_loader.py
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
python scripts/test_loader.py ../imed_v1_0
```

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

Moving-ArUco calibration sequences may be present when another included sequence points to them through pose-source metadata. They are calibration/raw sequences and are not expected to contain processed DiffuEraser videos.

## Metadata Filtering

The loader supports the task-specific filtering shown in the paper:

```python
from imed_dataset_loader import IMEDDataset

dataset = IMEDDataset("../imed_v1_0")

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

Stereo depth maps must be recomputed:

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
  /path/to/imed_v1_0 \
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
  /path/to/imed_v1_0 \
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

Install FoundationStereo's optional dependencies, including a CUDA-compatible PyTorch build and `omegaconf`, in the environment used to run the depth script. These dependencies are not part of the default loader requirements since they are GPU specific (best if  you can figure this out for your individual gpu).
