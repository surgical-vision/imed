#!/usr/bin/env python3
"""Run benchmark-aligned FoundationStereo depth generation for public iMED videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from imed_dataset_loader import IMEDDataset


DEFAULT_FOUNDATION_STEREO_URL = "https://github.com/NVlabs/FoundationStereo.git"
DEFAULT_RELATIVE_FOUNDATION_STEREO_DIR = Path("third_party") / "FoundationStereo"
MODEL_RELATIVE_DIR = Path("pretrained_models") / "23-51-11"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Path to imed_v1_0")
    parser.add_argument("--sequence-id", action="append", help="Sequence id to process. May be repeated. Defaults to all sequences.")
    parser.add_argument("--camera", choices=["endoscope1", "endoscope2", "both"], default="both")
    parser.add_argument("--raw", action="store_true", help="Use endoscope*_raw.mp4 instead of processed videos")
    parser.add_argument("--frame-idx", type=int, action="append", help="Frame index to process. May be repeated.")
    parser.add_argument("--max-frames", type=int, help="Evenly sample at most this many frames per sequence")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional external output root. Defaults to <sequence>/<camera>_depth inside the dataset tree.",
    )
    parser.add_argument("--foundation-stereo-dir", type=Path, help="Path to an existing FoundationStereo checkout")
    parser.add_argument("--download-foundation-stereo", action="store_true", help="Clone FoundationStereo if it is not found")
    parser.add_argument("--no-prompt", action="store_true", help="Fail instead of prompting to clone FoundationStereo")
    parser.add_argument("--foundation-stereo-url", default=DEFAULT_FOUNDATION_STEREO_URL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-image", action="store_true", help="Also save a colorized depth PNG next to each depth array")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Also save a colorized depth MP4 as endoscope1_depth.mp4 or endoscope2_depth.mp4",
    )
    parser.add_argument("--no-save-arrays", action="store_true", help="Do not save per-frame .npy depth arrays")
    args = parser.parse_args()

    dataset = IMEDDataset(args.root)
    foundation_dir = resolve_foundation_stereo_dir(args)
    if foundation_dir is None:
        return 2

    try:
        model, input_padder_cls = load_foundation_stereo_model(foundation_dir, args.device)
    except Exception as exc:
        print(f"Could not load FoundationStereo: {exc}", file=sys.stderr)
        print_model_help(foundation_dir)
        return 2

    sequence_ids = args.sequence_id or sorted(dataset.sequences)
    cameras = ["endoscope1", "endoscope2"] if args.camera == "both" else [args.camera]

    for sequence_id in sequence_ids:
        sequence = dataset.get_sequence(sequence_id)
        frame_indices = select_frame_indices(dataset, sequence_id, cameras[0], args.raw, args.frame_idx, args.max_frames)
        for camera in cameras:
            output_dir = resolve_output_dir(sequence, camera, args.raw, args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            depth_video_path = resolve_video_path(sequence, camera, args.raw, args.output_dir)
            video_writer = None
            video_metadata = None
            video_depth_min_m = None
            video_depth_max_m = None
            if args.save_video:
                if args.skip_existing and depth_video_path.exists():
                    print(f"skipping existing {depth_video_path}")
                    continue
                source_video = dataset.get_video_path(sequence_id, camera=camera, raw=args.raw)
                source_info = video_info(source_video)
                depth_video_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(depth_video_path),
                    fourcc,
                    source_info["fps"],
                    (source_info["width"], source_info["height"] // 2),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not open depth video writer: {depth_video_path}")
                video_metadata = {
                    "sequence_id": sequence_id,
                    "camera": camera,
                    "raw": args.raw,
                    "source_video": str(source_video),
                    "output_video": str(depth_video_path),
                    "fps": source_info["fps"],
                    "width": source_info["width"],
                    "height": source_info["height"] // 2,
                    "frame_indices": frame_indices,
                    "frame_count": len(frame_indices),
                    "depth_unit": "meters",
                    "depth_formula": "depth_m = focal_length_px * baseline_m / disparity_px",
                    "video_note": "Colorized left-to-right stereo depth visualization. Use .npy arrays for metric depth values.",
                }
            for frame_idx in frame_indices:
                depth_path = output_dir / f"frame_{frame_idx:06d}_depth.npy"
                if args.skip_existing and depth_path.exists() and not args.save_video:
                    continue
                inputs = dataset.get_foundation_stereo_inputs(sequence_id, frame_idx, camera=camera, raw=args.raw)
                depth = run_foundation_stereo_inference(
                    model,
                    input_padder_cls,
                    inputs["left"],
                    inputs["right"],
                    inputs["K_left"],
                    float(inputs["baseline_m"]),
                    args.device,
                )
                depth_stats = depth_min_max(depth)
                depth_image = colorize_depth(depth)
                if not args.no_save_arrays:
                    np.save(depth_path, depth)
                    np.save(output_dir / f"frame_{frame_idx:06d}_K_left.npy", inputs["K_left"])
                if args.save_image:
                    image_path = output_dir / f"frame_{frame_idx:06d}_depth.png"
                    cv2.imwrite(str(image_path), depth_image)
                frame_metadata = {
                    "sequence_id": sequence_id,
                    "camera": camera,
                    "camera_pair": list(inputs["camera_pair"]),
                    "stereo_direction": f"{inputs['camera_pair'][0]}_to_{inputs['camera_pair'][1]}",
                    "frame_idx": frame_idx,
                    "raw": args.raw,
                    "baseline_m": float(inputs["baseline_m"]),
                    "baseline_mm": float(inputs["baseline_mm"]),
                    "focal_length_px": float(inputs["focal_length_px"]),
                    "depth_formula": "depth_m = focal_length_px * baseline_m / disparity_px",
                    "depth_unit": "meters",
                    "depth_clip_m": inputs["foundation_stereo_depth_config"]["depth_clip_m"],
                    "depth_min_m": depth_stats["min_m"],
                    "depth_max_m": depth_stats["max_m"],
                }
                if not args.no_save_arrays:
                    metadata_path = output_dir / f"frame_{frame_idx:06d}_metadata.json"
                    metadata_path.write_text(
                        json.dumps(frame_metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                if video_writer is not None:
                    video_writer.write(depth_image)
                    if video_metadata is not None:
                        video_metadata["camera_pair"] = list(inputs["camera_pair"])
                        video_metadata["stereo_direction"] = f"{inputs['camera_pair'][0]}_to_{inputs['camera_pair'][1]}"
                        video_metadata["baseline_m"] = float(inputs["baseline_m"])
                        video_metadata["baseline_mm"] = float(inputs["baseline_mm"])
                        video_metadata["focal_length_px"] = float(inputs["focal_length_px"])
                        video_metadata["depth_clip_m"] = inputs["foundation_stereo_depth_config"]["depth_clip_m"]
                        video_depth_min_m = (
                            depth_stats["min_m"]
                            if video_depth_min_m is None
                            else min(video_depth_min_m, depth_stats["min_m"])
                        )
                        video_depth_max_m = (
                            depth_stats["max_m"]
                            if video_depth_max_m is None
                            else max(video_depth_max_m, depth_stats["max_m"])
                        )
                print(
                    f"saved sequence={sequence.sequence_id} camera={camera} frame={frame_idx} "
                    f"depth_min_m={depth_stats['min_m']:.6f} depth_max_m={depth_stats['max_m']:.6f}"
                )
            if video_writer is not None:
                video_writer.release()
                if video_metadata is not None:
                    video_metadata["depth_min_m"] = 0.0 if video_depth_min_m is None else video_depth_min_m
                    video_metadata["depth_max_m"] = 0.0 if video_depth_max_m is None else video_depth_max_m
                    video_metadata_path = depth_video_path.with_name(depth_video_path.stem + "_metadata.json")
                    video_metadata_path.write_text(
                        json.dumps(video_metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    print(f"saved {depth_video_path}")
    return 0


def resolve_output_dir(sequence, camera: str, raw: bool, output_root: Path | None) -> Path:
    folder_name = f"{camera}_raw_depth" if raw else f"{camera}_depth"
    if output_root is None:
        return sequence.path / folder_name
    return output_root / sequence.sequence_id.replace("/", "__") / ("raw" if raw else "processed") / folder_name


def resolve_video_path(sequence, camera: str, raw: bool, output_root: Path | None) -> Path:
    video_name = f"{camera}_raw_depth.mp4" if raw else f"{camera}_depth.mp4"
    if output_root is None:
        return sequence.path / video_name
    return output_root / sequence.sequence_id.replace("/", "__") / ("raw" if raw else "processed") / video_name


def video_info(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def resolve_foundation_stereo_dir(args: argparse.Namespace) -> Path | None:
    candidates = []
    if args.foundation_stereo_dir:
        candidates.append(args.foundation_stereo_dir)
    candidates.append(SCRIPT_ROOT / DEFAULT_RELATIVE_FOUNDATION_STEREO_DIR)

    for candidate in candidates:
        if is_foundation_stereo_dir(candidate):
            return candidate.resolve()

    target = candidates[0]
    should_download = args.download_foundation_stereo
    if not should_download and not args.no_prompt and sys.stdin.isatty():
        answer = input(f"FoundationStereo was not found. Clone it to {target}? [y/N] ").strip().lower()
        should_download = answer in {"y", "yes"}

    if should_download:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", args.foundation_stereo_url, str(target)], check=True)
        if is_foundation_stereo_dir(target):
            return target.resolve()

    print("FoundationStereo checkout not found.", file=sys.stderr)
    print("Pass --foundation-stereo-dir PATH or rerun with --download-foundation-stereo.", file=sys.stderr)
    return None


def is_foundation_stereo_dir(path: Path) -> bool:
    return path.is_dir() and (path / "core").is_dir() and (path / "scripts").is_dir()


def print_model_help(foundation_dir: Path) -> None:
    model_dir = foundation_dir / MODEL_RELATIVE_DIR
    print("Expected FoundationStereo model files:", file=sys.stderr)
    print(f"  {model_dir / 'model_best_bp2.pth'}", file=sys.stderr)
    print(f"  {model_dir / 'cfg.yaml'}", file=sys.stderr)
    print("Install optional dependencies such as torch and omegaconf in the environment used to run this script.", file=sys.stderr)


def load_foundation_stereo_model(foundation_dir: Path, device: str):
    try:
        import torch
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing optional dependency: {exc.name}") from exc

    model_path = foundation_dir / MODEL_RELATIVE_DIR / "model_best_bp2.pth"
    cfg_path = foundation_dir / MODEL_RELATIVE_DIR / "cfg.yaml"
    if not model_path.is_file() or not cfg_path.is_file():
        raise FileNotFoundError("FoundationStereo model weights/config are missing")

    sys.path.insert(0, str(foundation_dir))
    sys.path.insert(0, str(foundation_dir / "scripts"))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    cfg = OmegaConf.load(str(cfg_path))
    model_args = OmegaConf.create(cfg)
    model_args.scale = 0.5
    model_args.hiera = 0
    model_args.z_far = 0.20
    model_args.valid_iters = 32
    model_args.get_pc = 0
    model_args.remove_invisible = 1
    model_args.denoise_cloud = 0
    model_args.vit_size = "vitl"

    model = FoundationStereo(model_args)
    ckpt = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, InputPadder


def select_frame_indices(
    dataset: IMEDDataset,
    sequence_id: str,
    camera: str,
    raw: bool,
    explicit_indices: list[int] | None,
    max_frames: int | None,
) -> list[int]:
    if explicit_indices:
        return sorted(set(explicit_indices))
    video_path = dataset.get_video_path(sequence_id, camera=camera, raw=raw)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    if max_frames is None or max_frames >= frame_count:
        return list(range(frame_count))
    return sorted(set(int(i) for i in np.linspace(0, frame_count - 1, max_frames, dtype=int)))


def run_foundation_stereo_inference(model, input_padder_cls, left_img, right_img, k_left, baseline_m: float, device: str):
    import torch
    import torch.nn.functional as F

    scale = 0.5
    pad_amount = 90

    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) if left_img.ndim == 3 and left_img.shape[2] == 3 else left_img
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) if right_img.ndim == 3 and right_img.shape[2] == 3 else right_img

    left_tensor = torch.as_tensor(left_rgb).to(device).float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right_rgb).to(device).float()[None].permute(0, 3, 1, 2)
    left_padded = F.pad(left_tensor, pad=(pad_amount, 0, 0, 0), mode="constant", value=0)
    right_padded = F.pad(right_tensor, pad=(0, pad_amount, 0, 0), mode="constant", value=0)
    left_scaled = F.interpolate(left_padded, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 255)
    right_scaled = F.interpolate(right_padded, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 255)

    padder = input_padder_cls(left_scaled.shape, divis_by=32, force_square=False)
    left_final, right_final = padder.pad(left_scaled, right_scaled)

    autocast_context = torch.autocast(device_type="cuda") if device.startswith("cuda") else nullcontext()
    with torch.no_grad():
        with autocast_context:
            disp = model.forward(left_final, right_final, iters=32, test_mode=True)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(left_scaled.shape[-2:])
    disp = disp[:, int(pad_amount * scale):]
    disp_full_scale = disp / scale

    depth = np.zeros_like(disp_full_scale)
    valid = disp_full_scale > 0
    depth[valid] = (float(k_left[0, 0]) * baseline_m) / disp_full_scale[valid]
    depth[~valid] = 0
    depth = np.clip(depth, 0.002, 0.2)
    return cv2.resize(depth, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_LINEAR)


def colorize_depth(depth: np.ndarray, min_depth: float = 0.002, max_depth: float = 0.2) -> np.ndarray:
    valid = depth > 0
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = (np.clip(depth[valid], min_depth, max_depth) - min_depth) / (max_depth - min_depth)
    image = (255.0 * (1.0 - normalized)).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def depth_min_max(depth: np.ndarray) -> dict[str, float]:
    valid = depth > 0
    if not np.any(valid):
        return {"min_m": 0.0, "max_m": 0.0}
    valid_depth = depth[valid]
    return {"min_m": float(valid_depth.min()), "max_m": float(valid_depth.max())}


if __name__ == "__main__":
    raise SystemExit(main())
