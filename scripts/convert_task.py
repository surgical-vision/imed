"""Shared PE/NVS conversion from the public release, without private split files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imed_dataset_loader import IMEDDataset


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pose_row(index, transform):
    """Legacy task convention: index, translation in mm, quaternion xyzw."""
    rvec = cv2.Rodrigues(transform[:3, :3])[0].ravel()
    angle = np.linalg.norm(rvec)
    xyz = rvec * (np.sin(angle/2)/angle if angle > 1e-12 else .5)
    return [index, *(transform[:3, 3]*1000), *xyz, np.cos(angle/2)]


def select_frames(dataset, sequence, args):
    timestamps = dataset.read_timestamps(sequence.sequence_id)
    assert [int(t['index']) for t in timestamps] == list(range(len(timestamps)))
    selected = []
    for i in range(args.skip_start, len(timestamps)-args.skip_end, args.step):
        row = timestamps[i]
        times = [int(row[c].split('_')[3]) for c in ('L','R','SL','SR')]
        if times[0] != times[1] or times[2] != times[3]:
            raise ValueError(f'Stereo-eye timestamps differ at {sequence.sequence_id}:{i}')
        if abs(times[0]-times[2]) <= args.max_skew_ms:
            selected.append(i)
    if args.max_frames and len(selected) > args.max_frames:
        picks = np.unique(np.linspace(0,len(selected)-1,args.max_frames).round().astype(int))
        selected = [selected[i] for i in picks]
    if not selected:
        raise ValueError(f'No eligible timestamp-aligned frames: {sequence.sequence_id}')
    return selected


def main(task):
    parser = argparse.ArgumentParser(description=f'Convert public iMED into the existing {task.upper()} task layout.')
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--sequence-id',action='append',help='Released session/scene/sequence; repeat to select several.')
    parser.add_argument('--split',choices=['train','test'],help='Optional output subdirectory; no automatic split assignment.')
    parser.add_argument('--step',type=int,default=3)
    parser.add_argument('--max-frames',type=int,default=100 if task=='pe' else 0,help='0 exports every eligible sampled frame.')
    parser.add_argument('--skip-start',type=int,default=2)
    parser.add_argument('--skip-end',type=int,default=2)
    parser.add_argument('--max-skew-ms',type=float,default=10)
    parser.add_argument('--dry-run',action='store_true')
    if task=='nvs':
        parser.add_argument('--foundation-stereo-dir',type=Path,help='Optional model checkout with local weights; generate depthL in mm.')
        parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.step < 1 or min(args.max_frames,args.skip_start,args.skip_end,args.max_skew_ms)<0:
        parser.error('step must be positive; other sampling values must be nonnegative')
    dataset=IMEDDataset(args.root)
    eligible={s.sequence_id:s for s in dataset.get_task_sequences(task)}
    selected=sorted(eligible) if args.sequence_id is None else args.sequence_id
    if not selected or len(set(selected))!=len(selected): raise ValueError('Empty or duplicate sequence selection')
    plans=[]
    for sid in selected:
        if sid not in eligible: raise ValueError(f'Sequence is unavailable or ineligible for {task}: {sid}')
        sequence=eligible[sid]
        record=dataset.load_trajectory(sid)
        kind=record['metadata'].get('pose_type')
        if kind != ('dynamic' if task=='pe' else 'static'):
            raise ValueError(f'{task} requires a validated {"dynamic" if task=="pe" else "static"} pose: {sid}')
        frames=select_frames(dataset,sequence,args)
        transforms=dataset.get_poses(sid,frames)
        destination=args.output/(args.split or '')/sid.replace('/','_')
        if destination.exists(): raise FileExistsError(destination)
        if destination.resolve().is_relative_to(args.root.resolve()): raise ValueError('Task output must be outside the source release')
        plans.append((sequence,record,frames,transforms,destination))
    if args.dry_run:
        print(json.dumps(dict(task=task,sequences=len(plans),frames=sum(len(p[2]) for p in plans),sequence_ids=selected),indent=2))
        return
    model=padder=None
    if task=='nvs' and args.foundation_stereo_dir:
        from run_foundation_stereo_depths import load_foundation_stereo_model, run_foundation_stereo_inference
        model,padder=load_foundation_stereo_model(args.foundation_stereo_dir,args.device)
    for sequence,record,frames,transforms,destination in plans:
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.partial')
        temporary.mkdir()  # Interrupted output is never silently reused.
        calibration=dataset.get_calibration(sequence.session_id)
        with (temporary/'K.txt').open('w') as f:
            f.write('# Camera intrinsic matrices (3x3), native image resolution\n')
            for label,key in [('K1_L','L'),('K1_R','R'),('K2_L','SL'),('K2_R','SR')]:
                f.write('# '+label+'\n');np.savetxt(f,calibration[key][0],fmt='%.10f')
        if task=='pe':
            anchor=np.linalg.inv(transforms[0])
            rows=[pose_row(i,anchor@t) for i,t in zip(frames,transforms)]
            convention='inverse(T_SL_from_L(first exported frame)) @ T_SL_from_L(frame)'
        else:
            assert all(np.allclose(t,transforms[0]) for t in transforms)
            rows=[pose_row(0,np.eye(4)),pose_row(1,transforms[0])]
            convention='world=SL; row 0 is endoscope2 left identity; row 1 is endoscope1 left camera-to-world T_SL_from_L'
        np.savetxt(temporary/'pose.txt',rows,fmt=['%d']+['%.10f']*7)
        for camera,keys in [('endoscope1',('L','R')),('endoscope2',('SL','SR'))]:
            for view in ('L','R'):
                (temporary/camera/view).mkdir(parents=True)
            if task=='nvs':
                for name in ('toolL','toolR','depthL'): (temporary/camera/name).mkdir()
            for i in frames:
                left,right=dataset.get_stereo_pair(sequence.sequence_id,i,camera=camera)
                for view,frame in [('L',left),('R',right)]:
                    if not cv2.imwrite(str(temporary/camera/view/f'frame_{i:06d}.png'),frame):
                        raise RuntimeError('Image write failed')
                if task=='nvs':
                    if sequence.metadata['tools_present']:
                        from endoscope_dataloader import read_video_frame, split_stacked_stereo
                        masks=split_stacked_stereo(read_video_frame(sequence.tool_mask_path(camera),i))
                    else:
                        masks=[np.zeros(left.shape[:2],np.uint8),np.zeros(right.shape[:2],np.uint8)]
                    for name,mask in zip(('toolL','toolR'),masks):
                        if mask.ndim==3:mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                        assert mask.shape==left.shape[:2], 'Image/mask resolution mismatch'
                        if not cv2.imwrite(str(temporary/camera/name/f'frame_{i:06d}.png'),(mask>127).astype(np.uint8)*255):
                            raise RuntimeError('Mask write failed')
                    if model is not None:
                        depth=run_foundation_stereo_inference(model,padder,left,right,calibration[keys[0]][0],dataset.get_scope_baseline(),args.device)
                        assert depth.shape==left.shape[:2] and np.isfinite(depth).all()
                        np.save(temporary/camera/'depthL'/f'frame_{i:06d}.npy',depth.astype(np.float32)*1000)
        sources=[record['path'],sequence.relative_pose_file,sequence.timestamp_csv,*sequence.videos.values()]
        if task=='nvs':sources+=list(sequence.tool_masks.values())
        metadata=dict(task=task,sequence_id=sequence.sequence_id,frames=frames,pose_convention=convention,
            translation_unit='mm',quaternion_order='xyzw',image_resize=False,
            source_sha256={str(p.relative_to(args.root)):sha(p) for p in sources},
            settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('root','output','foundation_stereo_dir')},
            depth='generated_mm' if model is not None else 'not_generated',
            implementation_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).resolve().parents[1]/'imed_dataset_loader.py',Path(__file__).resolve().parents[1]/'endoscope_dataloader.py']},
            calibration_sha256={k:sha(args.root/'calibrations'/sequence.session_id/f'{k}_calibration_chessboard.yaml') for k in calibration})
        (temporary/'export_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        temporary.rename(destination)
        print(f'Exported {sequence.sequence_id}: {len(frames)} frames',flush=True)
