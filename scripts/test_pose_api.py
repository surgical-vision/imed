"""Regression tests for pose coverage, units, frame selection, and task conventions."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imed_dataset_loader import IMEDDataset
from convert_task import pose_row, select_frames


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.sid='session_001/scene_1/motion'
        self.folder=self.root/self.sid;self.folder.mkdir(parents=True)
        (self.root/'dataset_metadata.json').write_text(json.dumps(dict(pixels_undistorted=True)))
        metadata=dict(anatomical_regions=[dict(region_name='scene_1',sequences=[dict(sequence_name='motion',frame_count=6)])])
        (self.root/'session_001/session_metadata.json').write_text(json.dumps(metadata))
        self.pose_meta=dict(trajectory_file='trajectory_poses.txt',pose_type='dynamic')
        self.values=np.column_stack((np.arange(6),np.tile(np.eye(4).reshape(1,16),(6,1))))
        self.values[:,4]=np.arange(6)*.001
        self.write_pose();self.dataset=IMEDDataset(self.root)

    def write_pose(self):
        (self.folder/'relative_poses.json').write_text(json.dumps(self.pose_meta))
        np.savetxt(self.folder/'trajectory_poses.txt',self.values,header='frame_idx '+' '.join(f'T{i}{j}' for i in range(4) for j in range(4)),comments='')

    def test_nonsequential_pose_queries_keep_video_indices_and_metres(self):
        poses=self.dataset.get_poses(self.sid,[5,0,3,3])
        np.testing.assert_allclose(poses[:,0,3],[.005,0,.003,.003])
        np.testing.assert_allclose(self.dataset.get_pose(self.sid,5),poses[0])
        for bad in [-1,1.5,6]:
            with self.assertRaises((ValueError,IndexError)):self.dataset.get_pose(self.sid,bad)

    def test_static_interval_is_not_broadcast_outside_supported_frames(self):
        self.values=self.values[4:5];self.pose_meta.update(pose_type='static',valid_frame_range=[4,5]);self.write_pose()
        np.testing.assert_allclose(self.dataset.get_poses(self.sid,[4,5])[:,0,3],[.004,.004])
        with self.assertRaisesRegex(ValueError,'valid range'):self.dataset.get_pose(self.sid,3)

    def test_missing_pose_is_not_interpolated(self):
        self.values=self.values[[0,2,5]];self.write_pose()
        with self.assertRaisesRegex(ValueError,'unavailable'):self.dataset.get_pose(self.sid,1)

    def test_invalid_rotations_duplicate_indices_and_nan_fail(self):
        original=self.values.copy()
        for mutate in [lambda a:a.__setitem__((1,0),0),lambda a:a.__setitem__((0,1),2),lambda a:a.__setitem__((0,4),np.nan)]:
            self.values=original.copy();mutate(self.values);self.write_pose()
            with self.assertRaises(ValueError):self.dataset.load_trajectory(self.sid)

    def test_pose_path_cannot_escape(self):
        self.pose_meta['trajectory_file']='../trajectory_poses.txt';self.write_pose()
        with self.assertRaises(ValueError):self.dataset.load_trajectory(self.sid)

    def test_task_pose_translation_and_rotation_are_unambiguous(self):
        t=np.eye(4);t[:3,:3]=cv2.Rodrigues(np.array([0.,0.,np.pi]))[0];t[:3,3]=[.001,-.002,.003]
        row=pose_row(9,t)
        np.testing.assert_allclose(row[:4],[9,1,-2,3]);np.testing.assert_allclose(np.abs(row[4:]),[0,0,1,0],atol=1e-9)
        identity=pose_row(0,np.linalg.inv(t)@t);np.testing.assert_allclose(identity,[0,0,0,0,0,0,0,1],atol=1e-9)

    def test_timestamp_selection_uses_video_rows_not_capture_ids(self):
        rows=[]
        for i,delta in enumerate([1,20,3,4,30,2]):
            rows.append(dict(index=str(i),**{c:f'frame_000_{100+i}_{1000+i*30+(delta if c in ("SL","SR") else 0)}_0.bmp' for c in ['L','R','SL','SR']}))
        fake=SimpleNamespace(read_timestamps=lambda _:rows)
        opts=SimpleNamespace(skip_start=0,skip_end=0,step=1,max_skew_ms=10,max_frames=0)
        self.assertEqual(select_frames(fake,SimpleNamespace(sequence_id=self.sid),opts),[0,2,3,5])
        opts.max_skew_ms=0
        with self.assertRaisesRegex(ValueError,'No eligible'):select_frames(fake,SimpleNamespace(sequence_id=self.sid),opts)

    def test_image_calibration_zeroes_distortion_without_mutating_lens_record(self):
        from unittest.mock import patch
        d=np.ones(5)
        with patch.object(self.dataset,'get_calibration',return_value={'L':(np.eye(3),d)}):
            np.testing.assert_array_equal(self.dataset.get_image_calibration('session_001')['L'][1],np.zeros(5))
            np.testing.assert_array_equal(d,np.ones(5))

    def test_optional_depth_units_and_invalid_disparity(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest('Optional PyTorch dependency is not installed')
        from run_foundation_stereo_depths import run_foundation_stereo_inference
        class Padder:
            def __init__(self,*args,**kwargs):pass
            def pad(self,a,b):return a,b
            def unpad(self,a):return a
        class Model:
            def __init__(self,value):self.value=value
            def forward(self,left,right,**kwargs):
                return torch.full((1,1,*left.shape[-2:]),self.value)
        image=np.zeros((16,16,3),np.uint8)
        for disparity,expected in [(0.,0.),(-1.,0.),(4.,.05125)]:
            depth=run_foundation_stereo_inference(Model(disparity),Padder,image,image,np.diag([100.,100.,1.]),.0041,'cpu')
            self.assertEqual(depth.shape,(16,16))
            np.testing.assert_allclose(depth,expected,atol=1e-7)


if __name__=='__main__':unittest.main()
