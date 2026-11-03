#!/usr/bin/env python3
"""Pixel-level video indexing regression; generates temporary MP4 fixtures."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endoscope_dataloader import read_video_frame
from imed_dataset_loader import IMEDDataset


class VideoIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.sequence_id = 'session_001/scene_1/circular'
        sequence = root / self.sequence_id
        sequence.mkdir(parents=True)
        self.reference = {}
        for raw in (False, True):
            for camera in ('endoscope1', 'endoscope2'):
                path = sequence / f"{camera}{'_raw' if raw else ''}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
                self.assertTrue(writer.isOpened())
                try:
                    for frame_idx in range(12):
                        frame = np.zeros((64, 64, 3), np.uint8)
                        for eye in range(2):
                            frame[eye*32:(eye+1)*32] = (20+frame_idx*16, 40+eye*140, 30+int(raw)*80+int(camera[-1])*40)
                        writer.write(frame)
                finally:
                    writer.release()
                # Independent sequential decoding: no seek and no loader helpers.
                cap = cv2.VideoCapture(str(path))
                frames = []
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frames.append(frame)
                finally:
                    cap.release()
                self.assertEqual(len(frames), 12)
                self.reference[raw, camera] = frames
        calib = root / 'calibrations/session_001'
        calib.mkdir(parents=True)
        for camera in ('L', 'R', 'SL', 'SR'):
            storage = cv2.FileStorage(str(calib / f'{camera}_calibration_chessboard.yaml'), cv2.FILE_STORAGE_WRITE)
            storage.write('K', np.eye(3))
            storage.write('D', np.zeros((1, 5)))
            storage.release()
        self.dataset = IMEDDataset(root)

    def test_all_eyes_sequential_repeated_backward_and_last_frame(self):
        for raw in (False, True):
            for index in [0, 1, 2, 2, 8, 3, 11, 0]:
                actual = self.dataset.get_frame_from_video(self.sequence_id, index, raw=raw)
                self.assertEqual(actual['frame_index'], index)
                for camera, eyes in [('endoscope1', ('L', 'R')), ('endoscope2', ('SL', 'SR'))]:
                    expected = self.reference[raw, camera][index]
                    stacked = self.dataset.get_frame(self.sequence_id, index, camera=camera, split=False, raw=raw)
                    np.testing.assert_array_equal(stacked, expected)
                    for eye, half in zip(eyes, np.split(expected, 2)):
                        np.testing.assert_array_equal(actual[eye], half)

    def test_invalid_indices_fail(self):
        for raw in (False, True):
            with self.assertRaises(ValueError):
                self.dataset.get_frame(self.sequence_id, -1, raw=raw)
            with self.assertRaises(ValueError):
                self.dataset.get_frame(self.sequence_id, 1.5, raw=raw)
            with self.assertRaises(IndexError):
                self.dataset.get_frame(self.sequence_id, 12, raw=raw)

    def test_failed_seek_and_wrong_decoder_position_fail(self):
        path = self.dataset.get_video_path(self.sequence_id)
        with patch('endoscope_dataloader.cv2.VideoCapture') as factory:
            cap = factory.return_value
            cap.isOpened.return_value = True
            cap.set.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'seek'):
                read_video_frame(path, 3)
            cap.release.assert_called_once()
            cap.set.return_value = True
            cap.read.return_value = (True, self.reference[False, 'endoscope1'][4])
            cap.get.return_value = 5
            with self.assertRaisesRegex(RuntimeError, 'wrong frame position'):
                read_video_frame(path, 3)


if __name__ == '__main__':
    unittest.main()
