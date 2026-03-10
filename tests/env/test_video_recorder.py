"""Tests for video recorder."""

from pathlib import Path

import numpy as np

from diffusion_manipulation.env.video_recorder import VideoRecorder


class TestVideoRecorder:
    def test_recording_lifecycle(self) -> None:
        rec = VideoRecorder()
        assert not rec.is_recording
        assert rec.num_frames == 0

        rec.start()
        assert rec.is_recording

        for _ in range(5):
            rec.add_frame(np.zeros((84, 84, 3), dtype=np.uint8))
        assert rec.num_frames == 5

        rec.stop()
        assert not rec.is_recording

    def test_add_frame_when_not_recording(self) -> None:
        rec = VideoRecorder()
        rec.add_frame(np.zeros((84, 84, 3), dtype=np.uint8))
        assert rec.num_frames == 0

    def test_save_gif(self, tmp_path: Path) -> None:
        rec = VideoRecorder()
        rec.start()
        for i in range(10):
            frame = np.full((64, 64, 3), i * 25, dtype=np.uint8)
            rec.add_frame(frame)
        rec.stop()

        path = rec.save_gif(tmp_path / "test.gif")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_mp4(self, tmp_path: Path) -> None:
        rec = VideoRecorder()
        rec.start()
        for i in range(10):
            frame = np.full((64, 64, 3), i * 25, dtype=np.uint8)
            rec.add_frame(frame)
        rec.stop()

        path = rec.save_mp4(tmp_path / "test.mp4")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_reset(self) -> None:
        rec = VideoRecorder()
        rec.start()
        rec.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        rec.reset()
        assert not rec.is_recording
        assert rec.num_frames == 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        rec = VideoRecorder()
        rec.start()
        rec.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        rec.stop()

        path = rec.save_gif(tmp_path / "nested" / "dir" / "test.gif")
        assert path.exists()
