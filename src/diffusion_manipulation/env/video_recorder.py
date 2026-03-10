"""Episode video recording utilities."""

from pathlib import Path

import imageio
import numpy as np
import numpy.typing as npt


class VideoRecorder:
    """Records episode frames and saves as GIF or MP4."""

    def __init__(self) -> None:
        self._frames: list[npt.NDArray[np.uint8]] = []
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def start(self) -> None:
        """Start a new recording, clearing any previous frames."""
        self._frames = []
        self._recording = True

    def add_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Add a frame to the recording.

        Args:
            frame: (H, W, C) uint8 image array.
        """
        if not self._recording:
            return
        self._frames.append(frame.copy())

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False

    def save_gif(self, path: str | Path, fps: int = 20) -> Path:
        """Save recorded frames as a GIF.

        Args:
            path: Output file path.
            fps: Frames per second.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        duration_ms = 1000 / fps
        imageio.mimsave(
            str(path),
            self._frames,
            duration=duration_ms,
            loop=0,
        )
        return path

    def save_mp4(self, path: str | Path, fps: int = 20) -> Path:
        """Save recorded frames as MP4.

        Args:
            path: Output file path.
            fps: Frames per second.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = imageio.get_writer(str(path), fps=fps)
        for frame in self._frames:
            writer.append_data(frame)
        writer.close()
        return path

    def reset(self) -> None:
        """Clear all frames and stop recording."""
        self._frames = []
        self._recording = False
