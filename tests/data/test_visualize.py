"""Tests for visualization module."""

from pathlib import Path

import numpy as np

from diffusion_manipulation.data.replay_buffer import ReplayBuffer, load_replay_buffer
from diffusion_manipulation.data.visualize import (
    plot_action_distributions,
    plot_episode_actions,
    render_demo_gif,
)


class TestRenderDemoGif:
    def test_creates_gif(self, tmp_hdf5: Path, tmp_path: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        out = tmp_path / "demo.gif"

        result = render_demo_gif(buf, episode_idx=0, camera_name="agentview", output_path=out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_hdf5: Path, tmp_path: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        out = tmp_path / "nested" / "dir" / "demo.gif"

        result = render_demo_gif(buf, episode_idx=0, camera_name="agentview", output_path=out)
        assert result.exists()


class TestPlotActionDistributions:
    def test_creates_plot(self, tmp_hdf5: Path, tmp_path: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        out = tmp_path / "actions.png"

        result = plot_action_distributions(buf, output_path=out)
        assert result.exists()
        assert result.stat().st_size > 0


class TestPlotEpisodeActions:
    def test_creates_plot(self, tmp_hdf5: Path, tmp_path: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        out = tmp_path / "episode_0.png"

        result = plot_episode_actions(buf, episode_idx=0, output_path=out)
        assert result.exists()
        assert result.stat().st_size > 0
