"""Tests for dataset download module."""

from pathlib import Path

import h5py
import pytest

from diffusion_manipulation.data.download import (
    AVAILABLE_TASKS,
    DATASET_URLS,
    download_dataset,
    verify_dataset,
)


class TestDownloadConfig:
    def test_available_tasks(self) -> None:
        assert "lift" in AVAILABLE_TASKS
        assert "can" in AVAILABLE_TASKS
        assert "square" in AVAILABLE_TASKS

    def test_urls_exist_for_all_tasks(self) -> None:
        for task in AVAILABLE_TASKS:
            assert "ph_low_dim" in DATASET_URLS[task]
            assert "ph_image" in DATASET_URLS[task]


class TestDownloadDataset:
    def test_invalid_task_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            download_dataset("nonexistent", tmp_path)

    def test_invalid_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown dataset type"):
            download_dataset("lift", tmp_path, dataset_type="bad_type")

    def test_skip_existing(self, tmp_path: Path) -> None:
        # Create a fake file
        out_dir = tmp_path / "lift"
        out_dir.mkdir()
        fake_file = out_dir / "ph_image.hdf5"
        fake_file.write_text("fake")

        result = download_dataset("lift", tmp_path, dataset_type="ph_image")
        assert result == fake_file


class TestVerifyDataset:
    def test_valid_hdf5(self, tmp_hdf5: Path) -> None:
        assert verify_dataset(tmp_hdf5) is True

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert verify_dataset(tmp_path / "nope.hdf5") is False

    def test_invalid_hdf5(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.hdf5"
        bad.write_text("not hdf5")
        assert verify_dataset(bad) is False

    def test_hdf5_without_data_key(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.hdf5"
        with h5py.File(path, "w") as f:
            f.create_group("other")
        assert verify_dataset(path) is False
