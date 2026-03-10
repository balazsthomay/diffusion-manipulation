"""Download robomimic HDF5 datasets for training."""

import hashlib
import urllib.request
from pathlib import Path

from tqdm import tqdm

# robomimic dataset URLs (proficient-human, low_dim + images)
DATASET_URLS: dict[str, dict[str, str]] = {
    "lift": {
        "ph_low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/low_dim_v141.hdf5",
        "ph_image": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/image_v141.hdf5",
    },
    "can": {
        "ph_low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim_v141.hdf5",
        "ph_image": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/image_v141.hdf5",
    },
    "square": {
        "ph_low_dim": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/low_dim_v141.hdf5",
        "ph_image": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/image_v141.hdf5",
    },
}

AVAILABLE_TASKS = tuple(DATASET_URLS.keys())


class _DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_dataset(
    task_name: str,
    dataset_dir: str | Path,
    dataset_type: str = "ph_image",
    force: bool = False,
) -> Path:
    """Download a robomimic dataset.

    Args:
        task_name: Task name (lift, can, square).
        dataset_dir: Directory to save datasets.
        dataset_type: Dataset variant (ph_low_dim, ph_image).
        force: Re-download even if file exists.

    Returns:
        Path to the downloaded HDF5 file.
    """
    if task_name not in DATASET_URLS:
        raise ValueError(f"Unknown task: {task_name}. Available: {AVAILABLE_TASKS}")

    task_urls = DATASET_URLS[task_name]
    if dataset_type not in task_urls:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {tuple(task_urls.keys())}")

    url = task_urls[dataset_type]
    dataset_dir = Path(dataset_dir)
    output_dir = dataset_dir / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_type}.hdf5"
    output_path = output_dir / filename

    if output_path.exists() and not force:
        print(f"Dataset already exists: {output_path}")
        return output_path

    print(f"Downloading {task_name}/{dataset_type} from {url}")
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as pbar:
        urllib.request.urlretrieve(url, output_path, reporthook=pbar.update_to)

    print(f"Saved to {output_path}")
    return output_path


def verify_dataset(path: str | Path) -> bool:
    """Verify that a downloaded HDF5 file is valid."""
    import h5py

    path = Path(path)
    if not path.exists():
        return False

    try:
        with h5py.File(path, "r") as f:
            return "data" in f
    except Exception:
        return False
