"""Linear normalizer for action and observation data."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class NormalizerStats:
    """Statistics for a single data key."""

    min: npt.NDArray[np.float32]
    max: npt.NDArray[np.float32]


class LinearNormalizer:
    """Min-max normalizer mapping data to [-1, 1].

    Supports fitting on numpy arrays and normalizing/unnormalizing
    both numpy arrays and torch tensors.
    """

    def __init__(self) -> None:
        self._stats: dict[str, NormalizerStats] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._stats.keys())

    def fit(self, data: dict[str, npt.NDArray[np.float32]], eps: float = 1e-6) -> None:
        """Fit normalizer statistics from data.

        Args:
            data: Dict mapping key names to arrays of shape (N, D).
            eps: Small value to prevent division by zero.
        """
        self._stats = {}
        for key, arr in data.items():
            # Flatten all leading dims except last
            flat = arr.reshape(-1, arr.shape[-1])
            data_min = flat.min(axis=0).astype(np.float32)
            data_max = flat.max(axis=0).astype(np.float32)

            # Prevent zero range
            zero_range = (data_max - data_min) < eps
            data_min[zero_range] = data_min[zero_range] - eps
            data_max[zero_range] = data_max[zero_range] + eps

            self._stats[key] = NormalizerStats(min=data_min, max=data_max)

        self._fitted = True

    def normalize(self, key: str, data: npt.NDArray | torch.Tensor) -> npt.NDArray | torch.Tensor:
        """Normalize data to [-1, 1] range.

        Args:
            key: The data key to use for normalization stats.
            data: Array or tensor of shape (..., D).

        Returns:
            Normalized data in [-1, 1].
        """
        stats = self._get_stats(key)

        if isinstance(data, torch.Tensor):
            d_min = torch.tensor(stats.min, dtype=data.dtype, device=data.device)
            d_max = torch.tensor(stats.max, dtype=data.dtype, device=data.device)
        else:
            d_min = stats.min
            d_max = stats.max

        # Scale to [0, 1] then to [-1, 1]
        normalized = (data - d_min) / (d_max - d_min)
        return normalized * 2.0 - 1.0

    def unnormalize(self, key: str, data: npt.NDArray | torch.Tensor) -> npt.NDArray | torch.Tensor:
        """Unnormalize data from [-1, 1] back to original range.

        Args:
            key: The data key to use for normalization stats.
            data: Normalized array or tensor of shape (..., D).

        Returns:
            Data in original scale.
        """
        stats = self._get_stats(key)

        if isinstance(data, torch.Tensor):
            d_min = torch.tensor(stats.min, dtype=data.dtype, device=data.device)
            d_max = torch.tensor(stats.max, dtype=data.dtype, device=data.device)
        else:
            d_min = stats.min
            d_max = stats.max

        # Scale from [-1, 1] to [0, 1] then to original range
        normalized_01 = (data + 1.0) / 2.0
        return normalized_01 * (d_max - d_min) + d_min

    def state_dict(self) -> dict[str, dict[str, npt.NDArray[np.float32]]]:
        """Export normalizer state for serialization."""
        return {
            key: {"min": stats.min.copy(), "max": stats.max.copy()}
            for key, stats in self._stats.items()
        }

    def load_state_dict(self, state: dict[str, dict[str, npt.NDArray[np.float32]]]) -> None:
        """Load normalizer state from a serialized dict."""
        self._stats = {
            key: NormalizerStats(
                min=np.array(s["min"], dtype=np.float32),
                max=np.array(s["max"], dtype=np.float32),
            )
            for key, s in state.items()
        }
        self._fitted = True

    def _get_stats(self, key: str) -> NormalizerStats:
        if not self._fitted:
            raise RuntimeError("Normalizer has not been fitted yet. Call fit() first.")
        if key not in self._stats:
            raise KeyError(f"Unknown key: {key}. Available: {self.keys}")
        return self._stats[key]
