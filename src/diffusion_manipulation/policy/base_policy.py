"""Abstract base policy interface."""

from abc import ABC, abstractmethod

import torch

from diffusion_manipulation.data.normalizer import LinearNormalizer


class BasePolicy(ABC, torch.nn.Module):
    """Abstract base class for manipulation policies."""

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the training loss for a batch.

        Args:
            batch: Dict with observation and action tensors.

        Returns:
            Scalar loss tensor.
        """

    @abstractmethod
    def predict_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict actions given observations.

        Args:
            obs: Dict with observation tensors (no batch dim expected
                 from env, but policy may add it internally).

        Returns:
            (action_horizon, action_dim) action tensor.
        """

    @abstractmethod
    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set the data normalizer for this policy.

        Args:
            normalizer: Fitted normalizer for actions and observations.
        """
