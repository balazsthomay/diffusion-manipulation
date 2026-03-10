"""Tests for linear normalizer module."""

import numpy as np
import pytest
import torch

from diffusion_manipulation.data.normalizer import LinearNormalizer


class TestLinearNormalizer:
    def setup_method(self) -> None:
        self.rng = np.random.default_rng(42)
        self.data = {
            "actions": self.rng.uniform(-2, 2, size=(100, 7)).astype(np.float32),
            "lowdim_obs": self.rng.uniform(-5, 5, size=(100, 9)).astype(np.float32),
        }

    def test_fit_sets_fitted(self) -> None:
        norm = LinearNormalizer()
        assert not norm.is_fitted
        norm.fit(self.data)
        assert norm.is_fitted

    def test_keys_after_fit(self) -> None:
        norm = LinearNormalizer()
        norm.fit(self.data)
        assert set(norm.keys) == {"actions", "lowdim_obs"}

    def test_normalize_range(self) -> None:
        norm = LinearNormalizer()
        norm.fit(self.data)

        normalized = norm.normalize("actions", self.data["actions"])
        assert normalized.min() >= -1.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_round_trip_numpy(self) -> None:
        norm = LinearNormalizer()
        norm.fit(self.data)

        original = self.data["actions"]
        normalized = norm.normalize("actions", original)
        recovered = norm.unnormalize("actions", normalized)

        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_round_trip_torch(self) -> None:
        norm = LinearNormalizer()
        norm.fit(self.data)

        original = torch.from_numpy(self.data["actions"])
        normalized = norm.normalize("actions", original)
        recovered = norm.unnormalize("actions", normalized)

        assert isinstance(recovered, torch.Tensor)
        torch.testing.assert_close(recovered, original, atol=1e-5, rtol=1e-5)

    def test_normalize_batched(self) -> None:
        """Test normalization with extra batch dimensions."""
        norm = LinearNormalizer()
        norm.fit(self.data)

        batched = self.data["actions"].reshape(10, 10, 7)
        normalized = norm.normalize("actions", batched)
        assert normalized.shape == (10, 10, 7)

        recovered = norm.unnormalize("actions", normalized)
        np.testing.assert_allclose(recovered, batched, atol=1e-5)

    def test_state_dict_round_trip(self) -> None:
        norm = LinearNormalizer()
        norm.fit(self.data)

        state = norm.state_dict()
        norm2 = LinearNormalizer()
        norm2.load_state_dict(state)

        original = self.data["lowdim_obs"]
        result1 = norm.normalize("lowdim_obs", original)
        result2 = norm2.normalize("lowdim_obs", original)

        np.testing.assert_array_equal(result1, result2)

    def test_normalize_before_fit_raises(self) -> None:
        norm = LinearNormalizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            norm.normalize("actions", self.data["actions"])

    def test_unknown_key_raises(self) -> None:
        norm = LinearNormalizer()
        norm.fit(self.data)
        with pytest.raises(KeyError, match="Unknown key"):
            norm.normalize("nonexistent", self.data["actions"])

    def test_constant_data_handling(self) -> None:
        """Constant data should not cause division by zero."""
        norm = LinearNormalizer()
        constant_data = {"const": np.ones((50, 3), dtype=np.float32) * 5.0}
        norm.fit(constant_data)

        normalized = norm.normalize("const", constant_data["const"])
        assert np.all(np.isfinite(normalized))
