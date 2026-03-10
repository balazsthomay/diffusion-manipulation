"""Tests for EMA model."""

import torch
import torch.nn as nn

from diffusion_manipulation.training.ema import EMAModel


class TestEMAModel:
    def _make_model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

    def test_shadow_initialized(self) -> None:
        model = self._make_model()
        ema = EMAModel(model, decay=0.99)
        assert len(ema.shadow) > 0

        # Shadow should match initial params
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.equal(ema.shadow[name], param.data)

    def test_update_moves_shadow(self) -> None:
        model = self._make_model()
        ema = EMAModel(model, decay=0.99)

        # Store initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Update model params (simulate a gradient step)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        ema.update(model)

        # Shadow should have moved toward new params
        for name in ema.shadow:
            assert not torch.equal(ema.shadow[name], initial_shadow[name])

    def test_decay_rate(self) -> None:
        model = self._make_model()
        decay = 0.9
        ema = EMAModel(model, decay=decay)

        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Set model params to zeros
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

        ema.update(model)

        # shadow = decay * old_shadow + (1-decay) * 0 = decay * old_shadow
        for name in ema.shadow:
            expected = old_shadow[name] * decay
            torch.testing.assert_close(ema.shadow[name], expected)

    def test_apply_and_restore(self) -> None:
        model = self._make_model()
        ema = EMAModel(model, decay=0.99)

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param))

        ema.update(model)

        # Store current params
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        # Apply shadow
        ema.apply_shadow(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.equal(param.data, ema.shadow[name])

        # Restore original
        ema.restore(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.equal(param.data, original_params[name])

    def test_state_dict_roundtrip(self) -> None:
        model = self._make_model()
        ema = EMAModel(model, decay=0.99)

        # Update a few times
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema.update(model)

        state = ema.state_dict()

        ema2 = EMAModel(model, decay=0.99)
        ema2.load_state_dict(state)

        for name in ema.shadow:
            torch.testing.assert_close(ema.shadow[name], ema2.shadow[name])
