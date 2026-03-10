"""Tests for VisionEncoder."""

import torch

from diffusion_manipulation.model.vision_encoder import VisionEncoder


class TestVisionEncoder:
    def _make_encoder(self, **kwargs) -> VisionEncoder:
        defaults = dict(
            input_shape=(84, 84),
            crop_shape=(76, 76),
            feature_dim=512,
            pretrained=False,  # Faster for tests
            imagenet_norm=True,
        )
        defaults.update(kwargs)
        return VisionEncoder(**defaults)

    def test_output_shape_single_camera(self) -> None:
        enc = self._make_encoder()
        B, To = 2, 2
        images = {"agentview": torch.randn(B, To, 3, 84, 84)}
        lowdim = torch.randn(B, To, 9)

        out = enc(images, lowdim)
        # To * 1_camera * 512 + To * 9 = 2*512 + 2*9 = 1042
        assert out.shape == (B, 1042)

    def test_output_shape_multi_camera(self) -> None:
        enc = self._make_encoder(num_cameras=2)
        B, To = 2, 2
        images = {
            "agentview": torch.randn(B, To, 3, 84, 84),
            "eye_in_hand": torch.randn(B, To, 3, 84, 84),
        }
        lowdim = torch.randn(B, To, 9)

        out = enc(images, lowdim)
        # To * 2_cameras * 512 + To * 9 = 2*2*512 + 2*9 = 2066
        assert out.shape == (B, 2066)

    def test_no_images(self) -> None:
        enc = self._make_encoder()
        B, To = 2, 2
        lowdim = torch.randn(B, To, 9)

        out = enc({}, lowdim)
        assert out.shape == (B, To * 9)

    def test_gradient_flow(self) -> None:
        enc = self._make_encoder()
        images = {"agentview": torch.randn(1, 2, 3, 84, 84, requires_grad=True)}
        lowdim = torch.randn(1, 2, 9, requires_grad=True)

        out = enc(images, lowdim)
        out.sum().backward()

        assert images["agentview"].grad is not None
        assert lowdim.grad is not None

    def test_train_vs_eval_mode(self) -> None:
        enc = self._make_encoder()
        images = {"agentview": torch.randn(1, 2, 3, 84, 84)}
        lowdim = torch.randn(1, 2, 9)

        enc.train()
        with torch.no_grad():
            out_train = enc(images, lowdim)

        enc.eval()
        with torch.no_grad():
            out_eval = enc(images, lowdim)

        # Should differ due to random vs center crop
        assert out_train.shape == out_eval.shape

    def test_get_output_dim(self) -> None:
        enc = self._make_encoder()
        dim = enc.get_output_dim(obs_horizon=2, lowdim_dim=9, num_cameras=1)
        assert dim == 2 * 512 + 2 * 9  # 1042
