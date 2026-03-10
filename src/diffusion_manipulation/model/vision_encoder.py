"""ResNet18-based vision encoder for image observations."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """Replace all BatchNorm layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            _replace_bn_with_gn(child, num_groups)
    return module


class VisionEncoder(nn.Module):
    """ResNet18-based encoder for image observations.

    Processes image observations from one or more cameras across multiple
    timesteps, producing a flat global conditioning vector.

    For each camera and timestep, images are:
    1. Random cropped (train) or center cropped (eval) from input_shape to crop_shape
    2. Optionally normalized with ImageNet stats
    3. Passed through ResNet18 backbone to get feature_dim features
    4. Concatenated across cameras and timesteps with low-dim observations
    """

    def __init__(
        self,
        input_shape: tuple[int, int] = (84, 84),
        crop_shape: tuple[int, int] = (76, 76),
        feature_dim: int = 512,
        pretrained: bool = True,
        imagenet_norm: bool = True,
        num_cameras: int = 1,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.crop_shape = crop_shape
        self.feature_dim = feature_dim
        self.num_cameras = num_cameras

        # Build ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        _replace_bn_with_gn(resnet)

        # Remove final FC and avgpool, replace with adaptive pool + flatten + projection
        resnet_out_dim = 512  # ResNet18 fixed output dim
        backbone_layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
        if feature_dim != resnet_out_dim:
            backbone_layers.append(nn.Linear(resnet_out_dim, feature_dim))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)

        # Augmentation transforms
        self.train_transform = transforms.RandomCrop(crop_shape)
        self.eval_transform = transforms.CenterCrop(crop_shape)

        # ImageNet normalization
        self.imagenet_norm = imagenet_norm
        if imagenet_norm:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

    def forward(
        self,
        images: dict[str, torch.Tensor],
        lowdim: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image and low-dim observations into a global conditioning vector.

        Args:
            images: Dict of camera_name -> (B, To, C, H, W) image tensors in [0, 1].
            lowdim: (B, To, D) low-dim observation tensor.

        Returns:
            (B, global_cond_dim) conditioning vector.
        """
        B, To = lowdim.shape[:2]
        features = []

        for cam_name, img in images.items():
            # (B, To, C, H, W) -> (B*To, C, H, W)
            img_flat = img.reshape(B * To, *img.shape[2:])

            # Apply crop augmentation
            if self.training:
                img_flat = self.train_transform(img_flat)
            else:
                img_flat = self.eval_transform(img_flat)

            # Apply ImageNet normalization
            if self.imagenet_norm:
                img_flat = self.normalize(img_flat)

            # Extract features: (B*To, feature_dim)
            feat = self.backbone(img_flat)
            features.append(feat)

        # Concat image features: (B*To, num_cameras * feature_dim)
        if features:
            img_features = torch.cat(features, dim=-1)
            # Reshape to (B, To * num_cameras * feature_dim)
            img_features = img_features.reshape(B, -1)
        else:
            img_features = torch.zeros(B, 0, device=lowdim.device, dtype=lowdim.dtype)

        # Flatten low-dim: (B, To, D) -> (B, To*D)
        lowdim_flat = lowdim.reshape(B, -1)

        # Concat all features
        return torch.cat([img_features, lowdim_flat], dim=-1)

    def get_output_dim(self, obs_horizon: int, lowdim_dim: int, num_cameras: int = 1) -> int:
        """Calculate the output dimension of the encoder."""
        return obs_horizon * num_cameras * self.feature_dim + obs_horizon * lowdim_dim
