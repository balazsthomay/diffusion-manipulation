"""Frozen dataclass configurations for the diffusion policy pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_dir: Path = Path("data")
    task_name: str = "lift"  # lift, can, square
    dataset_type: str = "ph"  # ph (proficient-human), mh (multi-human)
    obs_horizon: int = 2  # number of observation steps (To)
    pred_horizon: int = 16  # number of predicted action steps (Tp)
    action_horizon: int = 8  # number of executed action steps (Ta)
    action_dim: int = 7  # OSC_POSE: 3 pos + 3 rot + 1 gripper
    lowdim_obs_dim: int = 9  # eef_pos(3) + eef_quat(4) + gripper_qpos(2)
    image_shape: tuple[int, int, int] = (3, 84, 84)
    camera_names: tuple[str, ...] = ("agentview",)
    max_train_episodes: int | None = None  # None = use all


@dataclass(frozen=True)
class UnetConfig:
    """Configuration for the ConditionalUnet1D architecture."""

    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 256
    cond_predict_scale: bool = True


@dataclass(frozen=True)
class VisionEncoderConfig:
    """Configuration for the ResNet18-based vision encoder."""

    pretrained: bool = True
    feature_dim: int = 512  # ResNet18 output dim
    crop_shape: tuple[int, int] = (76, 76)  # random crop target
    imagenet_norm: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for the diffusion policy."""

    n_diffusion_steps_train: int = 100
    n_diffusion_steps_infer: int = 16
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    unet: UnetConfig = field(default_factory=UnetConfig)
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training."""

    num_epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    ema_decay: float = 0.995
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 50
    use_wandb: bool = False
    wandb_project: str = "diffusion-manipulation"
    seed: int = 42
    num_workers: int = 4
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for evaluation."""

    num_episodes: int = 50
    seeds: tuple[int, ...] = (42, 123, 456)
    max_episode_steps: int = 400
    render: bool = False
    save_video: bool = True
    video_dir: Path = Path("results/videos")
    results_dir: Path = Path("results")
