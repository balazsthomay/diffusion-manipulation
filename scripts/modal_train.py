"""Train diffusion policy on Modal with GPU."""

import modal

app = modal.App("diffusion-manipulation")

# Persistent volume for datasets and checkpoints
vol = modal.Volume.from_name("diffusion-data", create_if_missing=True)
VOL_PATH = "/data"

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.10.0",
        "torchvision>=0.25.0",
        "diffusers>=0.37.0",
        "einops>=0.8.2",
        "h5py>=3.16.0",
        "imageio>=2.37.3",
        "imageio-ffmpeg>=0.6.0",
        "matplotlib>=3.10.8",
        "tqdm>=4.67.3",
        "wandb>=0.25.0",
        "zarr>=3.1.5",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("src/diffusion_manipulation", remote_path="/root/diffusion_manipulation")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_PATH: vol},
    timeout=4 * 3600,  # 4 hours max
)
def train(
    task: str = "lift",
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    max_episodes: int | None = None,
) -> dict:
    """Download dataset (if needed) and train diffusion policy on GPU."""
    from pathlib import Path

    import torch

    from diffusion_manipulation.config import DataConfig, PolicyConfig, TrainConfig
    from diffusion_manipulation.data.dataset import DiffusionDataset
    from diffusion_manipulation.data.download import download_dataset, verify_dataset
    from diffusion_manipulation.data.replay_buffer import load_replay_buffer
    from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy
    from diffusion_manipulation.training.trainer import Trainer

    data_dir = Path(VOL_PATH) / "datasets"
    if max_episodes is not None:
        checkpoint_dir = Path(VOL_PATH) / "checkpoints" / f"{task}_{max_episodes}demos"
    else:
        checkpoint_dir = Path(VOL_PATH) / "checkpoints" / task

    # --- Download dataset to persistent volume if needed ---
    dataset_path = data_dir / task / "ph_image.hdf5"
    if not dataset_path.exists() or not verify_dataset(dataset_path):
        print(f"Downloading {task} dataset to volume...")
        download_dataset(task_name=task, dataset_dir=data_dir, dataset_type="ph_image")
        vol.commit()
        print("Dataset downloaded and committed to volume.")
    else:
        print(f"Dataset already on volume: {dataset_path}")

    # --- Setup ---
    torch.manual_seed(seed)
    device = torch.device("cuda")
    print(f"Training on: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_cfg = DataConfig(task_name=task, dataset_dir=data_dir)
    policy_cfg = PolicyConfig()
    train_cfg = TrainConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        use_wandb=False,
        checkpoint_dir=checkpoint_dir,
    )

    # --- Load data ---
    print(f"Loading dataset: {dataset_path}")
    replay_buffer = load_replay_buffer(
        dataset_path,
        camera_names=data_cfg.camera_names,
        max_episodes=max_episodes,
    )
    print(f"Loaded {replay_buffer.num_episodes} episodes, {replay_buffer.num_steps} steps")

    dataset = DiffusionDataset(
        replay_buffer,
        obs_horizon=data_cfg.obs_horizon,
        pred_horizon=data_cfg.pred_horizon,
        camera_names=data_cfg.camera_names,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # --- Create policy ---
    policy = DiffusionUnetPolicy(
        action_dim=data_cfg.action_dim,
        obs_horizon=data_cfg.obs_horizon,
        pred_horizon=data_cfg.pred_horizon,
        action_horizon=data_cfg.action_horizon,
        lowdim_obs_dim=data_cfg.lowdim_obs_dim,
        n_diffusion_steps_train=policy_cfg.n_diffusion_steps_train,
        n_diffusion_steps_infer=policy_cfg.n_diffusion_steps_infer,
        down_dims=policy_cfg.unet.down_dims,
        camera_names=data_cfg.camera_names,
    )
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {param_count:,}")

    # --- Train ---
    trainer = Trainer(
        policy=policy,
        dataloader=dataloader,
        normalizer=dataset.normalizer,
        config=train_cfg,
        device=device,
    )

    print(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    history = trainer.train()

    # Commit checkpoints to volume
    vol.commit()
    print(f"Checkpoints saved to volume at {checkpoint_dir}")

    return {
        "final_loss": history["loss"][-1],
        "min_loss": min(history["loss"]),
        "task": task,
        "epochs": epochs,
        "device": torch.cuda.get_device_name(0),
    }


@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    timeout=60,
)
def list_checkpoints(task: str = "lift", max_episodes: int | None = None) -> list[str]:
    """List available checkpoints on the volume."""
    from pathlib import Path

    subdir = f"{task}_{max_episodes}demos" if max_episodes else task
    checkpoint_dir = Path(VOL_PATH) / "checkpoints" / subdir
    if not checkpoint_dir.exists():
        return []
    return sorted(str(p.name) for p in checkpoint_dir.glob("*.pt"))


@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    timeout=300,
)
def download_checkpoint(
    task: str = "lift",
    filename: str = "checkpoint_final.pt",
    max_episodes: int | None = None,
) -> bytes:
    """Download a checkpoint from the volume."""
    from pathlib import Path

    subdir = f"{task}_{max_episodes}demos" if max_episodes else task
    path = Path(VOL_PATH) / "checkpoints" / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.read_bytes()


@app.local_entrypoint()
def main(
    task: str = "lift",
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    max_episodes: int | None = None,
    ablation: bool = False,
) -> None:
    if ablation:
        # Run n_demos ablation: train 25 and 100 demos in parallel
        demo_counts = [25, 100]
        print(f"Starting n_demos ablation: {demo_counts} demos, {epochs} epochs each")

        handles = []
        for n in demo_counts:
            h = train.spawn(
                task=task,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                max_episodes=n,
            )
            handles.append((n, h))
            print(f"  Spawned training: {n} demos")

        for n, h in handles:
            result = h.get()
            print(f"\n[{n} demos] Training complete!")
            print(f"  Device: {result['device']}")
            print(f"  Final loss: {result['final_loss']:.6f}")
            print(f"  Min loss: {result['min_loss']:.6f}")
    else:
        result = train.remote(
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            max_episodes=max_episodes,
        )
        print(f"\nTraining complete!")
        print(f"  Task: {result['task']}")
        print(f"  Device: {result['device']}")
        print(f"  Final loss: {result['final_loss']:.6f}")
        print(f"  Min loss: {result['min_loss']:.6f}")
