"""CLI entry points for the diffusion manipulation pipeline."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion Policy for Robotic Manipulation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command
    dl = subparsers.add_parser("download", help="Download robomimic datasets")
    dl.add_argument("--task", choices=["lift", "can", "square"], default="lift")
    dl.add_argument("--dataset-dir", type=Path, default=Path("data"))
    dl.add_argument("--dataset-type", default="ph_image")
    dl.add_argument("--force", action="store_true")

    # Train command
    tr = subparsers.add_parser("train", help="Train diffusion policy")
    tr.add_argument("--task", choices=["lift", "can", "square"], default="lift")
    tr.add_argument("--dataset-dir", type=Path, default=Path("data"))
    tr.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--max-episodes", type=int, default=None)
    tr.add_argument("--device", type=str, default=None)

    # Evaluate command
    ev = subparsers.add_parser("evaluate", help="Evaluate trained policy")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--task", choices=["lift", "can", "square"], default="lift")
    ev.add_argument("--num-episodes", type=int, default=50)
    ev.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ev.add_argument("--save-videos", action="store_true")
    ev.add_argument("--video-dir", type=Path, default=Path("results/videos"))
    ev.add_argument("--device", type=str, default=None)

    # Visualize command
    viz = subparsers.add_parser("visualize", help="Visualize dataset demos")
    viz.add_argument("--dataset-path", type=Path, required=True)
    viz.add_argument("--output-dir", type=Path, default=Path("results/visualizations"))
    viz.add_argument("--num-episodes", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "download":
        _run_download(args)
    elif args.command == "train":
        _run_train(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "visualize":
        _run_visualize(args)


def _run_download(args: argparse.Namespace) -> None:
    from diffusion_manipulation.data.download import download_dataset, verify_dataset

    path = download_dataset(
        task_name=args.task,
        dataset_dir=args.dataset_dir,
        dataset_type=args.dataset_type,
        force=args.force,
    )
    if verify_dataset(path):
        print(f"Dataset verified: {path}")
    else:
        print(f"Warning: Dataset verification failed: {path}")


def _run_train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from diffusion_manipulation.config import DataConfig, PolicyConfig, TrainConfig
    from diffusion_manipulation.data.dataset import DiffusionDataset
    from diffusion_manipulation.data.replay_buffer import load_replay_buffer
    from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy
    from diffusion_manipulation.training.trainer import Trainer

    # Set seed
    torch.manual_seed(args.seed)

    # Configs
    data_cfg = DataConfig(task_name=args.task, dataset_dir=args.dataset_dir)
    policy_cfg = PolicyConfig()
    train_cfg = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on: {device}")

    # Load data
    dataset_path = data_cfg.dataset_dir / data_cfg.task_name / "ph_image.hdf5"
    print(f"Loading dataset: {dataset_path}")
    replay_buffer = load_replay_buffer(
        dataset_path,
        camera_names=data_cfg.camera_names,
        max_episodes=args.max_episodes,
    )
    print(f"Loaded {replay_buffer.num_episodes} episodes, {replay_buffer.num_steps} steps")

    dataset = DiffusionDataset(
        replay_buffer,
        obs_horizon=data_cfg.obs_horizon,
        pred_horizon=data_cfg.pred_horizon,
        camera_names=data_cfg.camera_names,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    # Create policy
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

    # Train
    trainer = Trainer(
        policy=policy,
        dataloader=dataloader,
        normalizer=dataset.normalizer,
        config=train_cfg,
        device=device,
    )

    print(f"Starting training for {train_cfg.num_epochs} epochs...")
    history = trainer.train()
    print(f"Training complete. Final loss: {history['loss'][-1]:.6f}")


def _run_evaluate(args: argparse.Namespace) -> None:
    import torch

    from diffusion_manipulation.config import DataConfig, PolicyConfig
    from diffusion_manipulation.data.normalizer import LinearNormalizer
    from diffusion_manipulation.env.robosuite_env import RobosuiteEnv
    from diffusion_manipulation.env.video_recorder import VideoRecorder
    from diffusion_manipulation.evaluation.evaluator import evaluate_multi_seed
    from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(task_name=args.task)
    policy_cfg = PolicyConfig()

    # Load policy
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

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    policy.set_normalizer(normalizer)
    policy.to(device)

    # Evaluate
    task_name_map = {"lift": "Lift", "can": "PickPlaceCan", "square": "NutAssemblySquare"}

    def env_factory(seed: int) -> RobosuiteEnv:
        return RobosuiteEnv(
            task_name=task_name_map[args.task],
            camera_names=data_cfg.camera_names,
            seed=seed,
        )

    video_recorder = VideoRecorder() if args.save_videos else None

    result = evaluate_multi_seed(
        policy=policy,
        env_factory=env_factory,
        seeds=tuple(args.seeds),
        num_episodes=args.num_episodes,
        obs_horizon=data_cfg.obs_horizon,
        action_horizon=data_cfg.action_horizon,
        device=device,
        video_recorder=video_recorder,
    )

    print(f"Success Rate: {result.mean_success_rate:.1%} ± {result.std_success_rate:.1%}")
    for r in result.per_seed_results:
        print(f"  Seed {r.seed}: {r.success_rate:.1%} ({r.num_successes}/{r.num_episodes})")


def _run_visualize(args: argparse.Namespace) -> None:
    from diffusion_manipulation.data.replay_buffer import load_replay_buffer
    from diffusion_manipulation.data.visualize import (
        plot_action_distributions,
        plot_episode_actions,
        render_demo_gif,
    )

    print(f"Loading dataset: {args.dataset_path}")
    replay_buffer = load_replay_buffer(args.dataset_path, camera_names=("agentview",))
    print(f"Loaded {replay_buffer.num_episodes} episodes")

    output_dir = Path(args.output_dir)

    # Render episode GIFs
    for ep_idx in range(min(args.num_episodes, replay_buffer.num_episodes)):
        if "agentview" in replay_buffer.images:
            gif_path = render_demo_gif(
                replay_buffer, ep_idx, "agentview",
                output_dir / f"demo_{ep_idx}.gif",
            )
            print(f"Saved demo GIF: {gif_path}")

        plot_path = plot_episode_actions(
            replay_buffer, ep_idx,
            output_dir / f"actions_ep_{ep_idx}.png",
        )
        print(f"Saved action plot: {plot_path}")

    # Plot overall action distributions
    dist_path = plot_action_distributions(
        replay_buffer, output_dir / "action_distributions.png"
    )
    print(f"Saved action distributions: {dist_path}")


if __name__ == "__main__":
    main()
