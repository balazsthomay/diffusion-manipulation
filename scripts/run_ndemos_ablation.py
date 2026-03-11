#!/usr/bin/env python3
"""Download n_demos ablation checkpoints from Modal and evaluate locally."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from diffusion_manipulation.config import DataConfig, PolicyConfig
from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.env.robosuite_env import RobosuiteEnv
from diffusion_manipulation.evaluation.analysis import (
    AblationResult,
    generate_results_table,
    plot_ablation_results,
)
from diffusion_manipulation.evaluation.evaluator import evaluate_policy
from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy
from diffusion_manipulation.training.ema import EMAModel


DEMO_CONFIGS = [
    {"n_demos": 25, "checkpoint_name": "checkpoint_epoch_250.pt"},
    {"n_demos": 100, "checkpoint_name": "checkpoint_epoch_250.pt"},
    {"n_demos": 200, "checkpoint_name": "checkpoint_epoch_250.pt"},
]


def download_checkpoint_from_modal(
    task: str, max_episodes: int | None, filename: str, local_path: Path
) -> None:
    """Download checkpoint from Modal volume."""
    if local_path.exists():
        print(f"  Already downloaded: {local_path}")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Use modal volume get to download
    subdir = f"{task}_{max_episodes}demos" if max_episodes else task
    remote_path = f"checkpoints/{subdir}/{filename}"
    print(f"  Downloading {remote_path} from Modal volume...")
    subprocess.run(
        ["modal", "volume", "get", "diffusion-data", remote_path, str(local_path)],
        check=True,
    )
    print(f"  Saved: {local_path}")


def load_policy(checkpoint_path: Path, device: torch.device) -> DiffusionUnetPolicy:
    """Load policy from checkpoint with EMA weights."""
    data_cfg = DataConfig(task_name="lift")
    policy_cfg = PolicyConfig()

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
        pretrained_vision=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])

    if "ema_state_dict" in checkpoint:
        ema = EMAModel(policy)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.apply_shadow(policy)

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    policy.set_normalizer(normalizer)

    policy.to(device)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run n_demos ablation evaluation")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading checkpoints (assume already local)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoints
    if not args.skip_download:
        print("\n--- Downloading checkpoints ---")
        for cfg in DEMO_CONFIGS:
            n = cfg["n_demos"]
            fname = cfg["checkpoint_name"]
            if n == 200:
                local = Path(f"checkpoints/lift/{fname}")
            else:
                local = Path(f"checkpoints/lift_{n}demos/{fname}")
            download_checkpoint_from_modal(
                task="lift",
                max_episodes=n if n != 200 else None,
                filename=fname,
                local_path=local,
            )

    # Evaluate each checkpoint
    print(f"\n{'='*60}")
    print(f"N_DEMOS ABLATION — {args.episodes} episodes/config, seed={args.seed}")
    print(f"{'='*60}")

    results = []
    for cfg in DEMO_CONFIGS:
        n = cfg["n_demos"]
        fname = cfg["checkpoint_name"]
        if n == 200:
            ckpt_path = Path(f"checkpoints/lift/{fname}")
        else:
            ckpt_path = Path(f"checkpoints/lift_{n}demos/{fname}")

        print(f"\n--- {n} demos ---")
        print(f"Checkpoint: {ckpt_path}")

        policy = load_policy(ckpt_path, device)

        env = RobosuiteEnv(task_name="Lift", camera_names=("agentview",), seed=args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        eval_result = evaluate_policy(
            policy=policy,
            env=env,
            num_episodes=args.episodes,
            obs_horizon=2,
            action_horizon=8,
            max_episode_steps=400,
            camera_names=("agentview",),
            device=device,
        )
        env.close()

        print(f"Success: {eval_result.success_rate:.1%} "
              f"({eval_result.num_successes}/{eval_result.num_episodes})")

        results.append(AblationResult(
            name=f"{n} demos",
            variable="n_demos",
            value=n,
            eval_result=eval_result,
        ))

        # Free GPU memory between configs
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Generate table
    table = generate_results_table(results)
    print(f"\n{'='*60}")
    print("N_DEMOS ABLATION RESULTS")
    print(f"{'='*60}")
    print(table)

    # Plot
    plot_path = plot_ablation_results(
        results,
        output_dir / "ndemos_ablation.png",
        title="Number of Demonstrations Ablation (Lift Task)",
    )
    print(f"\nPlot saved: {plot_path}")

    # Save results
    ablation_data = {
        "variable": "n_demos",
        "episodes_per_config": args.episodes,
        "seed": args.seed,
        "configs": [
            {
                "name": r.name,
                "value": r.value,
                "success_rate": r.eval_result.success_rate,
            }
            for r in results
        ],
    }

    # Merge with existing results
    results_path = output_dir / "analysis_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["ndemos_ablation"] = ablation_data

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
