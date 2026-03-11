#!/usr/bin/env python3
"""Run failure analysis and DDIM inference steps ablation on trained Lift policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_manipulation.config import DataConfig, PolicyConfig
from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.env.robosuite_env import RobosuiteEnv
from diffusion_manipulation.evaluation.analysis import (
    AblationResult,
    FailureType,
    analyze_failures,
    generate_results_table,
    plot_ablation_results,
)
from diffusion_manipulation.evaluation.evaluator import evaluate_policy
from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy
from diffusion_manipulation.training.ema import EMAModel


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

    # Apply EMA weights
    if "ema_state_dict" in checkpoint:
        ema = EMAModel(policy)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.apply_shadow(policy)

    # Load normalizer
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    policy.set_normalizer(normalizer)

    policy.to(device)
    policy.eval()
    return policy


def create_env(seed: int) -> RobosuiteEnv:
    return RobosuiteEnv(
        task_name="Lift",
        camera_names=("agentview",),
        seed=seed,
    )


def run_failure_analysis(
    policy: DiffusionUnetPolicy,
    num_episodes: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Run evaluation and analyze failure modes."""
    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS — {num_episodes} episodes, seed={seed}")
    print(f"{'='*60}")

    env = create_env(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    result = evaluate_policy(
        policy=policy,
        env=env,
        num_episodes=num_episodes,
        obs_horizon=2,
        action_horizon=8,
        max_episode_steps=400,
        camera_names=("agentview",),
        device=device,
    )
    env.close()

    print(f"\nSuccess rate: {result.success_rate:.1%} ({result.num_successes}/{result.num_episodes})")

    # Analyze failures
    analysis = analyze_failures(result, max_steps=400)
    print(f"Total failures: {analysis.total_failures}")

    if analysis.total_failures > 0:
        print("\nFailure breakdown:")
        for ft in FailureType:
            count = analysis.failure_counts.get(ft, 0)
            if count > 0:
                eps = analysis.failure_episodes[ft]
                print(f"  {ft.name}: {count} episodes {eps}")

        # Per-failure episode details
        print("\nFailed episode details:")
        for ep_idx, (reward, length, success) in enumerate(
            zip(result.episode_rewards, result.episode_lengths, result.episode_successes)
        ):
            if not success:
                print(f"  Episode {ep_idx}: reward={reward:.3f}, length={length}, "
                      f"avg_reward={reward/max(length,1):.4f}")
    else:
        print("No failures — all episodes successful!")

    # Summary stats
    rewards = np.array(result.episode_rewards)
    lengths = np.array(result.episode_lengths)
    print(f"\nReward stats: mean={rewards.mean():.2f}, std={rewards.std():.2f}, "
          f"min={rewards.min():.2f}, max={rewards.max():.2f}")
    print(f"Length stats: mean={lengths.mean():.1f}, std={lengths.std():.1f}, "
          f"min={lengths.min()}, max={lengths.max()}")

    return {
        "success_rate": result.success_rate,
        "num_episodes": result.num_episodes,
        "num_successes": result.num_successes,
        "total_failures": analysis.total_failures,
        "failure_counts": {ft.name: c for ft, c in analysis.failure_counts.items() if c > 0},
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
    }


def run_ddim_steps_ablation(
    policy: DiffusionUnetPolicy,
    num_episodes: int,
    seed: int,
    device: torch.device,
    inference_steps: list[int] = [4, 8, 16, 32],
) -> list[AblationResult]:
    """Ablation on number of DDIM inference steps."""
    print(f"\n{'='*60}")
    print(f"ABLATION: DDIM Inference Steps — {num_episodes} eps/config, seed={seed}")
    print(f"{'='*60}")

    results = []
    for n_steps in inference_steps:
        print(f"\n--- {n_steps} DDIM steps ---")
        policy.set_inference_steps(n_steps)

        env = create_env(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        eval_result = evaluate_policy(
            policy=policy,
            env=env,
            num_episodes=num_episodes,
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
            name=f"{n_steps} steps",
            variable="ddim_inference_steps",
            value=n_steps,
            eval_result=eval_result,
        ))

    # Reset to default 16 steps
    policy.set_inference_steps(16)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run failure analysis and ablation")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/lift/checkpoint_epoch_250.pt"))
    parser.add_argument("--failure-episodes", type=int, default=50)
    parser.add_argument("--ablation-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-failure", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, device)
    print("Policy loaded with EMA weights")

    all_results = {}

    # 1. Failure analysis
    if not args.skip_failure:
        failure_results = run_failure_analysis(
            policy, args.failure_episodes, args.seed, device
        )
        all_results["failure_analysis"] = failure_results

    # 2. DDIM steps ablation
    if not args.skip_ablation:
        ablation_results = run_ddim_steps_ablation(
            policy, args.ablation_episodes, args.seed, device
        )
        all_results["ablation"] = {
            "variable": "ddim_inference_steps",
            "configs": [
                {
                    "name": r.name,
                    "value": r.value,
                    "success_rate": r.eval_result.success_rate,
                }
                for r in ablation_results
            ],
        }

        # Generate table and plot
        table = generate_results_table(ablation_results)
        print(f"\n{'='*60}")
        print("ABLATION RESULTS TABLE")
        print(f"{'='*60}")
        print(table)

        plot_path = plot_ablation_results(
            ablation_results,
            output_dir / "ddim_steps_ablation.png",
            title="DDIM Inference Steps Ablation (Lift Task)",
        )
        print(f"\nPlot saved: {plot_path}")

    # Save all results as JSON
    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
