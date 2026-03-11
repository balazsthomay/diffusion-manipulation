"""Ablation framework and failure analysis for diffusion policy evaluation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffusion_manipulation.evaluation.evaluator import EvalResult, MultiSeedResult


class FailureType(Enum):
    """Categories of manipulation failures."""

    GRASP_FAILURE = auto()  # Failed to grasp the object
    TRANSPORT_FAILURE = auto()  # Grasped but dropped during transport
    PLACEMENT_FAILURE = auto()  # Transported but failed placement
    TIMEOUT = auto()  # Ran out of steps
    OTHER = auto()


@dataclass
class FailureAnalysis:
    """Analysis of failure modes from evaluation."""

    total_episodes: int
    total_failures: int
    failure_counts: dict[FailureType, int] = field(default_factory=dict)
    failure_episodes: dict[FailureType, list[int]] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return self.total_failures / max(self.total_episodes, 1)

    def failure_distribution(self) -> dict[str, float]:
        """Get failure type distribution as percentages."""
        if self.total_failures == 0:
            return {}
        return {
            ft.name: count / self.total_failures
            for ft, count in self.failure_counts.items()
            if count > 0
        }


def categorize_failure(
    episode_length: int,
    max_steps: int,
    episode_reward: float,
    reward_thresholds: tuple[float, float, float] = (0.1, 0.5, 0.8),
) -> FailureType:
    """Categorize a failed episode based on reward progression.

    Uses reward thresholds to infer failure stage:
    - Low reward → grasp failure
    - Medium reward → transport failure
    - High reward (but not success) → placement failure
    - Max steps reached → timeout

    Args:
        episode_length: Number of steps in the episode.
        max_steps: Maximum allowed steps.
        episode_reward: Total episode reward.
        reward_thresholds: (grasp, transport, placement) thresholds.

    Returns:
        FailureType classification.
    """
    if episode_length >= max_steps:
        return FailureType.TIMEOUT

    grasp_thresh, transport_thresh, placement_thresh = reward_thresholds

    if episode_reward < grasp_thresh:
        return FailureType.GRASP_FAILURE
    elif episode_reward < transport_thresh:
        return FailureType.TRANSPORT_FAILURE
    elif episode_reward < placement_thresh:
        return FailureType.PLACEMENT_FAILURE
    else:
        return FailureType.OTHER


def analyze_failures(
    eval_result: EvalResult,
    max_steps: int = 400,
) -> FailureAnalysis:
    """Analyze failure modes from evaluation results.

    Args:
        eval_result: Results from evaluate_policy.
        max_steps: Maximum episode steps (for timeout detection).

    Returns:
        FailureAnalysis with categorized failures.
    """
    failure_counts: dict[FailureType, int] = {ft: 0 for ft in FailureType}
    failure_episodes: dict[FailureType, list[int]] = {ft: [] for ft in FailureType}
    total_failures = eval_result.num_episodes - eval_result.num_successes

    for ep_idx, (reward, length, success) in enumerate(
        zip(
            eval_result.episode_rewards,
            eval_result.episode_lengths,
            eval_result.episode_successes,
        )
    ):
        if success:
            continue

        failure_type = categorize_failure(length, max_steps, reward)
        failure_counts[failure_type] += 1
        failure_episodes[failure_type].append(ep_idx)

    return FailureAnalysis(
        total_episodes=eval_result.num_episodes,
        total_failures=total_failures,
        failure_counts=failure_counts,
        failure_episodes=failure_episodes,
    )


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    name: str
    variable: str
    value: object
    eval_result: MultiSeedResult | EvalResult


def plot_ablation_results(
    results: list[AblationResult],
    output_path: str | Path,
    title: str = "Ablation Study",
) -> Path:
    """Plot ablation study results as a bar chart.

    Args:
        results: List of ablation results.
        output_path: Where to save the plot.
        title: Plot title.

    Returns:
        Path to saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results]
    means = []
    stds = []

    for r in results:
        if isinstance(r.eval_result, MultiSeedResult):
            means.append(r.eval_result.mean_success_rate)
            stds.append(r.eval_result.std_success_rate)
        else:
            means.append(r.eval_result.success_rate)
            stds.append(0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_results_table(
    results: list[AblationResult],
) -> str:
    """Generate a markdown results table.

    Args:
        results: List of ablation results.

    Returns:
        Markdown-formatted table string.
    """
    lines = ["| Configuration | Success Rate | Std |", "|---|---|---|"]

    for r in results:
        if isinstance(r.eval_result, MultiSeedResult):
            rate = f"{r.eval_result.mean_success_rate:.1%}"
            std = f"±{r.eval_result.std_success_rate:.1%}"
        else:
            rate = f"{r.eval_result.success_rate:.1%}"
            std = "N/A"
        lines.append(f"| {r.name} | {rate} | {std} |")

    return "\n".join(lines)
