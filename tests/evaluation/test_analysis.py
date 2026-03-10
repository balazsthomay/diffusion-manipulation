"""Tests for failure analysis and ablation framework."""

from pathlib import Path

from diffusion_manipulation.evaluation.analysis import (
    AblationResult,
    FailureAnalysis,
    FailureType,
    analyze_failures,
    categorize_failure,
    generate_results_table,
    plot_ablation_results,
)
from diffusion_manipulation.evaluation.evaluator import EvalResult, MultiSeedResult


class TestCategorizeFailure:
    def test_timeout(self) -> None:
        assert categorize_failure(400, 400, 0.3) == FailureType.TIMEOUT

    def test_grasp_failure(self) -> None:
        assert categorize_failure(50, 400, 0.05) == FailureType.GRASP_FAILURE

    def test_transport_failure(self) -> None:
        assert categorize_failure(100, 400, 0.3) == FailureType.TRANSPORT_FAILURE

    def test_placement_failure(self) -> None:
        assert categorize_failure(200, 400, 0.6) == FailureType.PLACEMENT_FAILURE

    def test_other(self) -> None:
        assert categorize_failure(100, 400, 0.9) == FailureType.OTHER


class TestAnalyzeFailures:
    def test_all_successful(self) -> None:
        result = EvalResult(
            success_rate=1.0,
            num_episodes=10,
            num_successes=10,
            episode_rewards=[1.0] * 10,
            episode_lengths=[50] * 10,
        )
        analysis = analyze_failures(result)
        assert analysis.total_failures == 0

    def test_mixed_failures(self) -> None:
        result = EvalResult(
            success_rate=0.5,
            num_episodes=4,
            num_successes=2,
            episode_rewards=[1.0, 1.0, 0.05, 0.3],
            episode_lengths=[50, 50, 100, 200],
        )
        analysis = analyze_failures(result)
        assert analysis.total_failures == 2
        assert analysis.total_episodes == 4

    def test_failure_distribution(self) -> None:
        result = EvalResult(
            success_rate=0.0,
            num_episodes=3,
            num_successes=0,
            episode_rewards=[0.01, 0.3, 0.6],
            episode_lengths=[50, 100, 200],
        )
        analysis = analyze_failures(result)
        dist = analysis.failure_distribution()
        assert len(dist) > 0
        assert sum(dist.values()) == 1.0


class TestPlotAblation:
    def test_creates_plot(self, tmp_path: Path) -> None:
        results = [
            AblationResult(
                name="25 demos",
                variable="n_demos",
                value=25,
                eval_result=EvalResult(success_rate=0.5, num_episodes=10, num_successes=5),
            ),
            AblationResult(
                name="100 demos",
                variable="n_demos",
                value=100,
                eval_result=MultiSeedResult(
                    mean_success_rate=0.8,
                    std_success_rate=0.05,
                ),
            ),
        ]

        path = plot_ablation_results(results, tmp_path / "ablation.png")
        assert path.exists()


class TestGenerateResultsTable:
    def test_generates_markdown(self) -> None:
        results = [
            AblationResult(
                name="Config A",
                variable="test",
                value="a",
                eval_result=MultiSeedResult(
                    mean_success_rate=0.85,
                    std_success_rate=0.03,
                ),
            ),
        ]
        table = generate_results_table(results)
        assert "Config A" in table
        assert "85.0%" in table
        assert "|" in table
