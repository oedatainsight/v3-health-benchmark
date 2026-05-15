from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DIR = REPO_ROOT / "artifacts" / "release"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"


AGENT_ORDER = [
    "baseline",
    "workflow",
    "causal_light",
    "stability_filtered",
    "structural_causal",
]

AGENT_COLORS = {
    "baseline": "#000000",
    "workflow": "#0072B2",
    "causal_light": "#009E73",
    "stability_filtered": "#D55E00",
    "structural_causal": "#CC79A7",
}

AGENT_MARKERS = {
    "baseline": "o",
    "workflow": "s",
    "causal_light": "^",
    "stability_filtered": "D",
    "structural_causal": "P",
}


def _configure_matplotlib(output_dir: Path):
    cache_root = Path(tempfile.gettempdir()) / "v3-health-matplotlib-cache"
    mpl_config = cache_root / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    font_cache = cache_root / "cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _scenario_label(name: str) -> str:
    labels = {
        "treatment_allocation_confounding": "S1 Treatment Allocation",
        "missing_data_bias": "S2 MNAR Measurement",
        "historical_bias_feedback": "S3 Historical Feedback",
    }
    return labels.get(name, name.replace("_", " ").title())


def _agent_label(name: str, labels: dict[str, str]) -> str:
    return labels.get(name, name.replace("_", " ").title())


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _size_from_gap(gap: float, max_gap: float) -> float:
    if max_gap <= 0:
        return 110.0
    return 80.0 + 620.0 * (gap / max_gap)


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def figure1_regime_dependence(release_dir: Path, output_dir: Path, plt) -> None:
    rows = _read_csv(release_dir / "tradeoff_summary.csv")
    payload = _read_json(release_dir / "results_summary.json")
    scenarios = payload["meta"]["scenarios"]
    labels = payload["meta"].get("agent_labels", {})
    agents = [agent for agent in AGENT_ORDER if agent in payload["meta"]["agents"]]
    max_gap = max(_float(row, "fairness_gap_mean") for row in rows)
    by_scenario = {scenario: [row for row in rows if row["scenario"] == scenario] for scenario in scenarios}

    fig, axes = plt.subplots(1, len(scenarios), figsize=(13.5, 4.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        for agent in agents:
            row = next((item for item in by_scenario[scenario] if item["agent"] == agent), None)
            if row is None:
                continue
            x = _float(row, "robustness_drop_mean")
            y = _float(row, "success_rate_mean")
            xerr = [
                [max(0.0, x - _float(row, "robustness_drop_ci_lower"))],
                [max(0.0, _float(row, "robustness_drop_ci_upper") - x)],
            ]
            yerr = [
                [max(0.0, y - _float(row, "success_rate_ci_lower"))],
                [max(0.0, _float(row, "success_rate_ci_upper") - y)],
            ]
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor="#666666",
                elinewidth=0.8,
                alpha=0.65,
                zorder=1,
            )
            ax.scatter(
                x,
                y,
                s=_size_from_gap(_float(row, "fairness_gap_mean"), max_gap),
                marker=AGENT_MARKERS[agent],
                facecolor=AGENT_COLORS[agent],
                edgecolor="white",
                linewidth=0.9,
                alpha=0.92,
                zorder=2,
            )
        ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle=":")
        ax.grid(True, linewidth=0.5, color="#dddddd")
        ax.set_title(_scenario_label(scenario))
        ax.set_xlabel("Success-rate drop\nnormal to adversarial")

    axes[0].set_ylabel("Overall success rate")

    agent_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=AGENT_MARKERS[agent],
            color="none",
            markerfacecolor=AGENT_COLORS[agent],
            markeredgecolor="white",
            markersize=8,
            label=_agent_label(agent, labels),
        )
        for agent in agents
    ]
    size_values = sorted({round(max_gap * value, 3) for value in (0.33, 0.66, 1.0)})
    size_handles = [
        plt.scatter(
            [],
            [],
            s=_size_from_gap(value, max_gap),
            marker="o",
            facecolor="#bbbbbb",
            edgecolor="#555555",
            label=f"{value:.3f}",
        )
        for value in size_values
    ]
    fig.legend(
        handles=agent_handles,
        loc="lower center",
        ncol=len(agent_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.legend(
        handles=size_handles,
        title="Bubble size: SES fairness gap",
        loc="center right",
        frameon=False,
        bbox_to_anchor=(1.03, 0.5),
    )
    fig.suptitle("Regime Dependence With Accessible Agent and Fairness Encoding", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 0.92, 1.0))
    _save(fig, output_dir, "figure1_regime_dependence_accessible")
    plt.close(fig)


def _seed_phase_values(payload: dict, scenario: str, agent: str) -> list[tuple[int, float, float]]:
    per_seed = payload["scenario_results"][scenario][agent]["seed_level"]["per_seed"]
    values = []
    for item in per_seed:
        normal = item["by_phase"]["normal"]["success_rate"]
        adversarial = item["by_phase"]["adversarial"]["success_rate"]
        values.append((int(item["seed"]), float(normal), float(adversarial)))
    return sorted(values)


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    import math
    from scipy import stats

    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    half = float(stats.t.ppf(0.975, df=n - 1)) * std / math.sqrt(n)
    return mean, mean - half, mean + half


def figure2_paired_seed_slopechart(release_dir: Path, output_dir: Path, plt) -> None:
    payload = _read_json(release_dir / "results_summary.json")
    scenarios = payload["meta"]["scenarios"]
    labels = payload["meta"].get("agent_labels", {})
    agents = [agent for agent in AGENT_ORDER if agent in payload["meta"]["agents"]]

    fig, axes = plt.subplots(
        len(scenarios),
        len(agents),
        figsize=(15.5, 9.2),
        sharex=True,
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]

    for row_idx, scenario in enumerate(scenarios):
        for col_idx, agent in enumerate(agents):
            ax = axes[row_idx][col_idx]
            values = _seed_phase_values(payload, scenario, agent)
            color = AGENT_COLORS[agent]
            drops = []
            normal_values = []
            adversarial_values = []
            for seed, normal, adversarial in values:
                drops.append(normal - adversarial)
                normal_values.append(normal)
                adversarial_values.append(adversarial)
                ax.plot(
                    [0, 1],
                    [normal, adversarial],
                    color="#888888",
                    linewidth=0.7,
                    alpha=0.35,
                    zorder=1,
                )
                ax.scatter(
                    [0, 1],
                    [normal, adversarial],
                    s=10,
                    color="#888888",
                    alpha=0.35,
                    zorder=2,
                )

            n_mean, n_lo, n_hi = _mean_ci(normal_values)
            a_mean, a_lo, a_hi = _mean_ci(adversarial_values)
            d_mean, d_lo, d_hi = _mean_ci(drops)
            ax.plot([0, 1], [n_mean, a_mean], color=color, linewidth=2.2, zorder=3)
            ax.errorbar(
                [0, 1],
                [n_mean, a_mean],
                yerr=[[n_mean - n_lo, a_mean - a_lo], [n_hi - n_mean, a_hi - a_mean]],
                fmt=AGENT_MARKERS[agent],
                markersize=5.5,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )
            ax.text(
                0.5,
                0.04,
                f"drop {d_mean:.3f}\n95% CI [{d_lo:.3f}, {d_hi:.3f}]",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=7,
                color="#333333",
            )
            ax.grid(True, axis="y", linewidth=0.5, color="#dddddd")
            ax.set_xticks([0, 1], ["normal", "adversarial"])
            if row_idx == 0:
                ax.set_title(_agent_label(agent, labels))
            if col_idx == 0:
                ax.set_ylabel(f"{_scenario_label(scenario)}\nseed success rate")
            ax.set_ylim(0.35, 0.95)

    fig.suptitle("Paired Seed Slopechart: Normal to Adversarial Success", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, output_dir, "figure2_paired_seed_slopechart")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = _configure_matplotlib(args.output_dir)
    figure1_regime_dependence(args.release_dir, args.output_dir, plt)
    figure2_paired_seed_slopechart(args.release_dir, args.output_dir, plt)
    print(f"Wrote Priority A figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
