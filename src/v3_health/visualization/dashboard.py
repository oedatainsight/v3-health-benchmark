"""
Visualization dashboard for v3_health benchmark results.

Loads results_summary.json and generates an interactive HTML dashboard
with multiple plots covering performance, fairness, robustness, tradeoffs,
and the strongest negative result.

Usage:
    python -m v3_health.visualization.dashboard \
        --results path/to/results_summary.json \
        --output path/to/dashboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_AGENT_ORDER = [
    "baseline",
    "workflow",
    "causal_light",
    "stability_filtered",
    "causal",
    "structural_causal",
]
PHASES = ["normal", "surface_shift", "adversarial"]
DEFAULT_AGENT_LABELS = {
    "baseline": "Baseline",
    "workflow": "Workflow",
    "stability_filtered": "Stability-Filtered",
    "causal": "Stability-Filtered (legacy key)",
    "causal_light": "Causal Light",
    "structural_causal": "Structural Causal",
}
AGENT_COLORS = {
    "baseline": "#d62728",
    "workflow": "#ff7f0e",
    "stability_filtered": "#2ca02c",
    "causal": "#2ca02c",
    "causal_light": "#17becf",
    "structural_causal": "#1f77b4",
}


def _load(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _scenario_results(payload: dict) -> dict:
    if "scenario_results" in payload:
        return payload["scenario_results"]
    return payload


def _pairwise_tests(payload: dict) -> list[dict]:
    return payload.get("pairwise_tests", [])


def _negative_result(payload: dict) -> dict | None:
    return payload.get("negative_result")


def _meta(payload: dict) -> dict:
    return payload.get("meta", {})


def _sort_key(agent: str) -> tuple[int, str]:
    if agent in DEFAULT_AGENT_ORDER:
        return (DEFAULT_AGENT_ORDER.index(agent), agent)
    return (len(DEFAULT_AGENT_ORDER), agent)


def _agent_ids(payload: dict, results: dict) -> list[str]:
    meta_agents = _meta(payload).get("agents")
    if meta_agents:
        return list(meta_agents)

    discovered = {
        agent
        for scenario_agents in results.values()
        for agent in scenario_agents.keys()
    }
    return sorted(discovered, key=_sort_key)


def _agent_labels(payload: dict, agents: list[str]) -> dict[str, str]:
    labels = dict(_meta(payload).get("agent_labels", {}))
    labels.update(DEFAULT_AGENT_LABELS)
    return {
        agent: labels.get(agent, agent.replace("_", " ").title())
        for agent in agents
    }


def _agent_label(agent: str, labels: dict[str, str]) -> str:
    return labels.get(agent, agent.replace("_", " ").title())


def _agent_color(agent: str) -> str:
    return AGENT_COLORS.get(agent, "#4c78a8")


def _primary_claim_agent(payload: dict, agents: list[str]) -> str:
    requested = _meta(payload).get("primary_claim_agent")
    if requested in agents:
        return requested
    for candidate in ("structural_causal", "stability_filtered", "causal_light", "causal"):
        if candidate in agents:
            return candidate
    return agents[0]


def _intro_text(agents: list[str]) -> str:
    if {
        "baseline",
        "workflow",
        "causal_light",
        "stability_filtered",
        "structural_causal",
    }.issubset(set(agents)):
        return (
            "Single-agent healthcare benchmark comparing a policy baseline, "
            "a workflow comparator, a causal-light structural-prior ablation, "
            "a stability-filtered heuristic, and a structural-causal agent "
            "across a continuous normal to surface-shift to adversarial "
            "trajectory under confounding, missing data, and historical SES bias."
        )
    return (
        "Single-agent healthcare benchmark comparing policy baselines, "
        "workflow and causal-scaffolding heuristics, and structural agents "
        "under confounding, missing data, and historical SES bias."
    )


def _negative_result_blurb(primary_agent: str) -> str:
    if primary_agent == "structural_causal":
        return (
            "This section is included on purpose so the benchmark shows where "
            "explicit structural modeling does not dominate simpler alternatives."
        )
    return (
        "This section is included on purpose so the benchmark shows where the "
        "focal causal claim does not dominate simpler alternatives."
    )


def _metric_stats(agent_result: dict, slice_name: str, metric: str) -> dict:
    seed_level = agent_result.get("seed_level", {})
    if slice_name == "overall":
        summary = seed_level.get("overall", {}).get(metric)
        pooled = agent_result.get("overall", {}).get(metric)
    else:
        summary = seed_level.get("by_phase", {}).get(slice_name, {}).get(metric)
        pooled = agent_result.get("by_phase", {}).get(slice_name, {}).get(metric)

    if summary and summary.get("mean") is not None:
        return {
            "mean": summary.get("mean"),
            "ci_lower": summary.get("ci_lower"),
            "ci_upper": summary.get("ci_upper"),
            "n": summary.get("n"),
        }

    if pooled is not None:
        return {
            "mean": pooled,
            "ci_lower": pooled,
            "ci_upper": pooled,
            "n": None,
        }

    return {
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
        "n": None,
    }


def _fmt_interval(stats: dict, digits: int = 3) -> str:
    mean = stats.get("mean")
    lower = stats.get("ci_lower")
    upper = stats.get("ci_upper")
    if mean is None:
        return "-"
    if lower is None or upper is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _scenario_label(name: str) -> str:
    return name.replace("_", " ")


def _phase_metric_subplots(
    results: dict,
    agents: list[str],
    labels: dict[str, str],
    metric: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    scenarios = list(results.keys())
    fig = make_subplots(
        rows=1,
        cols=len(scenarios),
        subplot_titles=[_scenario_label(s) for s in scenarios],
        shared_yaxes=True,
    )

    for col, scenario in enumerate(scenarios, start=1):
        for agent in agents:
            if agent not in results[scenario]:
                continue
            label = _agent_label(agent, labels)
            ys = []
            uppers = []
            lowers = []
            custom = []
            for phase in PHASES:
                stats = _metric_stats(results[scenario][agent], phase, metric)
                mean = stats["mean"]
                lower = stats["ci_lower"] if stats["ci_lower"] is not None else mean
                upper = stats["ci_upper"] if stats["ci_upper"] is not None else mean
                ys.append(mean)
                uppers.append(None if mean is None else max(0.0, upper - mean))
                lowers.append(None if mean is None else max(0.0, mean - lower))
                custom.append([lower, upper])

            fig.add_trace(
                go.Bar(
                    x=PHASES,
                    y=ys,
                    name=label,
                    marker_color=_agent_color(agent),
                    showlegend=(col == 1),
                    legendgroup=agent,
                    error_y=dict(type="data", array=uppers, arrayminus=lowers, visible=True),
                    customdata=custom,
                    hovertemplate=(
                        "%{x}<br>"
                        + label + ": %{y:.3f}<br>"
                        + "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=title,
        barmode="group",
        height=430,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=80, b=80),
    )
    fig.update_yaxes(title_text=yaxis_title, col=1)
    return fig


def _degradation_chart(results: dict, agents: list[str], labels: dict[str, str]) -> go.Figure:
    scenarios = list(results.keys())
    fig = go.Figure()

    for agent in agents:
        label = _agent_label(agent, labels)
        ys = []
        uppers = []
        lowers = []
        custom = []
        for scenario in scenarios:
            if agent not in results[scenario]:
                ys.append(None)
                uppers.append(None)
                lowers.append(None)
                custom.append([None, None])
                continue
            stats = _metric_stats(results[scenario][agent], "degradation", "success_rate_drop")
            mean = stats["mean"]
            lower = stats["ci_lower"] if stats["ci_lower"] is not None else mean
            upper = stats["ci_upper"] if stats["ci_upper"] is not None else mean
            ys.append(mean)
            uppers.append(None if mean is None else max(0.0, upper - mean))
            lowers.append(None if mean is None else max(0.0, mean - lower))
            custom.append([lower, upper])

        fig.add_trace(
            go.Bar(
                x=[_scenario_label(s) for s in scenarios],
                y=ys,
                name=label,
                marker_color=_agent_color(agent),
                error_y=dict(type="data", array=uppers, arrayminus=lowers, visible=True),
                customdata=custom,
                hovertemplate=(
                    label + ": %{y:.3f}<br>"
                    + "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Robustness: success-rate drop (normal to adversarial). Lower is better.",
        yaxis_title="delta success rate",
        barmode="group",
        height=400,
    )
    return fig


def _by_ses_heatmap(results: dict, agents: list[str], labels: dict[str, str], metric: str, title: str) -> go.Figure:
    rows = []
    row_labels = []
    for scenario, agent_data in results.items():
        for agent in agents:
            if agent not in agent_data:
                continue
            adv = agent_data[agent].get("by_phase", {}).get("adversarial", {})
            by_ses = adv.get("by_ses", {})
            row = [
                by_ses.get(str(s), {}).get(metric)
                if by_ses.get(str(s)) is not None
                else by_ses.get(s, {}).get(metric)
                for s in [0, 1, 2]
            ]
            rows.append(row)
            row_labels.append(f"{scenario[:20]} | {_agent_label(agent, labels)}")

    fig = go.Figure(
        data=go.Heatmap(
            z=rows,
            x=["SES 0 (low)", "SES 1 (med)", "SES 2 (high)"],
            y=row_labels,
            colorscale="RdYlGn" if metric == "success_rate" else "RdYlGn_r",
            colorbar=dict(title=metric),
            hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title} (adversarial phase, pooled by SES)",
        height=500,
        margin=dict(l=260),
    )
    return fig


def _summary_table(results: dict, agents: list[str], labels: dict[str, str]) -> go.Figure:
    headers = [
        "scenario",
        "agent",
        "success (95% CI)",
        "fairness gap (95% CI)",
        "near-optimal rate (95% CI)",
        "robustness drop (95% CI)",
    ]
    rows: list[list[str]] = [[] for _ in headers]

    for scenario, agent_data in results.items():
        for agent in agents:
            if agent not in agent_data:
                continue
            success = _metric_stats(agent_data[agent], "overall", "success_rate")
            fairness = _metric_stats(agent_data[agent], "overall", "fairness_gap_ses")
            precision = _metric_stats(agent_data[agent], "overall", "near_optimal_rate")
            robustness = _metric_stats(agent_data[agent], "degradation", "success_rate_drop")
            values = [
                _scenario_label(scenario),
                _agent_label(agent, labels),
                _fmt_interval(success),
                _fmt_interval(fairness),
                _fmt_interval(precision),
                _fmt_interval(robustness),
            ]
            for idx, value in enumerate(values):
                rows[idx].append(value)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#2c3e50",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=rows,
                    align="left",
                    fill_color=[["#f8f9fa", "#ffffff"] * (len(rows[0]) // 2 + 1)],
                ),
            )
        ]
    )
    fig.update_layout(title="Seed-level summary table", height=430)
    return fig


def _tradeoff_scatter(results: dict, agents: list[str], labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for agent in agents:
        label = _agent_label(agent, labels)
        x_vals = []
        y_vals = []
        sizes = []
        text = []
        custom = []

        for scenario, agent_results in results.items():
            if agent not in agent_results:
                continue
            success = _metric_stats(agent_results[agent], "overall", "success_rate")
            fairness = _metric_stats(agent_results[agent], "overall", "fairness_gap_ses")
            robustness = _metric_stats(agent_results[agent], "degradation", "success_rate_drop")
            if None in (success["mean"], fairness["mean"], robustness["mean"]):
                continue

            x_vals.append(fairness["mean"])
            y_vals.append(success["mean"])
            sizes.append(18 + 50 * max(0.0, 1.0 - robustness["mean"]))
            text.append(_scenario_label(scenario))
            custom.append([
                fairness["ci_lower"], fairness["ci_upper"],
                success["ci_lower"], success["ci_upper"],
                robustness["mean"], robustness["ci_lower"], robustness["ci_upper"],
            ])

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                name=label,
                text=text,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    color=_agent_color(agent),
                    opacity=0.78,
                    line=dict(color="white", width=1.5),
                ),
                customdata=custom,
                hovertemplate=(
                    "%{text}<br>"
                    + "fairness gap: %{x:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
                    + "success rate: %{y:.3f} [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>"
                    + "robustness drop: %{customdata[4]:.3f} [%{customdata[5]:.3f}, %{customdata[6]:.3f}]<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Tradeoffs: raw utility vs fairness vs robustness",
        xaxis_title="fairness gap across SES (lower is better)",
        yaxis_title="success rate (higher is better)",
        height=520,
    )
    return fig


def _negative_result_table(
    payload: dict,
    results: dict,
    agents: list[str],
    labels: dict[str, str],
) -> go.Figure | None:
    negative = _negative_result(payload)
    if not negative:
        return None

    scenario = negative["scenario"]
    primary_agent = _primary_claim_agent(payload, agents)
    focal_agent = negative.get("focal_agent", primary_agent)
    focal_label = negative.get("focal_label", _agent_label(focal_agent, labels))
    competitor = negative["competitor"]
    competitor_label = negative.get("competitor_label", _agent_label(competitor, labels))
    causal = results[scenario][focal_agent]
    other = results[scenario][competitor]

    metric_specs = [
        ("overall", "success_rate", "success rate"),
        ("overall", "fairness_gap_ses", "fairness gap"),
        ("degradation", "success_rate_drop", "robustness drop"),
    ]
    metric_labels = []
    causal_vals = []
    competitor_vals = []
    favored = []
    p_values = []

    for slice_name, metric, label in metric_specs:
        metric_labels.append(label)
        causal_stats = _metric_stats(causal, slice_name, metric)
        other_stats = _metric_stats(other, slice_name, metric)
        causal_vals.append(_fmt_interval(causal_stats))
        competitor_vals.append(_fmt_interval(other_stats))
        negative_metric = negative["metrics"][metric]
        if metric in {"fairness_gap_ses", "success_rate_drop"}:
            favored_agent = focal_label if negative_metric["focal"] < negative_metric["competitor"] else competitor_label
        else:
            favored_agent = focal_label if negative_metric["focal"] > negative_metric["competitor"] else competitor_label
        favored.append(favored_agent)
        p_values.append(_fmt_p(negative_metric.get("p_value")))

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["metric", focal_label, competitor_label, "favored", "paired p-value"],
                    fill_color="#5d4037",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[metric_labels, causal_vals, competitor_vals, favored, p_values],
                    align="left",
                    fill_color="#fff8f5",
                ),
            )
        ]
    )
    fig.update_layout(
        title=(
            f"Primary {_agent_label(primary_agent, labels)} Negative Result: "
            f"{_scenario_label(scenario)} ({competitor_label} vs {focal_label})"
        ),
        height=300,
    )
    return fig


def _significance_highlights(payload: dict, agents: list[str], labels: dict[str, str]) -> go.Figure | None:
    pairwise = _pairwise_tests(payload)
    primary_agent = _primary_claim_agent(payload, agents)
    primary_label = _agent_label(primary_agent, labels)
    causal_rows = [
        row for row in pairwise
        if primary_agent in {row["agent_a"], row["agent_b"]}
        and row["slice"] in {"overall", "degradation"}
        and row["metric"] in {"success_rate", "fairness_gap_ses", "success_rate_drop"}
    ]
    causal_rows = sorted(causal_rows, key=lambda row: (row["p_value"], -abs(row["effect_size_dz"])))[:9]
    if not causal_rows:
        return None

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["scenario", "slice", "metric", "comparison", "favored", "p", "effect size"],
                    fill_color="#1f3a5f",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[
                        [_scenario_label(row["scenario"]) for row in causal_rows],
                        [row["slice"] for row in causal_rows],
                        [row["metric"] for row in causal_rows],
                        [
                            f"{_agent_label(row['agent_a'], labels)} vs "
                            f"{_agent_label(row['agent_b'], labels)}"
                            for row in causal_rows
                        ],
                        [_agent_label(row["favored_agent"], labels) for row in causal_rows],
                        [_fmt_p(row["p_value"]) for row in causal_rows],
                        [f"{row['effect_size_dz']:.3f} ({row['effect_size_label']})" for row in causal_rows],
                    ],
                    align="left",
                    fill_color="#f5f8fb",
                ),
            )
        ]
    )
    fig.update_layout(title=f"Paired significance highlights for {primary_label}", height=360)
    return fig


def build_dashboard(payload: dict) -> str:
    results = _scenario_results(payload)
    agents = _agent_ids(payload, results)
    labels = _agent_labels(payload, agents)
    primary_agent = _primary_claim_agent(payload, agents)
    negative = _negative_result(payload)
    figures = [
        _summary_table(results, agents, labels),
        _tradeoff_scatter(results, agents, labels),
    ]

    negative_figure = _negative_result_table(payload, results, agents, labels)
    if negative_figure is not None:
        figures.append(negative_figure)

    significance_figure = _significance_highlights(payload, agents, labels)
    if significance_figure is not None:
        figures.append(significance_figure)

    figures.extend([
        _phase_metric_subplots(results, agents, labels, "success_rate", "Success rate by phase", "success rate"),
        _phase_metric_subplots(results, agents, labels, "near_optimal_rate", "Near-optimal rate by phase", "near-optimal rate"),
        _phase_metric_subplots(results, agents, labels, "fairness_gap_ses", "Fairness gap across SES (lower is better)", "gap"),
        _phase_metric_subplots(results, agents, labels, "under_treatment_disparity", "Under-treatment disparity (SES 0 - SES 2)", "disparity"),
        _degradation_chart(results, agents, labels),
        _by_ses_heatmap(results, agents, labels, "success_rate", "Success rate"),
        _by_ses_heatmap(results, agents, labels, "under_treatment_rate", "Under-treatment rate"),
    ])

    parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>v3_health: Healthcare Decision Benchmark</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1400px;"
        "margin:20px auto;padding:0 20px;background:#fafafa;}"
        "h1{color:#2c3e50;}h2{color:#34495e;margin-top:40px;}"
        ".chart{background:white;padding:10px;margin:20px 0;"
        "border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}"
        ".callout{background:#ffffff;border-left:4px solid #2c7fb8;padding:16px 18px;"
        "border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin:18px 0;}"
        ".negative{border-left-color:#b23a48;background:#fff7f8;}</style>",
        "</head><body>",
        "<h1>v3_health: Healthcare Decision Benchmark</h1>",
        f"<p>{_intro_text(agents)}</p>",
        "<div class='callout'><strong>Statistical view.</strong> "
        "All bar charts and tradeoff panels use seed-level means with 95% Student-t CIs (n=12 seeds). "
        "The primary robustness estimand is the matched-seed drop in success rate from normal to adversarial "
        "with agent state carried across phase boundaries. Paired comparisons use sign-flip permutation tests "
        "and Cohen's dz effect sizes."
        "</div>",
        "<div class='callout'><strong>Interpretation note.</strong> "
        "Workflow is the non-causal adaptive comparator. Causal Light and Stability-Filtered are "
        "causal-scaffolding heuristics rather than identified causal estimators. Structural Causal is the "
        "explicit latent-state interventional agent. Parent-alignment and negative-result panels are "
        "diagnostic summaries, not individual-level counterfactual evidence."
        "</div>",
    ]

    if negative:
        parts.append(
            "<div class='callout negative'><strong>Negative result.</strong> "
            + negative["summary"]
            + " "
            + _negative_result_blurb(primary_agent)
            + "</div>"
        )

    for idx, fig in enumerate(figures):
        parts.append("<div class='chart'>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False))
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def render_dashboard(results_path: Path, output_path: Path) -> Path:
    payload = _load(results_path)
    html = build_dashboard(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard written to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/results_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dashboard.html"),
    )
    args = parser.parse_args()

    render_dashboard(args.results, args.output)


if __name__ == "__main__":
    main()
