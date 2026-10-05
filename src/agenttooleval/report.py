"""CSV summary + matplotlib charts from results.csv."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"


def _truthy(v: str) -> bool:
    return v == "True"


def summarize(rows: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    out = []
    for model, rs in by_model.items():
        ok = [r for r in rs if not r["error"] or r["error"] == "max_turns"]
        n = len(rs)
        rec = [r for r in rs if r["recovered"] != ""]
        correct = sum(_truthy(r["answer_correct"]) for r in rs)
        cost = sum(float(r["est_cost_usd"]) for r in rs)
        out.append({
            "model": model,
            "runs": n,
            "errored_runs": n - len(ok),
            "accuracy_pct": round(100 * correct / n, 1),
            "tool_selection_pct": round(100 * sum(_truthy(r["tool_selection_ok"]) for r in rs) / n, 1),
            "args_correct_pct": round(100 * sum(_truthy(r["args_ok"]) for r in rs) / n, 1),
            "recovery_pct": round(100 * sum(_truthy(r["recovered"]) for r in rec) / len(rec), 1) if rec else "",
            # Efficiency only counts on correct runs; a wrong answer in 1 call isn't "efficient".
            "mean_efficiency": round(statistics.mean(float(r["efficiency"]) for r in rs if _truthy(r["answer_correct"])), 3) if correct else "",
            "avg_tool_calls": round(statistics.mean(int(r["tool_calls"]) for r in rs), 2),
            "invalid_calls_total": sum(int(r["invalid_calls"]) for r in rs),
            "avg_tokens_per_task": round(statistics.mean(int(r["input_tokens"]) + int(r["output_tokens"]) for r in rs)),
            "median_latency_s": round(statistics.median(float(r["latency_s"]) for r in rs), 3),
            "est_cost_total_usd": round(cost, 6),
            "est_cost_per_correct_usd": round(cost / correct, 8) if correct else "",
        })
    return sorted(out, key=lambda s: -s["accuracy_pct"])


def _bar_chart(names, values, title, subtitle, ylabel, fmt, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.bar(names, values, width=0.5, color=BAR, edgecolor=SURFACE, linewidth=2, zorder=3)
    for b, v in zip(bars, values):
        ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=22, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(0, max(values or [1]) * 1.18 or 1)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_report(results_csv: Path, cost_in: float, cost_out: float) -> list[dict]:
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        return []
    summary = summarize(rows)
    out_dir = results_csv.parent
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    names = [s["model"] for s in summary]
    _bar_chart(
        names, [s["est_cost_total_usd"] for s in summary],
        "Estimated cost per model (whole benchmark)",
        f"Tokens x assumed rate ${cost_in}/1M input, ${cost_out}/1M output. Runs were free.",
        "USD", lambda v: f"${v:.4f}", out_dir / "cost_by_model.png",
    )
    _bar_chart(
        names, [s["median_latency_s"] for s in summary],
        "Median decision time per task",
        "Wall-clock time spent in the model per task (tools are instant). Lower is faster.",
        "seconds", lambda v: f"{v:.2f}s", out_dir / "time_by_model.png",
    )
    _bar_chart(
        names, [s["accuracy_pct"] for s in summary],
        "Final-answer accuracy",
        "Share of task runs with the correct FINAL answer.",
        "% correct", lambda v: f"{v:.0f}%", out_dir / "accuracy_by_model.png",
    )
    return summary
