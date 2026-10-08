import argparse
import csv
import json
import os
from pathlib import Path

import ollama
from dotenv import load_dotenv

from .decisions import run_decisions, write_decision_report
from .report import write_report
from .runner import run_task, score
from .tools import DATA_DIR

ROOT = DATA_DIR.parent
DEFAULT_MODELS = "qwen3:1.7b,qwen2.5:1.5b,granite3.3:2b,smollm2:1.7b"


LOCAL_HOST = "http://localhost:11434"


def make_client(host: str, key: str = "") -> ollama.Client:
    headers = {"Authorization": f"Bearer {key}"} if key else None
    return ollama.Client(host=host, headers=headers)


def local_models() -> set[str]:
    """Models listed in LOCAL_MODELS always run on local Ollama, whatever OLLAMA_HOST says."""
    return {m.strip() for m in os.getenv("LOCAL_MODELS", "").split(",") if m.strip()}


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Benchmark LLM tool calling with fake deterministic tools.")
    p.add_argument("--models", default=os.getenv("MODELS", DEFAULT_MODELS), help="comma-separated Ollama model names")
    p.add_argument("--repeats", type=int, default=int(os.getenv("REPEATS", "1")))
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument("--report-only", action="store_true", help="rebuild summary + charts from results/results.csv")
    p.add_argument("--decisions", action="store_true",
                   help="run the next-action decision (classification) test instead; output in results/decisions/")
    a = p.parse_args()

    cost_in = float(os.getenv("COST_PER_1M_INPUT_USD", "0.10"))
    cost_out = float(os.getenv("COST_PER_1M_OUTPUT_USD", "0.40"))
    think_env = os.getenv("THINK", "").strip().lower()
    think = {"true": True, "false": False}.get(think_env)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    results_csv = out_dir / "results.csv"
    local = local_models()

    if a.decisions:
        dec_dir = out_dir / "decisions"
        if not a.report_only:
            models = [m.strip() for m in a.models.split(",") if m.strip()]
            models += [m for m in sorted(local) if m not in models]
            cloud_host = os.getenv("OLLAMA_HOST", LOCAL_HOST)
            cloud_client = make_client(cloud_host, os.getenv("OLLAMA_API_KEY", "").strip())
            local_client = make_client(LOCAL_HOST)
            run_decisions(
                models,
                client_for=lambda m: local_client if m in local else cloud_client,
                host_for=lambda m: LOCAL_HOST if m in local else cloud_host,
                options_for=lambda m: {"temperature": 0, "seed": 42, **({"num_ctx": 8192} if m in local else {})},
                out_dir=dec_dir,
            )
        print("\nmodel                method              acc%   in_tok  out_tok  med_s")
        for s in write_decision_report(dec_dir, local):
            print(f"{s['model']:<20} {s['method']:<18} {s['accuracy_pct']:>5} {s['avg_input_tokens']:>8} "
                  f"{s['avg_output_tokens']:>8} {s['median_latency_s']:>6}")
        print(f"\nWrote {dec_dir}/decisions.csv, decision_summary.csv and decision_*_by_model.png")
        return

    if not a.report_only:
        tasks = json.loads((DATA_DIR / "tasks.json").read_text())
        if a.tasks:
            wanted = {t.strip() for t in a.tasks.split(",")}
            tasks = [t for t in tasks if t["id"] in wanted]
        models = [m.strip() for m in a.models.split(",") if m.strip()]
        models += [m for m in sorted(local) if m not in models]
        cloud_client = make_client(os.getenv("OLLAMA_HOST", LOCAL_HOST), os.getenv("OLLAMA_API_KEY", "").strip())
        local_client = make_client(LOCAL_HOST)

        rows, traces = [], []
        for model in models:
            client = local_client if model in local else cloud_client
            options = {"temperature": 0, "seed": 42}
            if model in local:
                options["num_ctx"] = 8192  # small local models default to ~2k context and silently truncate
            # Warm-up so model load time doesn't count as decision time.
            try:
                client.chat(model=model, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
            except Exception as e:
                print(f"[{model}] warm-up failed: {e}")
            for task in tasks:
                for r in range(a.repeats):
                    trace = run_task(client, model, task, options, think)
                    trace["repeat"] = r
                    row = score(trace, task, cost_in, cost_out)
                    rows.append(row)
                    traces.append(trace)
                    mark = "ok " if row["answer_correct"] else "BAD"
                    print(f"[{model}] {task['id']} #{r} {mark} calls={row['tool_calls']} "
                          f"{row['latency_s']:.2f}s final={row['final_answer']!r} {row['error']}")

        with results_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out_dir / "traces.json").write_text(json.dumps(traces, indent=2, default=str))

    summary = write_report(results_csv, cost_in, cost_out, local)
    print("\nmodel                 acc%  tools%  args%  recov%  eff   med_s   est_cost$")
    for s in summary:
        print(f"{s['model']:<20} {s['accuracy_pct']:>5} {s['tool_selection_pct']:>6} {s['args_correct_pct']:>6} "
              f"{str(s['recovery_pct']):>6} {s['mean_efficiency']:>5} {s['median_latency_s']:>6} {s['est_cost_total_usd']:>10}")
    print(f"\nWrote {out_dir}/results.csv, summary.csv, traces.json and *_by_model.png")
