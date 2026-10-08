"""Decision (classification) benchmark: given a situation, which action should the agent take next?

Every model gets the same 20 situations and the same 5 options.
Decision models (Ollama capability "decision", e.g. tev1) answer via /v1/systemone;
chat models answer the same question as a plain prompt.
"""

import csv
import json
import re
import statistics
import time
from pathlib import Path

import httpx
import ollama

from .report import _bar_chart, token_chart
from .tools import DATA_DIR

OPTIONS = {
    "search_products": "Search the catalogue by name, category or max price. Returns id, name and price, not stock.",
    "check_stock": "Check how many units of a product id are in stock.",
    "place_order": "Place an order for a product id and quantity.",
    "get_shipping_estimate": "Estimate delivery days to a PIN code.",
    "answer_now": "Stop calling tools and give the customer the final answer.",
}
INSTRUCTIONS = (
    "You are the decision step of a shopping-assistant agent. Given the customer request and the steps so far, "
    "which action should the agent take next? If a tool failed with a temporary error, retrying it is allowed."
)


def is_decision_model(client: ollama.Client, model: str) -> bool:
    try:
        return "decision" in (client.show(model).capabilities or [])
    except Exception:
        return False


def ask_decision_model(host: str, model: str, state: str) -> tuple[str, int, int]:
    body = {
        "model": model,
        "state": state,
        "questions": {"next_action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": OPTIONS}},
    }
    r = httpx.post(f"{host.rstrip('/')}/v1/systemone", json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return data["answers"]["next_action"]["choice"], usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def ask_chat_model(client: ollama.Client, model: str, state: str, options: dict) -> tuple[str, int, int]:
    listed = "\n".join(f"- {k}: {v}" for k, v in OPTIONS.items())
    prompt = f"{INSTRUCTIONS}\n\n{state}\n\nOptions:\n{listed}\n\nReply with only the option name, nothing else."
    resp = client.chat(model=model, messages=[{"role": "user", "content": prompt}], options=options)
    text = (resp.message.content or "").lower()
    found = sorted((m.start(), k) for k in OPTIONS if (m := re.search(rf"\b{k}\b", text)))
    return (found[0][1] if found else ""), resp.prompt_eval_count or 0, resp.eval_count or 0


def run_decisions(models: list[str], client_for, host_for, options_for, out_dir: Path) -> None:
    points = json.loads((DATA_DIR / "decisions.json").read_text())
    rows = []
    for model in models:
        client = client_for(model)
        decision = is_decision_model(client, model)
        method = "decision endpoint" if decision else "chat prompt"
        print(f"[{model}] {len(points)} decisions via {method}")
        try:  # warm-up so model load time doesn't count
            client.generate(model=model, prompt="hi", options={"num_predict": 1})
        except Exception as e:
            print(f"[{model}] warm-up failed: {e}")
        for p in points:
            t0 = time.perf_counter()
            error = ""
            try:
                if decision:
                    choice, tin, tout = ask_decision_model(host_for(model), model, p["state"])
                else:
                    choice, tin, tout = ask_chat_model(client, model, p["state"], options_for(model))
            except Exception as e:
                choice, tin, tout, error = "", 0, 0, f"{type(e).__name__}: {e}"
            row = {
                "model": model, "method": method, "decision_id": p["id"], "kind": p["kind"],
                "accepted": "|".join(p["accept"]), "choice": choice, "correct": choice in p["accept"],
                "input_tokens": tin, "output_tokens": tout,
                "latency_s": round(time.perf_counter() - t0, 3), "error": error,
            }
            rows.append(row)
            print(f"[{model}] {p['id']} {'ok ' if row['correct'] else 'BAD'} chose={choice!r} "
                  f"want={row['accepted']} {row['latency_s']:.2f}s {error}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_decision_report(out_dir: Path, local: set[str]) -> list[dict]:
    rows = list(csv.DictReader((out_dir / "decisions.csv").open()))
    models = list(dict.fromkeys(r["model"] for r in rows))
    summary = []
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        summary.append({
            "model": m,
            "method": rs[0]["method"],
            "decisions": len(rs),
            "accuracy_pct": round(100 * sum(r["correct"] == "True" for r in rs) / len(rs), 1),
            "avg_input_tokens": round(statistics.mean(int(r["input_tokens"]) for r in rs)),
            "avg_output_tokens": round(statistics.mean(int(r["output_tokens"]) for r in rs)),
            "median_latency_s": round(statistics.median(float(r["latency_s"]) for r in rs), 3),
            "errors": sum(1 for r in rs if r["error"]),
        })
    summary.sort(key=lambda s: -s["accuracy_pct"])
    with (out_dir / "decision_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    names = [s["model"] + ("\n(local)" if s["model"] in local else "") for s in summary]
    _bar_chart(names, [s["accuracy_pct"] for s in summary], "Next-action decision accuracy",
               "Same 20 situations for every model: first tool, next tool, when to stop, retry vs give up.",
               "% correct", lambda v: f"{v:.0f}%", out_dir / "decision_accuracy_by_model.png")
    _bar_chart(names, [s["median_latency_s"] for s in summary], "Median time per decision",
               "Seconds to pick one action. Local and cloud hardware differ.",
               "seconds", lambda v: f"{v:.2f}s", out_dir / "decision_time_by_model.png")
    token_chart(summary, names, out_dir / "decision_tokens_by_model.png",
                title="Tokens per decision (input + output)",
                subtitle="Average per decision. One question, one answer: no tool loop.")
    return summary
