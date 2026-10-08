"""Agent loop against Ollama + per-run scoring."""

import json
import re
import time

import ollama

from .tools import TOOL_NAMES, TOOL_SPECS, Store

SYSTEM_PROMPT = """You are a shopping assistant for an Indian electronics store. Prices are in INR.
Use the tools to look things up. Never guess product data. Only call a tool when you need it.
You can also answer general questions that need no store data (for example simple maths or a
price calculation) directly, without calling any tool.
When you are done, end your reply with exactly one line in this form:
FINAL: <answer>

Answer formats:
- list of products -> product ids, comma separated (FINAL: P1, P3), or FINAL: NONE
- yes/no question -> FINAL: YES or FINAL: NO
- price or number -> digits only (FINAL: 1499)
- order placed -> the order id (FINAL: ORD-P1-1)
- order impossible because there is not enough stock -> FINAL: OUT_OF_STOCK
- a tool keeps failing and you cannot find out -> FINAL: UNKNOWN"""

MAX_TURNS = 10


def run_task(client: ollama.Client, model: str, task: dict, options: dict, think: bool | None) -> dict:
    """Run one task once. Returns the trace; never raises on model errors."""
    store = Store(task.get("failures"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]
    calls, turns = [], []
    final_text, error = "", None
    in_tok = out_tok = 0
    latency = 0.0

    for _ in range(MAX_TURNS):
        kwargs = {"model": model, "messages": messages, "tools": TOOL_SPECS, "options": options}
        if think is not None:
            kwargs["think"] = think
        t0 = time.perf_counter()
        try:
            resp = client.chat(**kwargs)
        except Exception as e:  # model missing, no tool support, network...
            error = f"{type(e).__name__}: {e}"
            break
        dt = time.perf_counter() - t0
        latency += dt
        in_tok += resp.prompt_eval_count or 0
        out_tok += resp.eval_count or 0
        msg = resp.message
        turns.append({"seconds": round(dt, 3), "content": msg.content or "", "tool_calls": len(msg.tool_calls or [])})
        messages.append(msg)

        if not msg.tool_calls:
            final_text = msg.content or ""
            break
        for tc in msg.tool_calls:
            name, args = tc.function.name, dict(tc.function.arguments or {})
            result = store.call(name, args)
            calls.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    else:
        error = "max_turns"

    return {
        "model": model,
        "task_id": task["id"],
        "category": task["category"],
        "calls": calls,
        "turns": turns,
        "final_text": final_text,
        "error": error,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_s": latency,
    }


# ---------- scoring ----------

def _norm(s: str) -> str:
    s = s.upper().replace(",", "")
    if num := re.search(r"\d+(\.\d+)?", s):
        if not re.search(r"[A-Z]-?\d", s):  # a plain number, not an id like ORD-P9-2
            return str(float(num.group()))
    return re.sub(r"[\s.]", "", s)


def extract_final(text: str) -> str | None:
    found = re.findall(r"FINAL\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return found[-1].strip().strip("`*") if found else None


def answer_correct(final: str | None, expected: dict) -> bool:
    if final is None:
        return False
    if expected["type"] == "set":
        got = {m.upper() for m in re.findall(r"\bP\d+\b", final, flags=re.IGNORECASE)}
        return got == set(expected["value"])
    return _norm(final) == _norm(expected["value"])


def _args_match(expected: dict, actual: dict) -> bool:
    for k, v in expected.items():
        if k not in actual or str(actual[k]).strip().upper() != str(v).strip().upper():
            return False
    return True


def score(trace: dict, task: dict, cost_in: float, cost_out: float) -> dict:
    calls = trace["calls"]
    called = [c["tool"] for c in calls]
    final = extract_final(trace["final_text"])
    correct = answer_correct(final, task["answer"])

    selection_ok = set(task["required_tools"]) <= set(called) and all(t in task["allowed_tools"] for t in called)
    args_ok = all(
        any(c["tool"] == e["tool"] and _args_match(e["args"], c["args"]) for c in calls)
        for e in task["expected_calls"]
    )
    invalid = sum(1 for c in calls if c["tool"] not in TOOL_NAMES or c["result"].get("error") == "invalid_argument")
    n, m = len(calls), task["min_calls"]
    efficiency = min(1.0, (m + 1) / (n + 1))

    recovered = None
    if task.get("failures"):
        recovered = correct

    cost = trace["input_tokens"] / 1e6 * cost_in + trace["output_tokens"] / 1e6 * cost_out
    return {
        "model": trace["model"],
        "task_id": trace["task_id"],
        "category": trace["category"],
        "repeat": trace.get("repeat", 0),
        "final_answer": final if final is not None else "",
        "answer_correct": correct,
        "tool_selection_ok": selection_ok,
        "args_ok": args_ok,
        "tool_calls": n,
        "min_calls": m,
        "efficiency": round(efficiency, 3),
        "invalid_calls": invalid,
        "recovered": "" if recovered is None else recovered,
        "input_tokens": trace["input_tokens"],
        "output_tokens": trace["output_tokens"],
        "latency_s": round(trace["latency_s"], 3),
        "est_cost_usd": round(cost, 8),
        "error": trace["error"] or "",
    }
