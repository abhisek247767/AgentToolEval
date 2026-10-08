# AgentToolEval

Measures whether LLM agents pick the right tools, pass correct arguments,
use as few steps as possible, recover from tool failures, and how long they
take to decide. Tools are fake and deterministic, so every run is reproducible.

## Setup
    uv sync
    cp .env.example .env          # set OLLAMA_HOST / OLLAMA_API_KEY / MODELS
    ollama pull qwen3:1.7b        # repeat for each model in MODELS (local only)

## Run
    uv run agenttooleval                          # all models, all tasks, 1 run each
    uv run agenttooleval --tasks T09,T10 --repeats 1
    uv run agenttooleval --report-only            # rebuild summary + charts
    uv run agenttooleval --decisions              # next-action decision test -> results/decisions/

## What is scored (per run)
| Column | Meaning |
|---|---|
| answer_correct | `FINAL:` line matches the expected answer |
| tool_selection_ok | all required tools called, no disallowed/distractor tools |
| args_ok | key calls had the right arguments (e.g. `place_order(P9, 2)`) |
| efficiency | (min_calls+1)/(calls+1), averaged over correct runs only |
| invalid_calls | unknown tool, bad argument, non-existent product id |
| recovered | on tasks with an injected tool failure, still got the right answer |
| latency_s | wall-clock time in the model (tools take ~0s) |
| est_cost_usd | tokens x assumed rate from `.env` (runs themselves are free) |

Tasks live in `data/tasks.json`; injected failures are the `failures` field
(`times: 1` = transient, `times: -1` = permanent, expected answer `UNKNOWN`).

## Output (`results/`)
- `results.csv` - one row per model x task x repeat
- `summary.csv` - one row per model
- `traces.json` - every tool call, argument and model reply
- `cost_by_model.png`, `time_by_model.png`, `accuracy_by_model.png`, `tokens_by_model.png`
