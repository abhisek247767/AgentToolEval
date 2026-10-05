# AgentCallBench (POC)

Measures whether LLM agents pick the right tools, pass correct arguments,
avoid wasted calls, and recover from tool failures. Tools are simulated
and deterministic.

## Run
    uv sync
    cp .env.example .env   # add your OLLAMA_API_KEY
    uv run agentcallbench

Results go to `results/results.csv` and `results/traces.json`.
