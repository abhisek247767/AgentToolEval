#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import time
import kaggle_benchmarks as kbench


# In[ ]:


PRODUCTS = [
    {"id": "P1", "name": "Lenovo IdeaPad Slim 3", "category": "laptop", "price": 42990, "stock": 5},
    {"id": "P2", "name": "HP 15s",                "category": "laptop", "price": 38500, "stock": 0},
    {"id": "P3", "name": "Acer Aspire 5",         "category": "laptop", "price": 45990, "stock": 3},
    {"id": "P4", "name": "ASUS Vivobook 15",      "category": "laptop", "price": 49999, "stock": 0},
    {"id": "P5", "name": "Dell Inspiron 3520",    "category": "laptop", "price": 50000, "stock": 7},
    {"id": "P6", "name": "MacBook Air M2",        "category": "laptop", "price": 89900, "stock": 2},
]
BY_ID = {p["id"]: p for p in PRODUCTS}

TASKS = [
    {"id": "T04", "prompt": "What is the cheapest laptop I can buy today (it must be in stock)?",
     "answer": ("set", ["P1"]), "required": ["search_products", "check_stock"],
     "expected": [("check_stock", {"product_id": "P1"})], "min_calls": 3, "fail_tool": None},
    {"id": "T09", "prompt": "What is the price of the MacBook Air M2?",
     "answer": ("value", "89900"), "required": ["search_products"],
     "expected": [], "min_calls": 2, "fail_tool": "search_products"},
]

SYSTEM_PROMPT = """You are a shopping assistant for an Indian electronics store. Prices are in INR.
Use the tools to look things up. Never guess product data. Only call a tool when you need it.
When you are done, end your reply with exactly one line:
FINAL: <answer>
- product -> its id, e.g. FINAL: P1
- price -> digits only, e.g. FINAL: 1499
- a tool keeps failing -> FINAL: UNKNOWN"""


# In[ ]:


def make_tools(fail_tool=None):
    log = []
    state = {"failed": False}

    def run(name, args, fn):
        if name == fail_tool and not state["failed"]:
            state["failed"] = True
            result = {"error": "timeout", "message": "Service timed out. Please retry."}
        else:
            result = fn()
        log.append({"tool": name, "args": args, "result": result})
        return result

    def search_products(query: str = "", category: str = "", max_price: int = 0) -> dict:
        """Search products by name words, category (laptop) and/or max_price in INR (0 = no limit). Returns id, name, category, price. Does NOT return stock."""
        def do():
            cat = str(category or "").strip().lower().rstrip("s")
            words = re.findall(r"[a-z0-9]+", str(query or "").lower())
            limit = float(max_price or 0)
            hits = []
            for p in PRODUCTS:
                text = (p["name"] + " " + p["category"]).lower()
                if cat and p["category"] != cat:
                    continue
                if limit and p["price"] > limit:
                    continue
                if words and not all(w in text for w in words):
                    continue
                hits.append({k: p[k] for k in ("id", "name", "category", "price")})
            return {"results": hits}
        return run("search_products", {"query": query, "category": category, "max_price": max_price}, do)

    def check_stock(product_id: str) -> dict:
        """Check if a product is in stock, by product id such as 'P1'."""
        def do():
            p = BY_ID.get(str(product_id).strip().upper())
            if p is None:
                return {"error": "invalid_argument", "message": f"No product with id {product_id!r}."}
            return {"product_id": p["id"], "in_stock": p["stock"] > 0, "units": p["stock"]}
        return run("check_stock", {"product_id": product_id}, do)

    def get_shipping_estimate(pincode: str) -> dict:
        """Estimate delivery days to an Indian PIN code."""
        return run("get_shipping_estimate", {"pincode": pincode}, lambda: {"days": 3})

    return [search_products, check_stock, get_shipping_estimate], log


# In[ ]:


def extract_final(text):
    found = re.findall(r"FINAL\s*:\s*(.+)", str(text or ""), flags=re.IGNORECASE)
    return found[-1].strip().strip("`*") if found else None

def answer_correct(final, answer):
    kind, value = answer
    if final is None:
        return False
    if kind == "set":
        ids = {m.upper() for m in re.findall(r"\bP\d+\b", final, flags=re.IGNORECASE)}
        return ids == set(value)
    digits = re.sub(r"[^\d]", "", final)
    return digits == value


# In[ ]:


@kbench.task(name="agent-tool-use",
             description="Right tool, right arguments, few steps, and recovery from a tool failure (2-scenario POC).")
def agent_tool_use(llm) -> float:
    scores = []
    for t in TASKS:
        print(f"{t['id']} starting", flush=True)
        tools, log = make_tools(t["fail_tool"])
        text, error = "", ""

        with kbench.chats.new(f"case-{t['id']}", system_instructions=SYSTEM_PROMPT):
            t0 = time.perf_counter()
            try:
                text = llm.prompt(t["prompt"], tools=tools)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            secs = time.perf_counter() - t0

        called = [c["tool"] for c in log]
        final = extract_final(text)
        correct = answer_correct(final, t["answer"])
        tools_ok = set(t["required"]) <= set(called) and all(c in t["required"] for c in called)
        args_ok = all(
            any(c["tool"] == tool and str(c["args"].get(k, "")).upper() == str(v).upper()
                for c in log for k, v in exp.items())
            for tool, exp in t["expected"]
        )
        if error:
            correct = tools_ok = args_ok = False
        efficiency = min(1.0, (t["min_calls"] + 1) / (len(log) + 1))

        score = 0.6 * correct + 0.15 * tools_ok + 0.15 * args_ok + 0.10 * (efficiency if correct else 0)
        kbench.assertions.assert_true(correct, expectation=f"{t['id']}: correct answer (got {final!r})")

        print(f"{t['id']} done: score={score:.2f} final={final!r} calls={called} "
              f"{secs:.1f}s error={error or 'none'}", flush=True)
        scores.append(score)

    return round(sum(scores) / len(scores), 4)


# In[ ]:


agent_tool_use.run(kbench.llm)

