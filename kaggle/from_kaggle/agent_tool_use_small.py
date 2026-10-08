
import re
import time
import kaggle_benchmarks as kbench




PRODUCTS = [
    {"id": "P1", "name": "Lenovo IdeaPad Slim 3", "category": "laptop", "price": 42990, "stock": 5},
    {"id": "P2", "name": "HP 15s",                "category": "laptop", "price": 38500, "stock": 0},
    {"id": "P3", "name": "Acer Aspire 5",         "category": "laptop", "price": 45990, "stock": 3},
    {"id": "P4", "name": "ASUS Vivobook 15",      "category": "laptop", "price": 49999, "stock": 0},
    {"id": "P5", "name": "Dell Inspiron 3520",    "category": "laptop", "price": 50000, "stock": 7},
    {"id": "P6", "name": "MacBook Air M2",        "category": "laptop", "price": 89900, "stock": 2},
    {"id": "P7", "name": "Redmi Note 13",         "category": "phone",  "price": 17999, "stock": 10},
    {"id": "P8", "name": "Samsung Galaxy M34",    "category": "phone",  "price": 15999, "stock": 0},
]
BY_ID = {p["id"]: p for p in PRODUCTS}

S, C, O = "search_products", "check_stock", "place_order"

TASKS = [
    {"id": "T03", "prompt": "Which laptops under 50000 rupees are in stock right now?",
     "answer": ("set", ["P1", "P3"]), "required": [S, C], "allowed": [S, C],
     "expected": [(C, {"product_id": "P1"}), (C, {"product_id": "P3"})],
     "min_calls": 5, "fail_tool": None, "fail_times": 0},
    {"id": "T04", "prompt": "What is the cheapest laptop I can buy today (it must be in stock)?",
     "answer": ("set", ["P1"]), "required": [S, C], "allowed": [S, C],
     "expected": [(C, {"product_id": "P1"})],
     "min_calls": 3, "fail_tool": None, "fail_times": 0},
    {"id": "T06", "prompt": "Order 1 Samsung Galaxy M34 for me.",
     "answer": ("text", "OUT_OF_STOCK"), "required": [S], "allowed": [S, C, O],
     "expected": [],
     "min_calls": 2, "fail_tool": None, "fail_times": 0},
    {"id": "T09", "prompt": "What is the price of the MacBook Air M2?",
     "answer": ("number", "89900"), "required": [S], "allowed": [S],
     "expected": [],
     "min_calls": 2, "fail_tool": S, "fail_times": 1},
    {"id": "T11", "prompt": "Is the Acer Aspire 5 in stock?",
     "answer": ("text", "UNKNOWN"), "required": [S, C], "allowed": [S, C],
     "expected": [(C, {"product_id": "P3"})],
     "min_calls": 2, "fail_tool": C, "fail_times": -1},
]

SYSTEM_PROMPT = """You are a shopping assistant for an Indian electronics store. Prices are in INR.
Use the tools to look things up. Never guess product data. Only call a tool when you need it.
When you are done, end your reply with exactly one line:
FINAL: <answer>

Answer formats:
- list of products -> ids, comma separated (FINAL: P1, P3), or FINAL: NONE
- yes/no question -> FINAL: YES or FINAL: NO
- price -> digits only (FINAL: 1499)
- order placed -> the order id (FINAL: ORD-P1-1)
- order impossible because there is not enough stock -> FINAL: OUT_OF_STOCK
- a tool keeps failing and you cannot find out -> FINAL: UNKNOWN"""



def make_tools(fail_tool=None, fail_times=0):
    log = []
    state = {"left": fail_times}

    def run(name, args, fn):
        if name == fail_tool and state["left"] != 0:
            state["left"] -= 1  # -1 keeps going negative, so it fails forever
            if fail_times == -1:
                result = {"error": "service_unavailable", "message": "503: service unavailable."}
            else:
                result = {"error": "timeout", "message": "Service timed out. Please retry."}
        else:
            result = fn()
        log.append({"tool": name, "args": args, "result": result})
        return result

    def search_products(query: str = "", category: str = "", max_price: int = 0) -> dict:
        """Search products by name words, category (laptop, phone) and/or max_price in INR (0 = no limit). Returns id, name, category, price. Does NOT return stock."""
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
        return run(S, {"query": query, "category": category, "max_price": max_price}, do)

    def check_stock(product_id: str) -> dict:
        """Check if a product is in stock, by product id such as 'P1'."""
        def do():
            p = BY_ID.get(str(product_id).strip().upper())
            if p is None:
                return {"error": "invalid_argument", "message": f"No product with id {product_id!r}."}
            return {"product_id": p["id"], "in_stock": p["stock"] > 0, "units": p["stock"]}
        return run(C, {"product_id": product_id}, do)

    def place_order(product_id: str, quantity: int) -> dict:
        """Place an order for a product id and quantity. Fails with out_of_stock if not enough units are available."""
        def do():
            p = BY_ID.get(str(product_id).strip().upper())
            if p is None:
                return {"error": "invalid_argument", "message": f"No product with id {product_id!r}."}
            try:
                qty = int(quantity)
            except (TypeError, ValueError):
                return {"error": "invalid_argument", "message": "quantity must be an integer."}
            if qty < 1:
                return {"error": "invalid_argument", "message": "quantity must be at least 1."}
            if p["stock"] < qty:
                return {"error": "out_of_stock", "message": f"Only {p['stock']} units of {p['id']} available."}
            return {"order_id": f"ORD-{p['id']}-{qty}", "status": "confirmed"}
        return run(O, {"product_id": product_id, "quantity": quantity}, do)

    def get_shipping_estimate(pincode: str) -> dict:
        """Estimate delivery days to an Indian PIN code."""
        return run("get_shipping_estimate", {"pincode": pincode}, lambda: {"days": 3})

    return [search_products, check_stock, place_order, get_shipping_estimate], log
def extract_final(text):
    found = re.findall(r"FINAL\s*:\s*(.+)", str(text or ""), flags=re.IGNORECASE)
    return found[-1].strip().strip("`*") if found else None

def answer_correct(final, answer):
    kind, value = answer
    if not final:
        return False
    if kind == "set":
        ids = {m.upper() for m in re.findall(r"\bP\d+\b", final, flags=re.IGNORECASE)}
        return ids == set(value)
    if kind == "number":
        return re.sub(r"[^\d]", "", final) == value
    # text answers like OUT_OF_STOCK, UNKNOWN, YES, NO: compare the first word
    first = re.sub(r"[^A-Z_]", "", final.upper().split()[0])
    return first == value

@kbench.task(name="agent-tool-use",
             description="Right tool, right arguments, few steps, refusing impossible orders, and recovery from tool failures (5-scenario POC).")
def agent_tool_use(llm) -> float:
    scores = []
    for t in TASKS:
        print(f"{t['id']} starting", flush=True)
        tools, log = make_tools(t["fail_tool"], t["fail_times"])
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
        tools_ok = set(t["required"]) <= set(called) and all(c in t["allowed"] for c in called)
        args_ok = all(
            any(c["tool"] == tool and all(str(c["args"].get(k, "")).upper() == str(v).upper()
                                          for k, v in exp.items())
                for c in log)
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

agent_tool_use.run(kbench.llm)

