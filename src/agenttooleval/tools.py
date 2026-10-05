"""Fake, deterministic store tools. Same inputs always give the same output."""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the store catalogue. Returns id, name, category and price (INR). Does NOT return stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words from the product name, e.g. 'HP 15s'."},
                    "category": {"type": "string", "enum": ["laptop", "phone", "headphones"]},
                    "max_price": {"type": "integer", "description": "Only return products priced at or below this (INR)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check how many units of a product are in stock.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string", "description": "Product id, e.g. 'P1'."}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place an order. Fails with out_of_stock if not enough units are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                },
                "required": ["product_id", "quantity"],
            },
        },
    },
    {
        # Distractor: never needed by any task. Calling it counts as a wrong tool.
        "type": "function",
        "function": {
            "name": "get_shipping_estimate",
            "description": "Estimate delivery days to an Indian PIN code.",
            "parameters": {
                "type": "object",
                "properties": {"pincode": {"type": "string"}},
                "required": ["pincode"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}


def load_products() -> dict[str, dict]:
    products = json.loads((DATA_DIR / "products.json").read_text())
    return {p["id"]: p for p in products}


def _err(kind: str, message: str) -> dict:
    return {"error": kind, "message": message}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Store:
    """One Store per run, so injected failures are counted per run."""

    def __init__(self, failures: list[dict] | None = None):
        self.products = load_products()
        self.failures = [dict(f, used=0) for f in (failures or [])]

    def call(self, name: str, args: dict) -> dict:
        if name not in TOOL_NAMES:
            return _err("invalid_argument", f"Unknown tool '{name}'.")
        for f in self.failures:
            if f["tool"] == name and (f["times"] < 0 or f["used"] < f["times"]):
                f["used"] += 1
                return _err(f["error"], f["message"])
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as e:
            return _err("invalid_argument", str(e))

    def _product(self, product_id) -> dict | None:
        return self.products.get(str(product_id).strip().upper())

    def search_products(self, query: str = "", category: str = "", max_price=None) -> dict:
        try:
            limit = None if max_price in (None, "") else float(max_price)
        except (TypeError, ValueError):
            return _err("invalid_argument", f"max_price must be a number, got {max_price!r}.")
        cat = str(category or "").strip().lower().rstrip("s")
        q = _tokens(str(query or ""))
        hits = []
        for p in self.products.values():
            if cat and p["category"].rstrip("s") != cat:
                continue
            if limit is not None and p["price"] > limit:
                continue
            if q and not all(t in _tokens(p["name"] + " " + p["category"]) for t in q):
                continue
            hits.append({k: p[k] for k in ("id", "name", "category", "price")})
        return {"results": hits}

    def check_stock(self, product_id) -> dict:
        p = self._product(product_id)
        if p is None:
            return _err("invalid_argument", f"No product with id {product_id!r}.")
        return {"product_id": p["id"], "in_stock": p["stock"] > 0, "units": p["stock"]}

    def place_order(self, product_id, quantity) -> dict:
        p = self._product(product_id)
        if p is None:
            return _err("invalid_argument", f"No product with id {product_id!r}.")
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            return _err("invalid_argument", f"quantity must be an integer, got {quantity!r}.")
        if qty < 1:
            return _err("invalid_argument", "quantity must be at least 1.")
        if p["stock"] < qty:
            return _err("out_of_stock", f"Only {p['stock']} units of {p['id']} available.")
        return {"order_id": f"ORD-{p['id']}-{qty}", "status": "confirmed"}

    def get_shipping_estimate(self, pincode) -> dict:
        return {"pincode": str(pincode), "days": 3}
