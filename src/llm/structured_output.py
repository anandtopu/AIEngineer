"""Structured output: function-calling-style JSON with schema validation.

Interview goal: modern LLM APIs (OpenAI, Anthropic, Gemini) all expose
"function calling" / "tools" where the model returns a JSON object that
must match a declared schema. Be able to build this layer yourself:
  1. declare a schema
  2. prompt the model with the schema
  3. parse + validate the result
  4. retry with the validation error as feedback ("self-healing")

This pattern generalizes to tool dispatch in agents and to any
extraction task (e.g. "pull structured fields from this email").
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


# ---------- a tiny JSON-schema validator (no jsonschema dep) ----------

def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    errors: List[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected object, got {type(instance).__name__}"]
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: required field missing")
        for k, sub in schema.get("properties", {}).items():
            if k in instance:
                errors += validate(instance[k], sub, f"{path}.{k}")
    elif t == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected array"]
        items = schema.get("items")
        if items:
            for i, v in enumerate(instance):
                errors += validate(v, items, f"{path}[{i}]")
    elif t == "string":
        if not isinstance(instance, str):
            errors.append(f"{path}: expected string")
        elif "enum" in schema and instance not in schema["enum"]:
            errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    elif t == "number":
        if not isinstance(instance, (int, float)) or isinstance(instance, bool):
            errors.append(f"{path}: expected number")
        else:
            if "minimum" in schema and instance < schema["minimum"]:
                errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
            if "maximum" in schema and instance > schema["maximum"]:
                errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    elif t == "integer":
        if not isinstance(instance, int) or isinstance(instance, bool):
            errors.append(f"{path}: expected integer")
    return errors


# ---------- schema + prompt + retry loop ----------

INVOICE_SCHEMA = {
    "type": "object",
    "required": ["vendor", "total", "currency", "line_items"],
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number", "minimum": 0},
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["desc", "price"],
                "properties": {
                    "desc": {"type": "string"},
                    "price": {"type": "number", "minimum": 0},
                },
            },
        },
    },
}


def build_prompt(doc: str, schema: Dict[str, Any], feedback: str = "") -> str:
    base = (
        "Extract the invoice fields from the document below. "
        "Return ONLY valid JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Document:\n{doc}\n"
    )
    if feedback:
        base += f"\nYour previous attempt was invalid: {feedback}\nRetry and fix the issues.\n"
    return base


def extract_json(text: str) -> Any:
    # Real models sometimes wrap JSON in code fences or prose.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def call_with_self_healing(llm, doc: str, schema: Dict[str, Any], max_retries: int = 2):
    feedback = ""
    for attempt in range(max_retries + 1):
        prompt = build_prompt(doc, schema, feedback)
        raw = llm(prompt, attempt)
        try:
            data = extract_json(raw)
        except Exception as e:
            feedback = f"JSON parse error: {e}"
            continue
        errors = validate(data, schema)
        if not errors:
            return data, attempt + 1
        feedback = "; ".join(errors[:3])
    raise RuntimeError(f"failed after {max_retries + 1} attempts: {feedback}")


def mock_llm(prompt: str, attempt: int) -> str:
    # Attempt 0: returns malformed JSON (extra trailing comma, wrong currency).
    # Attempt 1: fixes currency but total is a string.
    # Attempt 2: finally valid.
    if attempt == 0:
        return '{"vendor": "Acme Widgets", "total": 142.50, "currency": "AUD", '\
               '"line_items": [{"desc": "Sprocket", "price": 42.5}]}'
    if attempt == 1:
        return '{"vendor": "Acme Widgets", "total": "142.50", "currency": "USD", '\
               '"line_items": [{"desc": "Sprocket", "price": 42.5}, '\
               '{"desc": "Flange", "price": 100.0}]}'
    return '{"vendor": "Acme Widgets", "total": 142.50, "currency": "USD", '\
           '"line_items": [{"desc": "Sprocket", "price": 42.5}, '\
           '{"desc": "Flange", "price": 100.0}]}'


def main():
    doc = "Invoice from Acme Widgets. Sprocket $42.50; Flange $100.00. Total $142.50 USD."
    data, n = call_with_self_healing(mock_llm, doc, INVOICE_SCHEMA)
    print(f"Succeeded on attempt {n}:")
    print(json.dumps(data, indent=2))

    print("\nFailure cases caught by validator on earlier attempts:")
    print("  - 'AUD' not in enum ['USD','EUR','GBP']")
    print("  - total='142.50' is a string, schema requires number")
    print("\nIn production, set max_retries=2 and budget tokens accordingly.")


if __name__ == "__main__":
    main()
