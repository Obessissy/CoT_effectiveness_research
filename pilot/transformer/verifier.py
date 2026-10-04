from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
import unicodedata
from typing import Any


ANSWER_POLICY_VERSION = "provisional-v2-verifier-semantics"


def normalized_string(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _strict_bool(value: str) -> bool | None:
    table = {"true": True, "false": False, "yes": True, "no": False}
    return table.get(value.strip().casefold())


def verify_answer(record: dict[str, Any], prediction: str) -> dict[str, str]:
    answer_type = str(record.get("answer_type", "unknown"))
    gold = str(record.get("answer", ""))
    pred = normalized_string(prediction)
    if not pred:
        return {"status": "invalid", "method": "exact", "normalized_prediction": pred, "reason": "empty prediction"}
    if answer_type in {"symbol", "option_choice", "unknown"}:
        status = "correct" if pred == normalized_string(gold) else "incorrect"
        return {"status": status, "method": "exact", "normalized_prediction": pred, "reason": "normalized string comparison"}
    if answer_type == "boolean":
        p, g = _strict_bool(pred), _strict_bool(gold)
        if p is None or g is None:
            return {"status": "invalid", "method": "boolean", "normalized_prediction": pred, "reason": "unrecognized boolean spelling"}
        return {"status": "correct" if p == g else "incorrect", "method": "boolean", "normalized_prediction": pred, "reason": "explicit boolean table"}
    if answer_type == "integer":
        if not re.fullmatch(r"[+-]?\d+", pred):
            return {"status": "invalid", "method": "integer", "normalized_prediction": pred, "reason": "strict integer parse failed"}
        return {"status": "correct" if int(pred) == int(gold) else "incorrect", "method": "integer", "normalized_prediction": pred, "reason": "exact integer arithmetic"}
    if answer_type in {"decimal", "fraction"}:
        try:
            p = Fraction(pred) if answer_type == "fraction" else Decimal(pred)
            g = Fraction(gold) if answer_type == "fraction" else Decimal(gold)
        except (ValueError, ZeroDivisionError, InvalidOperation):
            return {"status": "invalid", "method": answer_type, "normalized_prediction": pred, "reason": "exact numeric parse failed"}
        return {"status": "correct" if p == g else "incorrect", "method": answer_type, "normalized_prediction": pred, "reason": "exact arithmetic"}
    # Expressions and structured/time/base formats require task-specific safe grammars.
    return {"status": "unsupported", "method": answer_type, "normalized_prediction": pred, "reason": "semantic rule not approved for pilot"}
