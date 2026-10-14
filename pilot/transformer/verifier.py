from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
import unicodedata
from typing import Any


ANSWER_POLICY_VERSION = "provisional-v3-verifier-semantics"

_INTEGER_RE = re.compile(r"[+-]?[0-9]+\Z")
_DECIMAL_RE = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_FRACTION_RE = re.compile(r"[+-]?[0-9]+/[0-9]+\Z")
_TIME_RE = re.compile(r"(?:[1-9]|1[0-2]):[0-5][0-9] [AP]M\Z", re.IGNORECASE)
_BASE_TOKEN_RE = re.compile(r"[+-]?[0-9a-f]+\Z", re.IGNORECASE)


def normalized_string(value: str) -> str:
    """Apply only the pilot's approved text normalization."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def _invalid(pred: str, method: str, reason: str) -> dict[str, str]:
    return {
        "status": "invalid",
        "method": method,
        "normalized_prediction": pred,
        "reason": reason,
    }


def _comparison(status: str, pred: str, method: str, reason: str) -> dict[str, str]:
    return {
        "status": status,
        "method": method,
        "normalized_prediction": pred,
        "reason": reason,
    }


def _strict_bool(value: str) -> bool | None:
    table = {"true": True, "false": False, "yes": True, "no": False}
    return table.get(value.casefold())


def _parse_integer(value: str) -> int | None:
    if not _INTEGER_RE.fullmatch(value):
        return None
    try:
        return int(value, 10)
    except ValueError:
        return None


def _parse_decimal(value: str) -> Decimal | None:
    if not _DECIMAL_RE.fullmatch(value):
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _parse_fraction(value: str) -> Fraction | None:
    if not _FRACTION_RE.fullmatch(value):
        return None
    numerator, denominator = value.split("/", 1)
    try:
        denominator_value = int(denominator, 10)
        if denominator_value == 0:
            return None
        return Fraction(int(numerator, 10), denominator_value)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_list_number(value: str) -> Fraction | None:
    """Parse one number-list element without binary floating point."""
    fraction = _parse_fraction(value)
    if fraction is not None:
        return fraction
    integer = _parse_integer(value)
    if integer is not None:
        return Fraction(integer, 1)
    decimal = _parse_decimal(value)
    if decimal is not None:
        numerator, denominator = decimal.as_integer_ratio()
        return Fraction(numerator, denominator)
    return None


def _parse_number_list(value: str) -> list[Fraction] | None:
    if not value or "," not in value and not value.strip():
        return None
    elements = value.split(",")
    if any(not element for element in elements):
        return None
    parsed = []
    for element in elements:
        element = element.strip()
        if not element:
            return None
        number = _parse_list_number(element)
        if number is None:
            return None
        parsed.append(number)
    return parsed


def _base_radix_from_question(question: Any) -> int | None:
    if not isinstance(question, str):
        return None
    arithmetic = re.fullmatch(
        r"\s*In base (?P<radix>[2-9]|1[0-6]), what is .+\?\s*",
        question,
        re.IGNORECASE,
    )
    if arithmetic:
        return int(arithmetic.group("radix"))
    conversion = re.fullmatch(
        r"\s*(?:Convert )?.+ \(base (?:[2-9]|1[0-6])\) (?:to|in) base (?P<target>[2-9]|1[0-6])(?:\?|\.)?\s*",
        question,
        re.IGNORECASE,
    )
    if conversion:
        return int(conversion.group("target"))
    return None


def _parse_base_integer(value: str, radix: int) -> int | None:
    if not _BASE_TOKEN_RE.fullmatch(value):
        return None
    sign = -1 if value.startswith("-") else 1
    digits = value[1:] if value[:1] in "+-" else value
    if not digits or len(digits) > 1 and digits[0] == "0":
        return None
    digit_values = []
    for digit in digits.lower():
        numeric = int(digit, 16)
        if numeric >= radix:
            return None
        digit_values.append(numeric)
    result = 0
    for digit in digit_values:
        result = result * radix + digit
    return sign * result


def _parse_time(value: str) -> int | None:
    if not _TIME_RE.fullmatch(value):
        return None
    hour_text, minute_text, meridiem = re.fullmatch(
        r"([1-9]|1[0-2]):([0-5][0-9]) ([AP]M)", value, re.IGNORECASE
    ).groups()
    hour = int(hour_text, 10) % 12
    if meridiem.upper() == "PM":
        hour += 12
    return hour * 60 + int(minute_text, 10)


def _string_type_result(record: dict[str, Any], pred: str) -> dict[str, str]:
    answer_type = str(record.get("answer_type", "unknown"))
    method = {
        "symbol": "symbolic",
        "option_choice": "choice",
        "unknown": "exact",
    }.get(answer_type, "exact")
    gold_value = record.get("answer")
    if not isinstance(gold_value, str) or not normalized_string(gold_value):
        return _invalid(pred, method, "gold policy error: non-empty text required")
    gold = normalized_string(gold_value)
    return _comparison(
        "correct" if pred == gold else "incorrect",
        pred,
        method,
        "normalized string comparison",
    )


def _scalar_result(record: dict[str, Any], pred: str, answer_type: str) -> dict[str, str]:
    method = "rational" if answer_type == "fraction" else answer_type
    parser = _parse_fraction if answer_type == "fraction" else (
        _parse_decimal if answer_type == "decimal" else _parse_integer
    )
    gold_value = record.get("answer")
    if not isinstance(gold_value, str):
        return _invalid(pred, method, "gold policy error: answer must be text")
    gold = normalized_string(gold_value)
    gold_parsed = parser(gold)
    if gold_parsed is None:
        return _invalid(pred, method, "gold policy error: strict parse failed")
    if answer_type == "decimal" and not isinstance(gold_parsed, Decimal):
        return _invalid(pred, method, "gold policy error: non-finite decimal")
    predicted = parser(pred)
    if predicted is None:
        return _invalid(pred, method, f"strict {answer_type} parse failed")
    return _comparison("correct" if predicted == gold_parsed else "incorrect", pred, method, "exact arithmetic")


def _number_list_result(record: dict[str, Any], pred: str) -> dict[str, str]:
    gold_value = record.get("answer")
    if not isinstance(gold_value, str):
        return _invalid(pred, "list", "gold policy error: answer must be text")
    gold = normalized_string(gold_value)
    gold_parsed = _parse_number_list(gold)
    if gold_parsed is None:
        return _invalid(pred, "list", "gold policy error: strict number-list parse failed")
    predicted = _parse_number_list(pred)
    if predicted is None:
        return _invalid(pred, "list", "strict number-list parse failed")
    return _comparison(
        "correct" if predicted == gold_parsed else "incorrect",
        pred,
        "list",
        "ordered exact numeric elements",
    )


def _base_number_result(record: dict[str, Any], pred: str) -> dict[str, str]:
    radix = _base_radix_from_question(record.get("question"))
    if radix is None:
        return {"status": "unsupported", "method": "base_number", "normalized_prediction": pred, "reason": "no approved radix metadata"}
    gold_value = record.get("answer")
    if not isinstance(gold_value, str):
        return _invalid(pred, "base_number", "gold policy error: answer must be text")
    gold = normalized_string(gold_value)
    gold_parsed = _parse_base_integer(gold, radix)
    if gold_parsed is None:
        return _invalid(pred, "base_number", "gold policy error: invalid radix numeral")
    predicted = _parse_base_integer(pred, radix)
    if predicted is None:
        return _invalid(pred, "base_number", "invalid radix numeral")
    return _comparison("correct" if predicted == gold_parsed else "incorrect", pred, "base_number", f"exact base-{radix} integer arithmetic")


def _time_result(record: dict[str, Any], pred: str) -> dict[str, str]:
    gold_value = record.get("answer")
    if not isinstance(gold_value, str):
        return _invalid(pred, "time", "gold policy error: answer must be text")
    gold = normalized_string(gold_value)
    gold_parsed = _parse_time(gold)
    if gold_parsed is None:
        return _invalid(pred, "time", "gold policy error: invalid 12-hour time")
    predicted = _parse_time(pred)
    if predicted is None:
        return _invalid(pred, "time", "invalid 12-hour time")
    return _comparison("correct" if predicted == gold_parsed else "incorrect", pred, "time", "exact time-of-day comparison")


def verify_answer(record: dict[str, Any], prediction: str, *, finished: bool = True) -> dict[str, str]:
    """Purely classify one generated answer under the versioned answer policy."""
    answer_type = str(record.get("answer_type", "unknown"))
    if not isinstance(prediction, str):
        return _invalid("", "exact", "prediction must be text")
    pred = normalized_string(prediction)
    if not finished:
        return _invalid(pred, answer_type, "generation did not terminate with EOS")
    if not pred:
        return _invalid(pred, "exact", "empty prediction")
    if answer_type in {"symbol", "option_choice", "unknown"}:
        return _string_type_result(record, pred)
    if answer_type == "boolean":
        gold_value = record.get("answer")
        if not isinstance(gold_value, str):
            return _invalid(pred, "boolean", "gold policy error: answer must be text")
        gold = _strict_bool(normalized_string(gold_value))
        if gold is None:
            return _invalid(pred, "boolean", "gold policy error: unrecognized boolean spelling")
        predicted = _strict_bool(pred)
        if predicted is None:
            return _invalid(pred, "boolean", "unrecognized boolean spelling")
        return _comparison("correct" if predicted == gold else "incorrect", pred, "boolean", "explicit boolean table")
    if answer_type in {"integer", "decimal", "fraction"}:
        return _scalar_result(record, pred, answer_type)
    if answer_type == "number_list":
        return _number_list_result(record, pred)
    if answer_type == "base_number":
        return _base_number_result(record, pred)
    if answer_type == "time":
        return _time_result(record, pred)
    return {"status": "unsupported", "method": answer_type, "normalized_prediction": pred, "reason": "semantic rule not approved for pilot"}
