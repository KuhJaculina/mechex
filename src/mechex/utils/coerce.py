"""Coercion helpers for values that cross a dict or a report boundary.

The report and table structures mix numbers and labels in one mapping, so a reader of the
field has to say which it expects. These helpers check rather than coerce silently: a
mistyped field should raise where the value's origin is still visible.
"""

from __future__ import annotations


def as_float(value: object, context: str = "value") -> float:
    """Read a numeric field, rejecting booleans and text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} is not numeric")
    return float(value)


def as_int(value: object, context: str = "value") -> int:
    """Read an integer field, rejecting booleans and floats with a fractional part."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} is not numeric")
    if float(value) != int(value):
        raise TypeError(f"{context} is not an integer")
    return int(value)


def as_text(value: object, context: str = "value") -> str:
    """Read a text field."""
    if not isinstance(value, str):
        raise TypeError(f"{context} is not text")
    return value


def as_mapping(value: object, context: str = "value") -> dict[str, object]:
    """Read a nested mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{context} is not a mapping")
    return {str(key): item for key, item in value.items()}


def as_optional_float(value: object, context: str = "value") -> float | None:
    """Read a field that may be absent."""
    if value is None:
        return None
    return as_float(value, context)
