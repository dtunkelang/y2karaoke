"""Shared trace/serialization helpers for Whisper assignment modules."""

from typing import Any


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except (TypeError, ValueError):
            return value
    return value
