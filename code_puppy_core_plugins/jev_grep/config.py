"""Settings for jev_grep. Set with `/set <key> <value>` or the environment."""

from __future__ import annotations

import os

from code_puppy.config import get_api_key, get_value

API_KEY_NAME = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-latest"
DEFAULT_THRESHOLD = 0.5


def get_typesafe_api_key() -> str:
    return os.environ.get(API_KEY_NAME) or get_api_key(API_KEY_NAME)


def get_jev_model_name() -> str:
    """Pin a version (e.g. ``jev-1.13.0``) once a threshold is tuned against it."""
    return get_value("jev_grep_model") or DEFAULT_MODEL


def get_threshold() -> float:
    try:
        value = float(get_value("jev_grep_threshold") or DEFAULT_THRESHOLD)
    except ValueError:
        return DEFAULT_THRESHOLD
    return value if 0 <= value <= 1 else DEFAULT_THRESHOLD
