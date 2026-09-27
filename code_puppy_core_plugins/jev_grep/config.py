"""Settings for jev_grep. Set with `/set <key> <value>` or the environment."""

from __future__ import annotations

import os

from code_puppy.config import get_api_key, get_value

# TypeSafe's official name first, then the Jev-branded alias. Config lookup
# is case-insensitive, so `/set jev_api_key ...` matches JEV_API_KEY.
API_KEY_NAMES = ("TYPESAFE_API_KEY", "JEV_API_KEY")
API_KEY_NAME = API_KEY_NAMES[0]
DEFAULT_MODEL = "jev-latest"
DEFAULT_THRESHOLD = 0.5


def get_typesafe_api_key() -> str:
    """First key found: environment beats config, official name beats alias."""
    for name in API_KEY_NAMES:
        if key := os.environ.get(name):
            return key
    for name in API_KEY_NAMES:
        if key := get_api_key(name):
            return key
    return ""


def get_jev_model_name() -> str:
    """Pin a version (e.g. ``jev-1.13.0``) once a threshold is tuned against it."""
    return get_value("jev_grep_model") or DEFAULT_MODEL


def get_threshold() -> float:
    try:
        value = float(get_value("jev_grep_threshold") or DEFAULT_THRESHOLD)
    except ValueError:
        return DEFAULT_THRESHOLD
    return value if 0 <= value <= 1 else DEFAULT_THRESHOLD
