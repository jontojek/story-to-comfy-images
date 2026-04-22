#!/usr/bin/env python3
"""Shared settings helpers for story-to-comfy pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_FILE = Path(".story_to_comfy_settings.json")


def get_default_settings_path() -> Path:
    return DEFAULT_SETTINGS_FILE


def load_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_settings(path: Path, updates: dict[str, Any]) -> None:
    current = load_settings(path)
    current.update(updates)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
