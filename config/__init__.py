"""YAML configuration loader kept separate from simulation code."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict[str, Any]:
    with (ROOT / f"{name}.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_all() -> dict[str, dict[str, Any]]:
    return {name: load(name) for name in ("robot", "motors", "sensors", "physics", "terrain", "rewards", "training", "randomization")}
