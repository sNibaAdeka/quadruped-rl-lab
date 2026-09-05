"""Append-only experiment event log used by manual motor tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: dict[str, Any]) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
