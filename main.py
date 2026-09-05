"""Local entry point for the interactive motor test laboratory."""
from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("ui.server:app", host="127.0.0.1", port=8020, reload=False)
