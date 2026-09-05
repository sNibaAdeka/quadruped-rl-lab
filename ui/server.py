"""Minimal live view: a real PPO trainer and a separate policy playback world."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from rl.policy_runner import LivePolicyRunner


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "ui" / "static"
RUNS = ROOT / "runs"
runner = LivePolicyRunner(ROOT)
runner_lock = asyncio.Lock()


def trainer_is_alive() -> bool:
    pid_file = RUNS / "training.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8"))
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def start_training_if_needed() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    status_path = RUNS / "training_state.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
    except (FileNotFoundError, json.JSONDecodeError):
        status = None
    if trainer_is_alive():
        return
    with (RUNS / "training.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "rl.train", "--timesteps", "900000"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (RUNS / "training.pid").write_text(str(process.pid), encoding="utf-8")


@asynccontextmanager
async def lifecycle(_: FastAPI):
    if os.getenv("QRL_NO_AUTOTRAIN") != "1":
        start_training_if_needed()
    running = True

    async def simulate_policy() -> None:
        while running:
            started = time.perf_counter()
            async with runner_lock:
                runner.step()
            await asyncio.sleep(max(.001, runner.env.world.control_dt - (time.perf_counter() - started)))

    task = asyncio.create_task(simulate_policy())
    try:
        yield
    finally:
        running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Quadruped RL Lab · Live Training", version="0.2.0", lifespan=lifecycle)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/state")
async def state() -> dict:
    async with runner_lock:
        return runner.snapshot()
