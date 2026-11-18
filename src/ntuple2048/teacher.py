"""Bridge to the WASM expert of the ``2048EndgameTablebase`` web demo.

The original project ships only a browser build (``ai_core.js`` + ``ai_core.wasm``),
so the only faithful way to query it is to run it.  ``ai_core.wasm`` is already a
statically compiled C++ binary (the JS around it is just embind glue), hence the
"rewrite in a static language" is a no-op: the bottleneck is the search depth,
not the host language.  What *does* cost time is the Python <-> JS boundary, so
this module keeps that boundary coarse-grained:

* every worker runs a whole self-play game inside one Node process,
* the trajectory is written as a flat binary file (17 bytes per state),
* Python parses it with numpy (``np.frombuffer``, zero copies).

Trajectory file layout (little endian)::

    magic  b"T2048V1\\0"   8 bytes
    u32    n_states
    u32    depth
    u64    seed
    per state: 16 x u8 cell exponents + 1 x u8 action (0..3 = L/R/U/D)
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

MAGIC = b"T2048V1\x00"
HEADER = 24
RECORD = 17
DEFAULT_WEB_DIR = Path(__file__).resolve().parents[2] / "2048-web"


def find_node() -> str:
    """Locate a Node.js binary (prefers the pip-installed one)."""
    try:
        import nodejs  # type: ignore

        candidate = Path(nodejs.__file__).parent / "bin" / "node"
        if candidate.exists():
            return str(candidate)
    except Exception:  # pragma: no cover - optional dependency
        pass
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError("Node.js not found. Install it, or `pip install nodejs-bin`.")


def read_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse one trajectory file into ``(boards, actions)``."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size < HEADER or raw[:8].tobytes() != MAGIC:
        raise ValueError(f"{path}: not a T2048V1 trajectory")
    n = int(struct.unpack("<I", raw[8:12].tobytes())[0])
    body = raw[HEADER : HEADER + n * RECORD]
    if body.size != n * RECORD:
        raise ValueError(f"{path}: truncated trajectory")
    records = body.reshape(n, RECORD)
    return records[:, :16].copy(), records[:, 16].copy()


class Teacher:
    """Runs the WASM expert in parallel Node processes to mint trajectories."""

    def __init__(
        self,
        web_dir: str | Path | None = None,
        node: str | None = None,
        speed: float = 1.0,
    ) -> None:
        self.web_dir = Path(web_dir).resolve() if web_dir else DEFAULT_WEB_DIR
        self.script = self.web_dir / "teacher_server.mjs"
        if not self.script.exists():
            raise FileNotFoundError(f"teacher_server.mjs not found under {self.web_dir}")
        self.node = node or find_node()
        # maps to the demo's speed slider: time_limit_ratio = 10**((100-slider)/100)
        self.speed = float(speed)

    def _run(self, games: int, seed: int, out_path: Path) -> dict:
        cmd = [
            self.node,
            str(self.script),
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--speed",
            str(self.speed),
            "--out",
            str(out_path),
        ]
        proc = subprocess.run(cmd, cwd=str(self.web_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"teacher failed (seed={seed}): {proc.stderr[-800:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def generate(
        self,
        n_games: int,
        out_dir: str | Path,
        workers: int = 8,
        seed: int = 0,
        keep_files: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Self-play ``n_games`` games in parallel and return all (state, action) pairs."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        workers = max(1, min(workers, n_games))
        base, extra = divmod(n_games, workers)

        jobs: list[tuple[int, Path]] = []
        for w in range(workers):
            games = base + (1 if w < extra else 0)
            if games == 0:
                continue
            jobs.append((games, out_dir / f"traj_{w:04d}.bin"))
        return self._generate_impl(jobs, seed, keep_files)

    def _generate_impl(
        self, jobs: list[tuple[int, Path]], seed: int, keep_files: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        def work(item: tuple[int, int, Path]) -> Path:
            index, games, path = item
            self._run(games, seed + index * 100_003, path)
            return path

        items = [(i, games, path) for i, (games, path) in enumerate(jobs)]
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            paths = list(pool.map(work, items))

        board_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        for path in paths:
            boards, actions = read_trajectory(path)
            board_chunks.append(boards)
            action_chunks.append(actions)
            if not keep_files:
                path.unlink(missing_ok=True)
        return np.concatenate(board_chunks), np.concatenate(action_chunks)


def merge_trajectories(paths: Iterable[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    boards, actions = [], []
    for path in paths:
        b, a = read_trajectory(path)
        boards.append(b)
        actions.append(a)
    return np.concatenate(boards), np.concatenate(actions)
