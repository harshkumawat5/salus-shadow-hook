"""Replay adapted trajectories through the real hook process, exactly as Claude Code would call it.

Each event is piped to `python3 salus_shadow/hook.py` on stdin, one process per tool call,
sequentially within a trajectory (so session state carries over), trajectories in parallel.
Only tools matched by examples/settings.json are sent; Read/Grep calls are counted, not hooked.

Every call must honour the shadow contract (exit 0, empty stdout/stderr); violations abort.

Usage:  python replay/replay.py      (writes results/shadow_log.jsonl, results/replay_stats.json)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / "salus_shadow" / "hook.py"
CALLS = ROOT / "data" / "calls"
RESULTS = ROOT / "results"
HOOKED = {"Bash", "Write", "Edit", "MultiEdit", "NotebookEdit"}
PKG_INSTALL = re.compile(r"\b(pip3?|python3? -m pip|apt(-get)?|conda|npm)\s+(install|i)\b")


def replay_trajectory(path: Path, log: Path, state_dir: str) -> Counter:
    stats: Counter = Counter()
    for line in path.read_text().splitlines():
        call = json.loads(line)
        meta, event = call["meta"], call["event"]
        stats["tool_calls"] += 1
        if event["tool_name"] not in HOOKED:
            continue
        stats["hooked_calls"] += 1
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "SALUS_MODE": "shadow",
            "SALUS_LOG": str(log),
            "SALUS_STATE_DIR": state_dir,
            "SALUS_REPO_ROOT": meta["repo_root"],
            "SALUS_HOME": "/root",  # SWE-agent containers run as root
            "SALUS_FS_MANIFEST": str(ROOT / "data" / "manifests" / f"{meta['instance']}.txt.gz"),
        }
        proc = subprocess.run(
            [sys.executable, str(HOOK)], input=json.dumps(event), capture_output=True, text=True, env=env, timeout=30
        )
        if proc.returncode != 0 or proc.stdout or proc.stderr:
            raise SystemExit(
                f"shadow contract violated on {event['tool_use_id']}: "
                f"rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
    return stats


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "shadow_log.jsonl"
    log.unlink(missing_ok=True)
    files = sorted(CALLS.glob("*.jsonl"))
    with tempfile.TemporaryDirectory() as state_dir, ThreadPoolExecutor(max_workers=8) as pool:
        per_file = dict(zip(files, pool.map(lambda p: replay_trajectory(p, log, state_dir), files)))

    by_run: dict[str, Counter] = {}
    for path, c in per_file.items():
        run = path.name.split("__", 1)[0]
        by_run.setdefault(run, Counter()).update(c + Counter(trajectories=1))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    errors = [r for r in records if "error" in r]
    latencies = sorted(r["latency_ms"] for r in records if "latency_ms" in r)
    installs = [
        (p.name, json.loads(line)["event"]["tool_input"]["command"])
        for p in files
        for line in p.read_text().splitlines()
        if json.loads(line)["event"]["tool_name"] == "Bash"
        and PKG_INSTALL.search(json.loads(line)["event"]["tool_input"]["command"])
    ]
    stats = {
        "trajectories": len(files),
        "uncovered": {"package_installs": len(installs), "trajectories": len({n for n, _ in installs})},
        "by_run": {k: dict(v) for k, v in sorted(by_run.items())},
        "hook_records": len(records),
        "hook_errors": len(errors),
        "policy_latency_ms": {
            "p50": latencies[len(latencies) // 2],
            "p99": latencies[int(len(latencies) * 0.99)],
            "max": latencies[-1],
        },
    }
    (RESULTS / "replay_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    if errors:
        print("hook errors:", errors[:5], file=sys.stderr)


if __name__ == "__main__":
    main()
