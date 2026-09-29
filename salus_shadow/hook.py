#!/usr/bin/env python3
"""Claude Code PreToolUse hook: evaluate five safety policies, log, never block.

Shadow mode (default) always exits 0 with empty stdout, which Claude Code
treats as "no decision": the normal permission flow runs unchanged. (Printing
`permissionDecision: "allow"` would be a bug - it skips the user's prompts.)
Every evaluated call is appended to a JSONL log; flagged calls carry the
finding(s) and a redacted preview of the input.

SALUS_MODE=enforce turns the same findings into real decisions: high-severity
findings deny, medium-severity findings ask the user.

Environment:
  SALUS_MODE          shadow (default) | enforce
  SALUS_LOG           JSONL log path          (default ~/.salus/shadow.jsonl)
  SALUS_STATE_DIR     per-session state dir   (default ~/.salus/state)
  SALUS_REPO_ROOT     override repo root      (default: git toplevel of $CLAUDE_PROJECT_DIR or cwd)
  SALUS_HOME          override $HOME for ~ expansion (replay: the agent's home, not ours)
  SALUS_FS_MANIFEST   replay only: file list standing in for the repo on disk
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salus_shadow.policies import Event, evaluate, side_effects  # noqa: E402
from salus_shadow.state import Filesystem, SessionState, load_manifest  # noqa: E402

HOOKED_TOOLS = {"Bash", "Write", "Edit", "MultiEdit", "NotebookEdit"}
_SECRET = re.compile(r"(?i)((?:api[_-]?key|token|secret|passw(?:or)?d|authorization)\S*?[=:]\s*)(\S+)")


def _salus_home() -> str:
    return os.path.expanduser("~/.salus")


def _repo_root(event: dict) -> str:
    if os.environ.get("SALUS_REPO_ROOT"):
        return os.environ["SALUS_REPO_ROOT"]
    base = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "-C", base, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=2
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return base


def _preview(tool_input: dict, limit: int = 400) -> str:
    text = json.dumps(tool_input, ensure_ascii=False)
    return _SECRET.sub(r"\1***", text)[:limit]


def _append(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:  # one write() per line; O_APPEND keeps concurrent lines intact
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(event: dict) -> tuple[dict, str | None]:
    """Evaluate one PreToolUse event. Returns (log record, stdout for Claude Code or None)."""
    t0 = time.perf_counter()
    mode = os.environ.get("SALUS_MODE", "shadow")
    tool_name = str(event.get("tool_name", ""))
    tool_input = event.get("tool_input") or {}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "mode": mode,
        "session_id": event.get("session_id"),
        "tool_use_id": event.get("tool_use_id"),
        "tool_name": tool_name,
        "decision": "allow",
        "findings": [],
        "input_sha256": hashlib.sha256(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:16],
    }
    if tool_name not in HOOKED_TOOLS:
        return record, None

    repo_root = _repo_root(event)
    home = os.environ.get("SALUS_HOME") or os.path.expanduser("~")
    state_dir = os.environ.get("SALUS_STATE_DIR") or os.path.join(_salus_home(), "state")
    manifest = load_manifest(os.environ.get("SALUS_FS_MANIFEST"))

    with SessionState.open(state_dir, str(event.get("session_id") or "unknown")) as state:
        ev = Event.build(
            tool_name, tool_input, event.get("cwd"), repo_root, home, Filesystem(repo_root, state, manifest)
        )
        findings = evaluate(ev)
        created, backups = side_effects(ev)
        state.created.update(created)
        state.backups.extend(backups)
        state.calls += 1

    stdout = None
    if findings:
        worst = "deny" if any(f.decision == "deny" for f in findings) else "ask"
        record["decision"] = worst if mode == "enforce" else f"would_{worst}"
        record["findings"] = [
            {
                "policy": f.policy,
                "severity": f.severity,
                "reason": f.reason,
                "evidence": f.evidence[:300],
                "tags": f.tags,
            }
            for f in findings
        ]
        record["cwd"] = event.get("cwd")
        record["repo_root"] = repo_root
        record["input_preview"] = _preview(tool_input)
        if mode == "enforce":
            stdout = json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": worst,
                        "permissionDecisionReason": "Salus: " + "; ".join(f"[{f.policy}] {f.reason}" for f in findings),
                    }
                }
            )
    record["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return record, stdout


def main() -> int:
    log_path = os.environ.get("SALUS_LOG") or os.path.join(_salus_home(), "shadow.jsonl")
    try:
        event = json.loads(sys.stdin.read() or "{}")
        record, stdout = run(event)
        _append(log_path, record)
        if stdout:
            sys.stdout.write(stdout)
    except Exception as exc:  # fail open: a broken guard must never break the agent
        try:
            _append(log_path, {"ts": datetime.now(timezone.utc).isoformat(), "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
