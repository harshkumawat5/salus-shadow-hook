"""The hook's process contract with Claude Code: shadow mode is invisible, enforce mode decides."""

import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "salus_shadow" / "hook.py"


def call(payload: object, tmp_path: Path, mode: str = "shadow") -> tuple[subprocess.CompletedProcess, list[dict]]:
    env = {
        "PATH": "/usr/bin:/bin",
        "SALUS_MODE": mode,
        "SALUS_LOG": str(tmp_path / "log.jsonl"),
        "SALUS_STATE_DIR": str(tmp_path / "state"),
        "SALUS_REPO_ROOT": "/work/repo",
    }
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    proc = subprocess.run([sys.executable, str(HOOK)], input=stdin, capture_output=True, text=True, env=env)
    log = tmp_path / "log.jsonl"
    return proc, [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


EVENT = {
    "session_id": "s",
    "cwd": "/work/repo",
    "hook_event_name": "PreToolUse",
    "tool_use_id": "t",
    "tool_name": "Bash",
    "tool_input": {"command": "git push --force"},
}


def test_shadow_mode_is_invisible(tmp_path: Path) -> None:
    proc, log = call(EVENT, tmp_path)
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""
    assert log[0]["decision"] == "would_deny" and log[0]["findings"][0]["policy"] == "force_push"


def test_enforce_mode_denies(tmp_path: Path) -> None:
    proc, _ = call(EVENT, tmp_path, mode="enforce")
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert (
        proc.returncode == 0 and out["permissionDecision"] == "deny" and "force_push" in out["permissionDecisionReason"]
    )


def test_enforce_mode_asks_for_medium(tmp_path: Path) -> None:
    event = {**EVENT, "tool_name": "Write", "tool_input": {"file_path": "/tmp/x", "content": ""}}
    proc, _ = call(event, tmp_path, mode="enforce")
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_clean_call_logs_allow_without_input(tmp_path: Path) -> None:
    proc, log = call({**EVENT, "tool_input": {"command": "ls"}}, tmp_path, mode="enforce")
    assert proc.stdout == "" and log[0]["decision"] == "allow" and "input_preview" not in log[0]


def test_fails_open_on_garbage(tmp_path: Path) -> None:
    proc, log = call("{not json", tmp_path)
    assert proc.returncode == 0 and proc.stdout == "" and "error" in log[0]


def test_secrets_redacted_in_preview(tmp_path: Path) -> None:
    _, log = call({**EVENT, "tool_input": {"command": "API_KEY=sk-live-123 git push -f"}}, tmp_path)
    assert "sk-live-123" not in log[0]["input_preview"]


def test_live_repo_through_symlink_and_git_provenance(tmp_path: Path) -> None:
    """Live mode: repo reached via a symlink (macOS /var -> /private/var) and git-tracked provenance."""
    real = tmp_path / "real"
    (real / "tests").mkdir(parents=True)
    (real / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
    git = ["git", "-C", str(real), "-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    (real / "tests" / "test_scratch.py").write_text("")
    link = tmp_path / "link"
    link.symlink_to(real)
    env = {
        "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        "CLAUDE_PROJECT_DIR": str(link),
        "SALUS_LOG": str(tmp_path / "log.jsonl"),
        "SALUS_STATE_DIR": str(tmp_path / "state"),
    }
    for name in ("test_a.py", "test_scratch.py"):
        event = {
            **EVENT,
            "cwd": str(link),
            "tool_name": "Edit",
            "tool_use_id": name,
            "tool_input": {"file_path": f"{link}/tests/{name}", "old_string": "a", "new_string": "b"},
        }
        subprocess.run([sys.executable, str(HOOK)], input=json.dumps(event), text=True, env=env, check=True)
    log = {r["tool_use_id"]: r for r in map(json.loads, (tmp_path / "log.jsonl").read_text().splitlines())}
    for name, tracked in (("test_a.py", True), ("test_scratch.py", False)):
        policies = {f["policy"]: f for f in log[name]["findings"]}
        assert set(policies) == {"test_file_edit"}  # in-repo despite the symlink
        assert policies["test_file_edit"]["tags"]["preexisting"] is tracked
