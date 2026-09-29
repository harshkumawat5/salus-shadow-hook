"""Translate SWE-agent trajectories into Claude Code PreToolUse events.

SWE-agent action              -> Claude Code tool
-------------------------------------------------------------------------
bash command (any version)    -> Bash      {command}
str_replace_editor create     -> Write     {file_path, content}
str_replace_editor str_replace-> Edit      {file_path, old_string, new_string}
str_replace_editor insert     -> Edit      {file_path, old_string: "", new_string}
str_replace_editor undo_edit  -> Edit      {file_path}            (contents unknown)
0.x `create FILE`             -> Write     {file_path, content: ""}
0.x `edit N:M ... end_of_edit`-> Edit      {file_path: <open file>, old_string: "", new_string}
view/open/goto/scroll/search  -> Read/Grep (not matched by the hook, counted only)
submit / exit_*               -> dropped

`cwd` for step i is the working_dir SWE-agent recorded for that step (the state
the model saw before acting), which is what Claude Code would put in `cwd`.

Usage:  python replay/adapt.py      (reads .cache/trajs, writes data/calls/*.jsonl)
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "trajs"
OUT = ROOT / "data" / "calls"
READ_ONLY_0X = {"open", "goto", "scroll_up", "scroll_down", "search_dir", "search_file", "find_file"}
MAX_CONTENT = 4000  # file bodies are irrelevant to the policies; keep the committed data small


def _state(step: dict) -> dict:
    st = step.get("state") or {}
    if isinstance(st, str):
        try:
            st = json.loads(st)
        except json.JSONDecodeError:
            st = {}
    return st if isinstance(st, dict) else {}


def _abs(path: str, cwd: str) -> str:
    return path if path.startswith("/") else f"{cwd.rstrip('/')}/{path}"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CONTENT else text[:MAX_CONTENT] + "\n...[clipped by adapter]"


def editor_call(action: str, cwd: str) -> tuple[str, dict, str]:
    """Parse a SWE-agent 1.x `str_replace_editor ...` action string."""
    try:
        toks = shlex.split(action)
    except ValueError:  # model-written quoting (open-weights runs) can be unbalanced
        m = re.match(r"str_replace_editor\s+(\w+)\s+(\S+)", action)
        if not m:
            return "Bash", {"command": action}, "unparsed editor action -> Bash"
        cmd, path = m.groups()
        return (
            ("Read", {"file_path": path}, "view")
            if cmd == "view"
            else ("Edit", {"file_path": _abs(path, cwd), "old_string": "", "new_string": ""}, f"{cmd} (unparsed args)")
        )
    cmd, path = (toks[1:3] + ["", ""])[:2]
    opts: dict[str, str] = {}
    i = 3
    while i < len(toks):
        if toks[i].startswith("--") and i + 1 < len(toks):
            opts[toks[i][2:]] = toks[i + 1]
            i += 2
        else:
            i += 1
    path = _abs(path, cwd)
    if cmd == "view":
        return "Read", {"file_path": path}, "view"
    if cmd == "create":
        return "Write", {"file_path": path, "content": _clip(opts.get("file_text", ""))}, "create"
    if cmd == "str_replace":
        return (
            "Edit",
            {
                "file_path": path,
                "old_string": _clip(opts.get("old_str", "")),
                "new_string": _clip(opts.get("new_str", "")),
            },
            "str_replace",
        )
    if cmd == "insert":
        return "Edit", {"file_path": path, "old_string": "", "new_string": _clip(opts.get("new_str", ""))}, "insert"
    return "Edit", {"file_path": path, "old_string": "", "new_string": ""}, cmd or "editor"


def legacy_call(action: str, cwd: str, open_file: str | None) -> tuple[str, dict, str] | None:
    """Parse a SWE-agent 0.x ACI action."""
    head = action.split(None, 1)[0] if action.strip() else ""
    if head in READ_ONLY_0X:
        return ("Read" if head in ("open", "goto", "scroll_up", "scroll_down") else "Grep"), {"pattern": action}, head
    if head == "create":
        arg = action.split(None, 1)[1].strip() if len(action.split(None, 1)) > 1 else ""
        return "Write", {"file_path": _abs(arg, cwd), "content": ""}, "create"
    if head == "edit":
        body = action.split("\n", 1)[1] if "\n" in action else ""
        body = re.sub(r"\n?end_of_edit\s*$", "", body)
        return "Edit", {"file_path": open_file or "", "old_string": "", "new_string": _clip(body)}, "edit N:M"
    return None


def adapt(run: str, traj_path: Path) -> list[dict]:
    iid = traj_path.stem
    data = json.loads(traj_path.read_text())
    steps = data["trajectory"]
    first_wd = _state(steps[0]).get("working_dir") if steps else None
    legacy = not any(str(s.get("action", "")).startswith("str_replace_editor") for s in steps) and any(
        re.match(r"(edit \d+:\d+|open |create )", str(s.get("action", ""))) for s in steps
    )
    repo_root = first_wd if legacy and first_wd else "/testbed"
    cwd, open_file = repo_root, None
    out = []
    for i, step in enumerate(steps):
        st = _state(step)  # SWE-agent records the state the model saw *before* this action
        cwd = st.get("working_dir") or cwd
        if st.get("open_file") and st["open_file"] != "n/a":
            open_file = st["open_file"]
        action = str(step.get("action") or "").strip()
        head = action.split(None, 1)[0] if action else ""
        if not action or head == "submit" or head.startswith("exit"):
            tool, tool_input, origin = None, None, head or "empty"
        elif action.startswith("str_replace_editor"):
            tool, tool_input, origin = editor_call(action, cwd)
        elif legacy and (parsed := legacy_call(action, cwd, open_file)):
            tool, tool_input, origin = parsed
        else:
            tool, tool_input, origin = "Bash", {"command": action}, "bash"
        if tool:
            out.append(
                {
                    "meta": {
                        "run": run,
                        "instance": iid,
                        "step": i,
                        "origin": origin,
                        "repo_root": repo_root,
                        "sweagent_action": action[:600],
                    },
                    "event": {
                        "session_id": f"{run}/{iid}",
                        "transcript_path": f"swe-agent:{run}/{iid}.traj",
                        "cwd": cwd,
                        "permission_mode": "default",
                        "hook_event_name": "PreToolUse",
                        "tool_name": tool,
                        "tool_input": tool_input,
                        "tool_use_id": f"{run}/{iid}#{i}",
                    },
                }
            )
        if origin == "create" and tool_input:  # the ACI opens the file it just created
            open_file = tool_input["file_path"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for run_dir in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        for traj in sorted(run_dir.glob("*.traj")):
            calls = adapt(run_dir.name, traj)
            (OUT / f"{run_dir.name}__{traj.stem}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
            total += len(calls)
    print(f"wrote {total} events from {len(list(OUT.glob('*.jsonl')))} trajectories", file=sys.stderr)


if __name__ == "__main__":
    main()
