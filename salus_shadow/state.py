"""Per-session memory shared across hook invocations.

Each PreToolUse call is a fresh process, so anything a policy needs to know
about *earlier* calls (was a DB backup taken? did the agent create this file?)
lives in a small JSON file per session, guarded by flock because Claude Code
runs matching hooks for parallel tool calls concurrently.
"""

from __future__ import annotations

import fcntl
import fnmatch
import glob
import gzip
import hashlib
import json
import os
import posixpath
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def within(path: str, root: str) -> bool:
    """`path` is `root` or inside it, also when one side is spelled through a symlink (/var vs /private/var)."""

    def _in(p: str, r: str) -> bool:
        return p == r or p.startswith(r.rstrip("/") + "/")

    return _in(path, root) or _in(os.path.realpath(path), os.path.realpath(root))


def relative(path: str, root: str) -> str:
    return (
        posixpath.relpath(path, root)
        if path.startswith(root.rstrip("/") + "/") or path == root
        else posixpath.relpath(os.path.realpath(path), os.path.realpath(root))
    )


class SessionState:
    def __init__(self, data: dict) -> None:
        self.created: set[str] = set(data.get("created", []))
        self.backups: list[str] = list(data.get("backups", []))
        self.calls: int = int(data.get("calls", 0))

    def to_json(self) -> dict:
        return {"created": sorted(self.created), "backups": self.backups[-20:], "calls": self.calls}

    @classmethod
    @contextmanager
    def open(cls, state_dir: str, session_id: str) -> Iterator[SessionState]:
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        path = Path(state_dir) / (hashlib.sha1(session_id.encode()).hexdigest()[:16] + ".json")
        with open(path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            try:
                state = cls(json.loads(fh.read() or "{}"))
            except json.JSONDecodeError:
                state = cls({})
            yield state
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(state.to_json()))


class Filesystem:
    """What the policies may ask about paths.

    "Did this exist before the session?" means "is it tracked by git and not
    something the agent created this session". Live, git answers (falling back
    to the disk outside a git repo); in replay, `manifest` - the repo's file
    list at the task's base commit - stands in for `git ls-files`.
    """

    def __init__(self, repo_root: str, state: SessionState, manifest: set[str] | None = None) -> None:
        self.repo_root = posixpath.normpath(repo_root)
        self.state = state
        self.manifest = manifest

    def created_by_agent(self, path: str) -> bool:
        p = path
        while p not in ("/", ""):
            if p in self.state.created:
                return True
            p = posixpath.dirname(p)
        return False

    def preexisting(self, path: str) -> bool | None:
        if self.created_by_agent(path):
            return False
        in_repo = within(path, self.repo_root)
        if self.manifest is not None:
            return relative(path, self.repo_root) in self.manifest if in_repo else None
        if in_repo:
            tracked = self._git_tracked(path)
            if tracked is not None:
                return tracked
        return os.path.lexists(path)

    def _git_tracked(self, path: str) -> bool | None:
        try:
            out = subprocess.run(
                ["git", "-C", self.repo_root, "ls-files", "--", relative(path, self.repo_root)],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return bool(out.stdout.strip()) if out.returncode == 0 else None

    def expand(self, pattern: str) -> list[str]:
        """Paths an absolute glob pattern would match right now (disk, or manifest + agent-created files)."""
        if self.manifest is None:
            return glob.glob(pattern)
        parts = pattern.split("/")
        known = {posixpath.join(self.repo_root, rel) for rel in self.manifest} | self.state.created
        return sorted(
            p
            for p in known
            if len(p.split("/")) == len(parts) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(p.split("/"), parts))
        )

    @property
    def backup_seen(self) -> bool:
        return bool(self.state.backups)


def load_manifest(path: str | None) -> set[str] | None:
    if not path:
        return None
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:  # type: ignore[operator]
        return {line.rstrip("\n") for line in fh if line.strip()}
