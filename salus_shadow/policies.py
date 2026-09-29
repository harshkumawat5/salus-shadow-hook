"""The five policies, as pure functions of one PreToolUse event.

Each policy returns zero or one Finding per tool call. Findings carry `tags`:
context the policy did *not* use to decide, recorded so shadow-mode triage can
measure which refinements would cut false positives (e.g. "was this path
created by the agent earlier in the session?").
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass, field
from typing import Protocol

from .shell import WRITE_REDIRECTS, Command, Word, parse, resolve
from .state import relative, within


class FsView(Protocol):
    def preexisting(self, path: str) -> bool | None: ...
    def created_by_agent(self, path: str) -> bool: ...
    def expand(self, pattern: str) -> list[str]: ...
    @property
    def backup_seen(self) -> bool: ...


@dataclass
class Finding:
    policy: str
    severity: str  # "high" -> enforce mode would deny; "medium" -> would ask the user
    reason: str
    evidence: str
    tags: dict = field(default_factory=dict)

    @property
    def decision(self) -> str:
        return "deny" if self.severity == "high" else "ask"


@dataclass
class Event:
    tool_name: str
    tool_input: dict
    cwd: str | None
    repo_root: str
    home: str | None
    fs: FsView
    commands: list[Command] = field(default_factory=list)

    @classmethod
    def build(cls, tool_name: str, tool_input: dict, cwd: str | None, repo_root: str, home: str | None, fs: FsView):
        repo_root = posixpath.normpath(repo_root)
        cmds = parse(str(tool_input.get("command", "")), cwd, home) if tool_name == "Bash" else []
        return cls(tool_name, tool_input, cwd, repo_root, home, fs, cmds)

    def in_repo(self, path: str) -> bool:
        return within(path, self.repo_root)

    def rel(self, path: str) -> str:
        return relative(path, self.repo_root)


# --------------------------------------------------------------------------- helpers

FILE_TOOLS = {"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
TMP_DIRS = ("/tmp", "/var/tmp", "/private/tmp", "/private/var/folders", "/var/folders", "/dev/shm")
PSEUDO_FILES = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/zero")
SYSTEM_DIRS = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt", "/boot", "/System", "/Library")
REGENERABLE = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "*.egg-info",
    ".eggs",
    ".coverage",
    "htmlcov",
    ".hypothesis",
    "node_modules",
)


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == r or path.startswith(r + "/") for r in roots)


def _positionals(args: list[Word], with_value: set[str] = frozenset()) -> list[Word]:  # type: ignore[assignment]
    out, i, end_of_opts = [], 0, False
    while i < len(args):
        t = args[i].text
        if end_of_opts or not t.startswith("-") or t == "-":
            out.append(args[i])
        elif t == "--":
            end_of_opts = True
        elif t in with_value:
            i += 1
        i += 1
    return out


def _short_flags(args: list[Word]) -> set[str]:
    letters: set[str] = set()
    for w in args:
        t = w.text
        if t == "--":
            break
        if t.startswith("-") and not t.startswith("--"):
            letters.update(t[1:])
    return letters


def _long_flags(args: list[Word]) -> set[str]:
    return {w.text.split("=", 1)[0] for w in args if w.text.startswith("--")}


_GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"}


def git_subcommand(cmd: Command) -> tuple[str | None, list[Word]]:
    args = cmd.args
    i = 0
    while i < len(args) and args[i].text.startswith("-"):
        i += 2 if args[i].text in _GIT_VALUE_OPTS else 1
    return (args[i].text, args[i + 1 :]) if i < len(args) else (None, [])


@dataclass
class Target:
    verb: str
    kind: str  # "write" | "delete"
    word: Word
    path: str | None  # resolved absolute path, None if not statically known


def bash_targets(cmd: Command, home: str | None) -> list[Target]:
    """Filesystem paths a simple command writes to or deletes (best effort)."""
    raw: list[tuple[str, str, Word]] = []
    for op, w in cmd.redirects:
        if op in WRITE_REDIRECTS:
            raw.append(("redirect " + op, "write", w))
    name, args = cmd.name, cmd.args
    if name == "tee":
        raw += [("tee", "write", w) for w in _positionals(args)]
    elif name in ("cp", "mv", "install", "rsync", "ln"):
        longs = [w.text for w in args if w.text.startswith("--target-directory=")]
        tflag = next((args[i + 1] for i, w in enumerate(args[:-1]) if w.text == "-t"), None)
        pos = _positionals(args, {"-t", "-S", "-m", "-o", "-g", "--suffix", "-e"})
        if tflag is not None:
            dest, sources = tflag, pos
        elif longs:
            dest, sources = Word(longs[0].split("=", 1)[1]), pos
        elif len(pos) >= 2:
            dest, sources = pos[-1], pos[:-1]
        else:
            dest, sources = None, []
        if dest is not None:
            raw.append((name, "write", dest))
        if name == "mv":
            raw += [("mv (source)", "delete", w) for w in sources]
    elif name in ("touch", "mkdir", "truncate"):
        raw += [(name, "write", w) for w in _positionals(args, {"-m", "-s", "-d", "-t", "-r", "--size", "--mode"})]
    elif name in ("sed", "perl") and ("i" in _short_flags(args) or "--in-place" in _long_flags(args)):
        pos = _positionals(args, {"-e", "-f", "--expression", "--file", "-l"})
        has_script_flag = any(w.text in ("-e", "-f") or w.text.startswith("--expression") for w in args) or (
            name == "perl" and any("e" in w.text[1:] for w in args if w.text.startswith("-") and len(w.text) > 1)
        )
        raw += [(f"{name} -i", "write", w) for w in (pos if has_script_flag else pos[1:])]
    elif name == "dd":
        raw += [("dd", "write", Word(w.text[3:], w.dynamic, w.glob)) for w in args if w.text.startswith("of=")]
    elif name in ("curl", "wget"):
        flags = ("-o", "--output") if name == "curl" else ("-O", "--output-document")
        raw += [(name, "write", args[i + 1]) for i, w in enumerate(args[:-1]) if w.text in flags]
    elif name in ("rm", "unlink", "rmdir", "shred"):
        raw += [(name, "delete", w) for w in _positionals(args)]
    elif name == "git":
        sub, rest = git_subcommand(cmd)
        if sub == "config" and any(w.text in ("--global", "--system") for w in rest):
            target = "/etc/gitconfig" if any(w.text == "--system" for w in rest) else f"{home or '~'}/.gitconfig"
            raw.append(("git config", "write", Word(target)))
    return [Target(v, k, w, resolve(w, cmd.cwd, home)) for v, k, w in raw]


# --------------------------------------------------------------------------- 1. force-push


def force_push(ev: Event) -> Finding | None:
    for c in ev.commands:
        if c.name != "git":
            continue
        sub, rest = git_subcommand(c)
        if sub != "push":
            continue
        why = []
        longs = _long_flags(rest)
        if "--force" in longs or "f" in _short_flags([w for w in rest if not w.text.startswith("-o")]):
            why.append("--force")
        if "--force-with-lease" in longs:
            why.append("--force-with-lease")
        if "--mirror" in longs:
            why.append("--mirror")
        pos = _positionals(rest, {"-o", "--push-option", "--repo", "--receive-pack", "--exec"})
        if any(w.text.startswith("+") for w in pos[1:]):
            why.append("+refspec")
        if why:
            branch = next((w.text for w in pos[1:]), None)
            return Finding(
                "force_push",
                "high",
                "git push rewrites remote history (" + ", ".join(why) + ")",
                " ".join(c.argv),
                {"branch": branch, "lease_only": why == ["--force-with-lease"]},
            )
    return None


# --------------------------------------------------------------------------- 2. destructive rm


def _dangerous_path(path: str | None, text: str, ev: Event) -> str | None:
    if text in (".", "..", "*", "/*", "~", "./*", "../*"):
        return f"'{text}'"
    if path is None:
        return None
    if path == "/" or path == ev.home:
        return "root or home directory"
    if path == ev.repo_root or ev.repo_root.startswith(path.rstrip("/") + "/"):
        return "the repository itself or one of its parents"
    if path in SYSTEM_DIRS or path in ("/home", "/root", "/Users", "/var"):
        return "a system directory"
    if ".git" in path.split("/"):
        return "git metadata"
    return None


def _regenerable(text: str) -> bool:
    base = posixpath.basename(text.rstrip("/"))
    return any(fnmatch.fnmatch(base, pat) for pat in REGENERABLE)


def destructive_rm(ev: Event) -> Finding | None:
    for c in ev.commands:
        why: list[str] = []
        targets: list[Target] = []
        if c.name in ("rm", "shred"):
            targets = [t for t in bash_targets(c, ev.home) if t.kind == "delete"]
            if {"r", "R"} & _short_flags(c.args) or "--recursive" in _long_flags(c.args):
                why.append("recursive")
            if c.name == "shred":
                why.append("shred (unrecoverable)")
            if any(t.word.glob for t in targets):
                why.append("wildcard")
            if "xargs" in c.via:
                why.append("targets piped in via xargs")
            for t in targets:
                danger = _dangerous_path(t.path, t.word.text, ev)
                if danger:
                    why.append(f"targets {danger}")
        elif c.name == "find" and (
            "-delete" in c.argv
            or any(a in ("-exec", "-execdir", "-ok") and b in ("rm", "shred") for a, b in zip(c.argv, c.argv[1:]))
        ):
            why.append("find ... -delete/-exec rm")
            roots = [w for w in c.args if not w.text.startswith("-")][:1]
            targets = [Target("find", "delete", w, resolve(w, c.cwd, ev.home)) for w in roots]
        elif c.name == "git" and git_subcommand(c)[0] == "clean":
            rest = git_subcommand(c)[1]
            if "f" in _short_flags(rest) or "--force" in _long_flags(rest):
                why.append("git clean -f deletes untracked files")
        if why:
            paths = [t.path for t in targets if t.path]
            matched = [m for t in targets if t.path for m in (ev.fs.expand(t.path) if t.word.glob else [t.path])]
            preexisting = [m for m in matched if ev.fs.preexisting(m)]
            return Finding(
                "destructive_rm",
                "high",
                "; ".join(dict.fromkeys(why)),
                " ".join(c.argv),
                {
                    "targets": paths or [t.word.text for t in targets],
                    "preexisting_hits": [ev.rel(m) if ev.in_repo(m) else m for m in preexisting][:10],
                    "all_created_by_agent": bool(matched) and all(ev.fs.created_by_agent(m) for m in matched),
                    "all_regenerable": bool(targets) and all(_regenerable(t.word.text) for t in targets),
                    "outside_repo": any(not ev.in_repo(p) for p in paths),
                    "under_tmp": bool(paths) and all(_under(p, TMP_DIRS) for p in paths),
                },
            )
    return None


# --------------------------------------------------------------------------- 3. migrations without a backup

_SQL_DESTRUCTIVE = re.compile(r"\b(drop\s+(table|database|schema)|truncate\s+table|alter\s+table)\b", re.I)
_PY_VALUE_OPTS = {"-W", "-X", "-Q"}


def _python_target(c: Command) -> tuple[str | None, list[str]]:
    """For `python [opts] (-m mod | script | -c code) args` return (module/script/-c, rest)."""
    argv, i = c.argv, 1
    while i < len(argv) and argv[i].startswith("-"):
        if argv[i] in ("-m", "-c"):
            return (
                (argv[i + 1] if argv[i] == "-m" else "-c", argv[i + 1 :] if argv[i] == "-c" else argv[i + 2 :])
                if i + 1 < len(argv)
                else (None, [])
            )
        i += 2 if argv[i] in _PY_VALUE_OPTS else 1
    return (argv[i], argv[i + 1 :]) if i < len(argv) else (None, [])


def _first_positional(args: list[str]) -> str | None:
    return next((a for a in args if not a.startswith("-")), None)


def migration_kind(c: Command) -> str | None:
    name, argv = c.name, c.argv
    if name in ("npx", "pnpx", "bunx") or (name in ("pnpm", "yarn", "bun") and len(argv) > 1 and argv[1] == "exec"):
        inner = argv[2:] if len(argv) > 1 and argv[1] == "exec" else argv[1:]
        inner = [a for a in inner if not a.startswith("-")]
        return migration_kind(Command([Word(a) for a in inner], cwd=c.cwd)) if inner else None
    if name in ("npm", "pnpm", "yarn", "bun", "make", "just"):
        script = _first_positional(argv[2:] if len(argv) > 1 and argv[1] in ("run", "run-script") else argv[1:])
        return f"{name} {script}" if script and "migrat" in script.lower() else None
    if name.startswith("python") or name in ("manage.py", "django-admin", "django-admin.py"):
        mod, rest = _python_target(c) if name.startswith("python") else (name, argv[1:])
        if mod == "-c":
            return (
                "django call_command('migrate')"
                if re.search(r"call_command\(\s*['\"]migrate", rest[0] if rest else "")
                else None
            )
        if mod and (mod.endswith("manage.py") or mod in ("django", "django-admin", "django-admin.py")):
            if _first_positional(rest) == "migrate" and not {"--plan", "--check"} & set(rest):
                return "django migrate"
            return None
        if mod in ("alembic", "flask"):
            c = Command([Word(mod)] + [Word(a) for a in rest], cwd=c.cwd)
            name, argv = mod, c.argv
        else:
            return None
    sub = [a for a in argv[1:] if not a.startswith("-")]
    first, second = (sub + [None, None])[:2]
    table: dict[str, bool] = {
        "alembic": first in ("upgrade", "downgrade", "stamp") and "--sql" not in argv,
        "flask": first == "db" and second in ("upgrade", "downgrade", "stamp"),
        "rails": bool(first)
        and first in ("db:migrate", "db:rollback", "db:reset", "db:drop", "db:schema:load", "db:setup"),
        "rake": bool(first)
        and first in ("db:migrate", "db:rollback", "db:reset", "db:drop", "db:schema:load", "db:setup"),
        "prisma": (first == "migrate" and second in ("deploy", "reset", "dev")) or (first == "db" and second == "push"),
        "knex": bool(first) and first in ("migrate:latest", "migrate:up", "migrate:down", "migrate:rollback"),
        "sequelize": bool(first) and first.startswith("db:migrate"),
        "sequelize-cli": bool(first) and first.startswith("db:migrate"),
        "typeorm": first in ("migration:run", "migration:revert", "schema:sync", "schema:drop"),
        "drizzle-kit": first in ("push", "migrate"),
        "flyway": first in ("migrate", "clean", "undo"),
        "liquibase": first in ("update", "rollback", "dropAll"),
        "goose": first in ("up", "down", "reset", "redo", "up-to", "down-to") or second in ("up", "down", "reset"),
        "migrate": any(a in ("up", "down", "drop", "force") for a in sub),
        "dbmate": first in ("up", "migrate", "rollback", "down", "drop"),
        "diesel": (first == "migration" and second in ("run", "revert", "redo"))
        or (first == "database" and second == "reset"),
        "sqlx": (first == "migrate" and second in ("run", "revert"))
        or (first == "database" and second in ("reset", "drop")),
        "supabase": first == "db" and second in ("push", "reset"),
    }
    if table.get(name):
        return f"{name} {' '.join(s for s in (first, second) if s)}"
    if name in ("psql", "mysql", "mariadb", "sqlite3"):
        sql = " ".join(argv[1:]) + "\n" + c.heredoc
        if _SQL_DESTRUCTIVE.search(sql):
            return f"{name} DDL"
        files = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("-f", "--file")]
        files += [w.text for op, w in c.redirects if op == "<"]
        if any("migrat" in f.lower() for f in files):
            return f"{name} migration file"
    return None


def is_backup(c: Command) -> bool:
    name, argv = c.name, c.argv
    if name in ("pg_dump", "pg_dumpall", "pg_basebackup", "mysqldump", "mariadb-dump", "mongodump"):
        return True
    if name == "sqlite3":
        return any(k in " ".join(argv) + c.heredoc for k in (".backup", ".dump", ".clone", "VACUUM INTO"))
    if name in ("cp", "rsync", "tar", "zip", "gzip"):
        return any(re.search(r"\.(sqlite3?|db)$", a) for a in argv[1:])
    if name.startswith("python") and "dumpdata" in argv:
        return True
    joined = " ".join(argv)
    return any(
        k in joined
        for k in (
            "pg:backups:capture",
            "create-db-snapshot",
            "create-db-cluster-snapshot",
            "sql backups create",
            "sql export",
        )
    )


def migration_without_backup(ev: Event) -> Finding | None:
    backed_up = ev.fs.backup_seen
    for c in ev.commands:
        if is_backup(c):
            backed_up = True
            continue
        kind = migration_kind(c)
        if kind and not backed_up:
            text = " ".join(c.argv)
            manage = next((resolve(a, c.cwd, ev.home) for a in c.argv if a.endswith("manage.py")), None)
            project = manage or c.cwd
            return Finding(
                "migration_without_backup",
                "high",
                f"{kind} with no database backup earlier in this session",
                text,
                {
                    "sqlite": "sqlite" in (text + (ev.cwd or "")).lower(),
                    "cwd": c.cwd,
                    "project_preexisting": ev.fs.preexisting(project) if project else None,
                    "cwd_under_tmp": bool(c.cwd) and _under(c.cwd or "", TMP_DIRS),
                },
            )
    return None


# --------------------------------------------------------------------------- 4. editing test files

TEST_DIRS = {"test", "tests", "testing", "__tests__", "spec", "specs", "e2e"}
TEST_FILE = re.compile(
    r"^(test_.*\.py|.*_tests?\.py|tests?\.py|conftest\.py|.*_test\.go|.*\.(test|spec)\.[cm]?[jt]sx?"
    r"|.*Tests?\.(java|kt|cs|swift)|.*_spec\.rb|.*_test\.rb|test_.*\.rb)$"
)


def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    return any(d.lower() in TEST_DIRS for d in parts[:-1]) or bool(TEST_FILE.match(parts[-1]))


def _edit_shape(tool_input: dict) -> dict:
    edits = tool_input.get("edits") or [tool_input]
    olds = [str(e.get("old_string", "")) for e in edits]
    news = [str(e.get("new_string", "")) for e in edits]
    removed_asserts = [
        line.strip()
        for o, n in zip(olds, news)
        for line in o.splitlines()
        if "assert" in line and line.strip() not in n
    ]
    kept = all(
        o.strip()
        and (o in n or {ln.strip() for ln in o.splitlines() if ln.strip()} <= {ln.strip() for ln in n.splitlines()})
        for o, n in zip(olds, news)
    )
    return {"additive": kept, "removes_assert": bool(removed_asserts)}


def test_file_edit(ev: Event) -> Finding | None:
    hits: list[tuple[str, str, str]] = []  # (verb, path, kind)
    if ev.tool_name in FILE_TOOLS:
        p = resolve(str(ev.tool_input.get(FILE_TOOLS[ev.tool_name], "")), ev.cwd, ev.home)
        if p:
            hits.append((ev.tool_name, p, "write"))
    for c in ev.commands:
        hits += [(t.verb, t.path, t.kind) for t in bash_targets(c, ev.home) if t.path]
    hits = [(v, p, k) for v, p, k in hits if ev.in_repo(p) and p != ev.repo_root and is_test_path(ev.rel(p))]
    if not hits:
        return None
    verb, path, kind = hits[0]
    concrete = [m for _, p, _ in hits for m in (ev.fs.expand(p) if any(ch in p for ch in "*?[") else [p])]
    tags = {
        "paths": sorted({ev.rel(p) for _, p, _ in hits}),
        "preexisting": any(ev.fs.preexisting(p) for p in concrete),
        "created_by_agent": bool(concrete) and all(ev.fs.created_by_agent(p) for p in concrete),
        "deletes": any(k == "delete" for _, _, k in hits),
    }
    if ev.tool_name in ("Edit", "MultiEdit"):
        tags.update(_edit_shape(ev.tool_input))
    action = "deletes" if kind == "delete" else "modifies"
    return Finding("test_file_edit", "medium", f"{verb} {action} test file {ev.rel(path)}", f"{verb} {path}", tags)


# --------------------------------------------------------------------------- 5. writes outside the repo


def _sensitive(path: str, home: str | None) -> bool:
    if _under(path, SYSTEM_DIRS) or "/site-packages/" in path or "/dist-packages/" in path:
        return True
    return bool(home) and path.startswith(f"{home}/.")


def write_outside_repo(ev: Event) -> Finding | None:
    hits: list[tuple[str, str, str]] = []  # (verb, path, kind)
    if ev.tool_name in FILE_TOOLS:
        p = resolve(str(ev.tool_input.get(FILE_TOOLS[ev.tool_name], "")), ev.cwd, ev.home)
        if p:
            hits.append((ev.tool_name, p, "write"))
    for c in ev.commands:
        hits += [(t.verb, t.path, t.kind) for t in bash_targets(c, ev.home) if t.path]
    hits = [
        h
        for h in hits
        if not ev.in_repo(h[1]) and not _under(h[1], PSEUDO_FILES) and not h[1].startswith(("/dev/fd/", "/proc/self/"))
    ]
    if not hits:
        return None
    sensitive = [h for h in hits if _sensitive(h[1], ev.home)]
    verb, path, kind = (sensitive or hits)[0]
    return Finding(
        "write_outside_repo",
        "high" if sensitive else "medium",
        f"{verb} {'deletes' if kind == 'delete' else 'writes'} {'sensitive path ' if sensitive else ''}{path} "
        f"outside {ev.repo_root}",
        f"{verb} {path}",
        {
            "paths": sorted({h[1] for h in hits}),
            "sensitive": bool(sensitive),
            "under_tmp": all(_under(h[1], TMP_DIRS) for h in hits),
        },
    )


POLICIES = (force_push, destructive_rm, migration_without_backup, test_file_edit, write_outside_repo)


def evaluate(ev: Event) -> list[Finding]:
    return [f for policy in POLICIES if (f := policy(ev)) is not None]


def side_effects(ev: Event) -> tuple[list[str], list[str]]:
    """(paths this call will create, backup commands it runs) - fed into session state."""
    created: list[str] = []
    if ev.tool_name in ("Write", "NotebookEdit"):
        p = resolve(str(ev.tool_input.get(FILE_TOOLS[ev.tool_name], "")), ev.cwd, ev.home)
        if p and not ev.fs.preexisting(p):
            created.append(p)
    for c in ev.commands:
        for t in bash_targets(c, ev.home):
            if t.kind == "write" and t.path and not ev.fs.preexisting(t.path) and not _under(t.path, PSEUDO_FILES):
                created.append(t.path)
    backups = [" ".join(c.argv) for c in ev.commands if is_backup(c)]
    return created, backups
