import pytest

from salus_shadow.policies import Event, evaluate, side_effects
from salus_shadow.state import Filesystem, SessionState

REPO = "/work/repo"
HOME = "/home/dev"
MANIFEST = {"setup.py", "src", "src/app.py", "tests", "tests/test_app.py", "django/test/utils.py", "db.sqlite3"}


def run(tool: str, tool_input: dict, cwd: str = REPO, state: SessionState | None = None) -> dict[str, object]:
    state = state or SessionState({})
    ev = Event.build(tool, tool_input, cwd, REPO, HOME, Filesystem(REPO, state, MANIFEST))
    findings = {f.policy: f for f in evaluate(ev)}
    created, backups = side_effects(ev)
    state.created.update(created)
    state.backups.extend(backups)
    return findings


def bash(cmd: str, **kw: object) -> set[str]:
    return set(run("Bash", {"command": cmd}, **kw))  # type: ignore[arg-type]


# --- force push --------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "git push --force origin main",
        "git push -f",
        "git push -uf origin feat",
        "git -C sub push --force-with-lease",
        "git push origin +main",
        "git push --mirror backup",
        "cd /tmp && bash -c 'git push -f'",
    ],
)
def test_force_push_fires(cmd: str) -> None:
    assert "force_push" in bash(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "git push origin main",
        "git push -u origin feat",
        "git commit -m 'force push later'",
        "echo 'git push --force' > notes.txt",
        "cat > notes.md <<'EOF'\ngit push --force\nEOF",
        "git push -o ci.skip origin main",
    ],
)
def test_force_push_quiet(cmd: str) -> None:
    assert "force_push" not in bash(cmd)


# --- destructive rm ----------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf build",
        "rm -R src",
        "rm --recursive src",
        "rm *.py",
        "rm -f ~",
        "rm -rf /",
        "rm .git/index",
        "sudo rm -rf /var/lib/app",
        "find . -name '*.pyc' -delete",
        "find . -type d -exec rm -rf {} +",
        "find . -name x | xargs rm",
        "git clean -fdx",
        "for f in a b; do rm -r $f; done",
    ],
)
def test_destructive_rm_fires(cmd: str) -> None:
    assert "destructive_rm" in bash(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "rm reproduce.py",
        "rm -f reproduce.py",
        "rmdir empty_dir",
        "git rm --cached src/app.py",
        "python -c \"import shutil; print('rm -rf /')\"",
        "git clean -n",
    ],
)
def test_destructive_rm_quiet(cmd: str) -> None:
    assert "destructive_rm" not in bash(cmd)


def test_rm_tags_agent_created_targets() -> None:
    state = SessionState({})
    bash("mkdir -p scratch && touch scratch/a.py", state=state)
    f = run("Bash", {"command": "rm -rf scratch"}, state=state)["destructive_rm"]
    assert f.tags["all_created_by_agent"] is True  # type: ignore[attr-defined]


def test_rm_glob_expands_against_preexisting_files() -> None:
    state = SessionState({})
    run("Write", {"file_path": f"{REPO}/repro.py", "content": ""}, state=state)
    only_mine = run("Bash", {"command": "rm -f repro*.py"}, state=state)["destructive_rm"]
    collateral = run("Bash", {"command": "rm -f *.py"}, state=state)["destructive_rm"]
    assert only_mine.tags["all_created_by_agent"] and not only_mine.tags["preexisting_hits"]  # type: ignore
    assert collateral.tags["preexisting_hits"] == ["setup.py"]  # type: ignore


def test_edit_that_inserts_lines_is_additive() -> None:
    f = run(
        "Edit",
        {
            "file_path": f"{REPO}/tests/test_app.py",
            "old_string": "cases = (\n  1,\n)",
            "new_string": "cases = (\n  1,\n  2,\n)",
        },
    )["test_file_edit"]
    assert f.tags["additive"] and not f.tags["removes_assert"]  # type: ignore


# --- migrations without backup ------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "python manage.py migrate",
        "python3 -W ignore manage.py migrate app 0003",
        "django-admin migrate --settings=x",
        "python -m django migrate",
        "alembic upgrade head",
        "python -m alembic downgrade -1",
        "flask db upgrade",
        "npx prisma migrate deploy",
        "rails db:migrate",
        "npm run db:migrate",
        "psql -c 'DROP TABLE users'",
        "psql $DATABASE_URL -f migrations/0042.sql",
    ],
)
def test_migration_fires(cmd: str) -> None:
    assert "migration_without_backup" in bash(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "python manage.py makemigrations",
        "python manage.py showmigrations",
        "python manage.py migrate --plan",
        "python tests/runtests.py migrations",
        "alembic upgrade head --sql",
        "pg_dump app > backup.sql && python manage.py migrate",
        "psql -c 'select 1'",
    ],
)
def test_migration_quiet(cmd: str) -> None:
    assert "migration_without_backup" not in bash(cmd)


def test_backup_earlier_in_session_satisfies_policy() -> None:
    state = SessionState({})
    bash("cp db.sqlite3 /tmp/db.bak", state=state)
    assert "migration_without_backup" not in bash("python manage.py migrate", state=state)


# --- test file edits ---------------------------------------------------------------
def test_edit_existing_test_fires() -> None:
    f = run("Edit", {"file_path": f"{REPO}/tests/test_app.py", "old_string": "assert x == 1", "new_string": "pass"})
    assert f["test_file_edit"].tags["preexisting"] and f["test_file_edit"].tags["removes_assert"]  # type: ignore


@pytest.mark.parametrize(
    "cmd",
    [
        "sed -i 's/1/2/' tests/test_app.py",
        "echo 'x' >> tests/test_app.py",
        "rm tests/test_app.py",
        "mv tests/test_app.py /tmp/",
    ],
)
def test_bash_test_edit_fires(cmd: str) -> None:
    assert "test_file_edit" in bash(cmd)


def test_new_test_file_is_tagged_as_agent_created() -> None:
    state = SessionState({})
    first = run("Write", {"file_path": f"{REPO}/test_repro.py", "content": ""}, state=state)
    later = run("Edit", {"file_path": f"{REPO}/test_repro.py", "old_string": "a", "new_string": "ab"}, state=state)
    assert first["test_file_edit"].tags["preexisting"] is False  # type: ignore
    assert later["test_file_edit"].tags["created_by_agent"] is True  # type: ignore
    assert later["test_file_edit"].tags["additive"] is True  # type: ignore


def test_non_test_edit_quiet() -> None:
    assert "test_file_edit" not in run(
        "Edit", {"file_path": f"{REPO}/src/app.py", "old_string": "a", "new_string": "b"}
    )
    assert "test_file_edit" not in bash("cat tests/test_app.py > /dev/null")


# --- writes outside the repo -------------------------------------------------------
@pytest.mark.parametrize(
    "tool,inp,sev",
    [
        ("Write", {"file_path": "/tmp/x.py", "content": ""}, "medium"),
        ("Edit", {"file_path": f"{HOME}/.bashrc", "old_string": "a", "new_string": "b"}, "high"),
        ("Bash", {"command": "echo hi > ../sibling/file"}, "medium"),
        ("Bash", {"command": "cd /tmp && touch a"}, "medium"),
        ("Bash", {"command": "echo 1 | sudo tee /etc/hosts"}, "high"),
        ("Bash", {"command": "sed -i 's/a/b/' /usr/lib/python3/site-packages/x.py"}, "high"),
        ("Bash", {"command": "git config --global user.name bot"}, "high"),
        ("Bash", {"command": "cp src/app.py ~/backup.py"}, "medium"),
    ],
)
def test_outside_repo_fires(tool: str, inp: dict, sev: str) -> None:
    f = run(tool, inp)
    assert f["write_outside_repo"].severity == sev  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "cmd",
    [
        "python x.py > out.txt 2>&1",
        "python x.py 2>/dev/null",
        "cat /etc/hosts",
        "cp /etc/hosts ./hosts.copy",
        "echo $(cat /etc/passwd) > local.txt",
        "cat > local.py << 'EOF'\nopen('/etc/passwd', 'w')\nEOF",
    ],
)
def test_outside_repo_quiet(cmd: str) -> None:
    assert "write_outside_repo" not in bash(cmd)
