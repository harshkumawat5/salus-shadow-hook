"""A small, dependency-free analyser for bash command strings.

This is deliberately not a full bash parser. It does the handful of things the
policies need and that naive regexes get wrong:

* splits a command line into simple commands on ``&& || ; | & newline ( )``;
* keeps heredoc bodies and quoted strings out of the command stream, so
  ``cat > notes.md <<EOF ... rm -rf / ... EOF`` is one ``cat``, not an ``rm``;
* records redirections (``> file``, ``>> file``, ``&> file``) with their targets;
* strips wrappers (``sudo``, ``env``, ``timeout``, ``nohup``, ``xargs``, ...);
* recurses into ``bash -c '...'``, ``eval ...`` and ``$(...)`` / backticks;
* tracks ``cd`` so relative paths resolve against the right directory.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field

SEPARATORS = {"&&", "||", ";", ";;", "|", "|&", "&", "\n"}
WRITE_REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&", "<>"}
_REDIRECT_OPS = ["&>>", "<<<", "<<-", "&>", ">>", ">|", ">&", "<<", "<&", "<>", ">", "<"]
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
_RESERVED = {"if", "then", "else", "elif", "while", "until", "do", "!", "{"}


@dataclass
class Word:
    text: str
    dynamic: bool = False  # contains $VAR, $(...) or backticks: value unknown statically
    glob: bool = False  # contains unquoted * ? [


@dataclass
class Command:
    """One simple command after wrapper stripping."""

    words: list[Word]
    redirects: list[tuple[str, Word]] = field(default_factory=list)
    heredoc: str = ""
    cwd: str | None = None
    via: tuple[str, ...] = ()  # wrappers that were stripped, e.g. ("sudo",) or ("xargs",)

    @property
    def argv(self) -> list[str]:
        return [w.text for w in self.words]

    @property
    def name(self) -> str:
        return posixpath.basename(self.words[0].text) if self.words else ""

    @property
    def args(self) -> list[Word]:
        return self.words[1:]


# --------------------------------------------------------------------------- tokenizer


def _balanced(src: str, i: int, open_: str, close: str) -> int:
    """Index just past the `close` matching the `open_` at src[i], skipping quotes."""
    depth, j, n = 0, i, len(src)
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "'":
            k = src.find("'", j + 1)
            j = n if k < 0 else k + 1
            continue
        if c == '"':
            j += 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


class _Tokenizer:
    def __init__(self, src: str) -> None:
        self.src = src
        self.i = 0
        self.tokens: list[tuple[str, object]] = []  # ("w", Word) | ("op", str) | ("heredoc", str)
        self.substitutions: list[str] = []
        self._buf: list[str] = []
        self._in_word = False
        self._dynamic = False
        self._glob = False
        self._pending_heredocs: list[tuple[str, bool]] = []

    # word buffer -----------------------------------------------------------
    def _flush(self) -> None:
        if self._in_word:
            word = Word("".join(self._buf), self._dynamic, self._glob)
            if self.tokens and self.tokens[-1] in (("op", "<<"), ("op", "<<-")):
                self._pending_heredocs.append((word.text, self.tokens[-1][1] == "<<-"))
            self.tokens.append(("w", word))
        self._buf, self._in_word, self._dynamic, self._glob = [], False, False, False

    def _take_dollar(self) -> None:
        src, i = self.src, self.i
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if nxt == "(":
            end = _balanced(src, i + 1, "(", ")")
            if not src.startswith("$((", i):
                self.substitutions.append(src[i + 2 : end - 1])
            self._buf.append(src[i:end])
            self.i = end
        elif nxt == "{":
            end = _balanced(src, i + 1, "{", "}")
            self._buf.append(src[i:end])
            self.i = end
        else:
            self._buf.append("$")
            self.i += 1
        self._dynamic = self._dynamic or nxt not in ("", " ", "\t", "\n", '"')
        self._in_word = True

    def _take_backtick(self) -> None:
        end = self.src.find("`", self.i + 1)
        end = len(self.src) if end < 0 else end
        self.substitutions.append(self.src[self.i + 1 : end])
        self._buf.append(self.src[self.i : end + 1])
        self.i = end + 1
        self._in_word = self._dynamic = True

    def _take_double_quoted(self) -> None:
        src, n = self.src, len(self.src)
        self.i += 1
        self._in_word = True
        while self.i < n and src[self.i] != '"':
            c = src[self.i]
            if c == "\\" and self.i + 1 < n and src[self.i + 1] in '"\\$`\n':
                if src[self.i + 1] != "\n":
                    self._buf.append(src[self.i + 1])
                self.i += 2
            elif c == "$":
                self._take_dollar()
            elif c == "`":
                self._take_backtick()
            else:
                self._buf.append(c)
                self.i += 1
        self.i += 1

    def _read_heredocs(self) -> None:
        src, n = self.src, len(self.src)
        for delim, strip_tabs in self._pending_heredocs:
            body: list[str] = []
            while self.i < n:
                j = src.find("\n", self.i)
                j = n if j < 0 else j
                line = src[self.i : j]
                self.i = j + 1
                probe = line.lstrip("\t") if strip_tabs else line
                if probe.rstrip("\r") == delim:
                    break
                body.append(line)
            self.tokens.append(("heredoc", "\n".join(body)))
        self._pending_heredocs = []

    # main loop ---------------------------------------------------------------
    def run(self) -> _Tokenizer:
        src, n = self.src, len(self.src)
        while self.i < n:
            c = src[self.i]
            if c == "\\":
                if src.startswith("\\\n", self.i):
                    self.i += 2
                    continue
                self._buf.append(src[self.i + 1 : self.i + 2])
                self._in_word = True
                self.i += 2
            elif c == "'":
                end = src.find("'", self.i + 1)
                end = n if end < 0 else end
                self._buf.append(src[self.i + 1 : end])
                self._in_word = True
                self.i = end + 1
            elif c == '"':
                self._take_double_quoted()
            elif c == "$":
                self._take_dollar()
            elif c == "`":
                self._take_backtick()
            elif c in " \t\r":
                self._flush()
                self.i += 1
            elif c == "\n":
                self._flush()
                self.i += 1
                if self._pending_heredocs:  # bodies belong to the command on this line
                    self._read_heredocs()
                self.tokens.append(("op", "\n"))
            elif c == "#" and not self._in_word:
                end = src.find("\n", self.i)
                self.i = n if end < 0 else end
            elif c in "<>" or (c == "&" and src.startswith("&>", self.i)):
                if c != "&" and self._in_word and "".join(self._buf).isdigit():
                    self._buf, self._in_word = [], False  # "2>" : fd prefix, not a word
                self._flush()
                op = next(o for o in _REDIRECT_OPS if src.startswith(o, self.i))
                self.tokens.append(("op", op))
                self.i += len(op)
            elif c in ";|&":
                self._flush()
                two = src[self.i : self.i + 2]
                op = two if two in ("&&", "||", ";;", "|&") else c
                self.tokens.append(("op", op))
                self.i += len(op)
            elif c == ")" or (c == "(" and not self._in_word):
                self._flush()
                self.tokens.append(("op", c))
                self.i += 1
            else:
                if c in "*?[":
                    self._glob = True
                self._buf.append(c)
                self._in_word = True
                self.i += 1
        self._flush()
        if self._pending_heredocs:
            self._read_heredocs()
        return self


# --------------------------------------------------------------------------- paths


def resolve(word: Word | str, cwd: str | None, home: str | None) -> str | None:
    """Absolute, normalised path for a word, or None if it can't be known statically."""
    text = word.text if isinstance(word, Word) else word
    if home:
        if text == "~" or text.startswith("~/"):
            text = home + text[1:]
        for var in ("$HOME", "${HOME}"):
            if text == var or text.startswith(var + "/"):
                text = home + text[len(var) :]
    if "$" in text or "`" in text or text.startswith("~"):
        return None
    if not text.startswith("/"):
        if not cwd:
            return None
        text = posixpath.join(cwd, text)
    return posixpath.normpath(text)


# --------------------------------------------------------------------------- parser

_SUDO_ARG_OPTS = {"-u", "-g", "-C", "-h", "-p", "-r", "-t", "-U", "-D", "-R"}
_XARGS_ARG_OPTS = {"-I", "-i", "-n", "-P", "-L", "-l", "-d", "-E", "-e", "-s", "-a", "--max-args", "--max-procs"}


def _is_assignment(text: str) -> bool:
    name, eq, _ = text.partition("=")
    return bool(eq) and name.replace("_", "a").isalnum() and not name[:1].isdigit()


def _strip_wrappers(words: list[Word]) -> tuple[list[Word], tuple[str, ...]]:
    """Remove VAR=x prefixes and transparent wrappers; return (words, wrappers)."""
    via: list[str] = []
    while words:
        name = posixpath.basename(words[0].text)
        if _is_assignment(words[0].text) or words[0].text in _RESERVED:
            words = words[1:]
            continue
        if name in ("sudo", "doas"):
            via.append(name)
            i = 1
            while i < len(words) and words[i].text.startswith("-"):
                i += 2 if words[i].text in _SUDO_ARG_OPTS else 1
            words = words[i:]
        elif name == "env":
            via.append(name)
            i = 1
            while i < len(words) and (words[i].text.startswith("-") or _is_assignment(words[i].text)):
                i += 2 if words[i].text in ("-u", "-C", "-S") else 1
            words = words[i:]
        elif name in ("nohup", "time", "command", "builtin", "exec", "stdbuf", "unbuffer"):
            via.append(name)
            i = 1
            while i < len(words) and words[i].text.startswith("-"):
                i += 1
            words = words[i:]
        elif name == "nice":
            via.append(name)
            i = 1
            while i < len(words) and words[i].text.startswith("-"):
                i += 2 if words[i].text == "-n" else 1
            words = words[i:]
        elif name == "timeout":
            via.append(name)
            i = 1
            while i < len(words) and words[i].text.startswith("-"):
                i += 2 if words[i].text in ("-s", "-k", "--signal", "--kill-after") else 1
            words = words[i + 1 :]  # skip DURATION
        elif name == "xargs":
            via.append(name)
            i = 1
            while i < len(words) and words[i].text.startswith("-"):
                i += 2 if words[i].text in _XARGS_ARG_OPTS else 1
            words = words[i:]
        else:
            break
    return words, tuple(via)


def _shell_c_script(words: list[Word]) -> str | None:
    """The script of `bash -c '...'` / `sh -lc '...'`, else None."""
    if not words or posixpath.basename(words[0].text) not in _SHELLS:
        return None
    for i, w in enumerate(words[1:], start=1):
        t = w.text
        if t.startswith("-") and not t.startswith("--") and "c" in t[1:]:
            return words[i + 1].text if i + 1 < len(words) else None
        if not t.startswith("-"):
            return None
    return None


def parse(command: str, cwd: str | None = None, home: str | None = None, _depth: int = 0) -> list[Command]:
    """Split `command` into simple commands (in execution order, approximately)."""
    tok = _Tokenizer(command).run()
    out: list[Command] = []
    cwd_stack: list[str | None] = []
    cur_words: list[Word] = []
    cur_redirects: list[tuple[str, Word]] = []
    cur_heredoc: list[str] = []
    state = {"cwd": cwd}

    def finish() -> None:
        nonlocal cur_words, cur_redirects, cur_heredoc
        if cur_words or cur_redirects:
            words, via = _strip_wrappers(cur_words)
            out.append(Command(words, cur_redirects, "\n".join(cur_heredoc), state["cwd"], via))
            script = _shell_c_script(words)
            if script is None and words and words[0].text == "eval":
                script = " ".join(w.text for w in words[1:])
            if script is not None and _depth < 4:
                inner = parse(script, state["cwd"], home, _depth + 1)
                out.extend(
                    Command(c.words, c.redirects, c.heredoc, c.cwd, via + (words[0].text,) + c.via) for c in inner
                )
            else:
                if words and words[0].text in ("cd", "pushd"):
                    target = next((w for w in words[1:] if not w.text.startswith("-")), None)
                    if target is None:
                        state["cwd"] = home
                    else:
                        state["cwd"] = resolve(target, state["cwd"], home)
        cur_words, cur_redirects, cur_heredoc = [], [], []

    toks = tok.tokens
    i = 0
    while i < len(toks):
        kind, val = toks[i]
        if kind == "w":
            cur_words.append(val)  # type: ignore[arg-type]
        elif kind == "heredoc":
            cur_heredoc.append(val)  # type: ignore[arg-type]
        elif val in SEPARATORS:
            finish()
        elif val == "(":
            finish()
            cwd_stack.append(state["cwd"])
        elif val == ")":
            finish()
            if cwd_stack:
                state["cwd"] = cwd_stack.pop()
        else:  # redirect operator; its target is the next word
            target = toks[i + 1][1] if i + 1 < len(toks) and toks[i + 1][0] == "w" else None
            if target is not None:
                i += 1
                if not (val in (">&", "<&") and (target.text.isdigit() or target.text == "-")):  # type: ignore[union-attr]
                    cur_redirects.append((str(val), target))  # type: ignore[arg-type]
        i += 1
    finish()

    if _depth < 4:
        for sub in tok.substitutions:
            out.extend(parse(sub, cwd, home, _depth + 1))
    return out
