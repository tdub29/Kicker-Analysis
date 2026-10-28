"""Repository hygiene: no personal paths in committed text, no dead links in the docs.

WHY: neither check is about the model, and neither defect is caught by running
code, because a path in a docstring or a CSV never executes and Markdown is never
executed at all.

  personal paths  predict_week.py once named the site payload's location as a
                  module-level constant under the author's Windows profile
                  directory. That works on one machine only, and in a public
                  repo it tells every reader a personal folder layout. The same
                  leak can arrive through a log, an experiment's saved output or
                  a CSV written with an absolute path, so every committed text
                  file is read, not just the source.
  dead links      README.md and METHOD.md are what a reader opens first, and a
                  relative link or image that does not resolve renders on GitHub
                  as a dead link or a broken-image icon with no error anywhere.
                  The README's charts are generated files (figures.py writes
                  them, light and dark), so a renamed chart breaks the page
                  silently. A target must exist with the exact case GitHub will
                  look for (a case-insensitive disk would hide a mismatch) and
                  must be committable: an image that exists only because
                  .gitignore keeps it out of the repo is still broken on GitHub.

"Committed" means what `git add -A` would commit: tracked files still on disk
plus untracked files .gitignore does not exclude, so a new file is checked before
it is added. Without git (a source archive) the tree is walked, skipping the
directories .gitignore excludes.

Standard library only, offline, and well under a second.

    pytest -q tests/test_repo_hygiene.py
"""
from __future__ import annotations

import html
import os
import pathlib
import re
import subprocess
import urllib.parse

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LINKED_DOCS = ["README.md", "METHOD.md"]

# Used only when git cannot list the files. Mirrors the directories .gitignore
# excludes; data/ is all regenerable cache and build output.
WALK_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
                  ".venv", "venv", ".env", "build", "dist", "node_modules", ".agents"}
WALK_SKIP_TOP = {"data"}

# --- personal paths -------------------------------------------------------------
#
# A home directory followed by an account name: a Windows profile (any drive,
# either slash, doubled backslashes as written in source), a macOS or Linux home,
# and root's home. The leading lookbehind keeps a URL path segment such as
# example.com/home/ from matching. The patterns are assembled so this file does
# not contain the literal strings it searches for.
_NAME = r"([^\\/\s\"'<>|:*?`]+)"
PERSONAL_PATTERNS = [
    re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]+(?:users|documents and settings)[\\/]+" + _NAME),
    re.compile(r"(?<![\w.~-])/(?:home|Users)/" + _NAME),
    re.compile(r"(?<![\w.~-])/(root)/"),
]
# Account names that are placeholders a reader substitutes, not a real person.
# "user" is deliberately absent: it is a real account name on many machines, so
# a path under it cannot be told apart from a leak.
PLACEHOLDER_NAMES = {"someone", "you", "username", "<user>", "<username>", "<you>",
                     "<name>", "%username%", "$user", "${user}"}


def personal_paths(text: str) -> list[str]:
    """Every personal absolute path in `text`, as the matched prefix."""
    hits = []
    for pat in PERSONAL_PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1)
            if pat.pattern.endswith("(root)/") or name.lower() not in PLACEHOLDER_NAMES:
                hits.append(m.group(0))
    return hits


# --- which files ------------------------------------------------------------------

def _git_files() -> list[str] | None:
    """Repo-relative paths `git add -A` would commit, or None without git."""
    try:
        r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others",
                            "--exclude-standard"], cwd=ROOT, capture_output=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    names = {n for n in r.stdout.decode("utf-8", "surrogateescape").split("\0") if n}
    # A tracked file deleted from the working tree is about to leave the repo.
    return sorted(n for n in names if (ROOT / n).is_file())


def _walked_files() -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = pathlib.Path(dirpath).relative_to(ROOT)
        dirnames[:] = [d for d in dirnames
                       if d not in WALK_SKIP_DIRS and not d.endswith(".egg-info")
                       and not (rel == pathlib.Path(".") and d in WALK_SKIP_TOP)]
        out += [(rel / f).as_posix() for f in filenames]
    return sorted(out)


GIT_FILES = _git_files()
COMMITTED = GIT_FILES if GIT_FILES is not None else _walked_files()


def _read_text(rel: str) -> str | None:
    """The file as text, or None for a binary file (a NUL byte in its first 8 KiB)."""
    data = (ROOT / rel).read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


# --- links --------------------------------------------------------------------------

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1[`~]*[ \t]*$", re.M | re.S)
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
# Inline link or image: the destination after "](", bare or in <angle brackets>.
# Matching on "](" alone also handles a link wrapped around an image.
_INLINE = re.compile(r"\]\(\s*(<[^>\n]*>|[^)\s]+)")
# A reference definition, "[label]: target". "[^1]: text" is a footnote, not a link.
_REFDEF = re.compile(r"^ {0,3}\[(?!\^)[^\]\n]+\]:[ \t]*(<[^>\n]*>|\S+)", re.M)
_TAG = re.compile(r"<(img|source|a)\b([^>]*)>", re.I)
# Quoted or bare: GitHub renders <img src=docs/x.png> as readily as the quoted form.
_ATTR = re.compile(r"""\b(src|srcset|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""",
                   re.I)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def link_targets(markdown: str) -> list[str]:
    """Every link and image destination in `markdown`, as written, outside code."""
    text = _COMMENT.sub("", _CODE_SPAN.sub("", _FENCE.sub("", markdown)))
    found = [m.group(1) for m in _INLINE.finditer(text)]
    found += [m.group(1) for m in _REFDEF.finditer(text)]
    for tag in _TAG.finditer(text):
        for a in _ATTR.finditer(tag.group(2)):
            value = html.unescape(next(g for g in a.group(2, 3, 4) if g is not None))
            if a.group(1).lower() == "srcset":
                # "a.png 1x, b.png 2x": the URL is each candidate's first token.
                found += [c.split()[0] for c in value.split(",") if c.split()]
            else:
                found.append(value)
    return [t[1:-1] if t.startswith("<") and t.endswith(">") else t for t in found]


def local_target(target: str) -> str | None:
    """The repo path a relative link points at, or None for a URL or a bare anchor."""
    if not target or target.startswith(("#", "//")) or _SCHEME.match(target):
        return None
    path = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
    return path or None


def _exists_exact_case(path: pathlib.Path) -> bool:
    """True if `path` exists under ROOT with every component's case as written."""
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return False
    here = ROOT
    for part in rel.parts:
        if part not in os.listdir(here):
            return False
        here = here / part
    return True


def broken_links(doc: str) -> list[str]:
    src = ROOT / doc
    problems = []
    committed = set(COMMITTED)
    for target in link_targets(src.read_text(encoding="utf-8")):
        rel = local_target(target)
        if rel is None:
            continue
        # GitHub resolves a leading slash against the repository root.
        base = ROOT if rel.startswith("/") else src.parent
        path = pathlib.Path(os.path.normpath(base / rel.lstrip("/")))
        if not _exists_exact_case(path):
            problems.append(f"{target}: no such file (exact case) at "
                            f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
            continue
        r = path.relative_to(ROOT).as_posix()
        if path.is_file() and r not in committed:
            problems.append(f"{target}: exists but would not be committed "
                            f"(ignored by .gitignore?)")
        elif path.is_dir() and r != "." and not any(c.startswith(r + "/") for c in committed):
            problems.append(f"{target}: a directory with no committed file in it")
    return problems


# --- tests -----------------------------------------------------------------------------

def test_files_were_listed():
    # An empty list would make every check below pass on nothing.
    assert COMMITTED, "no files found to check"
    assert "README.md" in COMMITTED and "tests/test_repo_hygiene.py" in COMMITTED, (
        "the file list is missing files it must contain")


def test_personal_path_patterns():
    """The scanner catches the real thing, allows placeholders, ignores URLs."""
    sep = "\\"
    profile = "C:" + sep + "Users" + sep + "JaneDoe" + sep + "Downloads"
    assert personal_paths(f'SITE = pathlib.Path(r"{profile}")')
    assert personal_paths(profile.replace(sep, sep * 2))       # as written in a str
    assert personal_paths("d:" + "/users/jane/site")           # any drive, forward slash
    assert personal_paths("cache at " + "/" + "Users/jane/kick")
    assert personal_paths("cwd: " + "/" + "home/jane/Kicker-Analysis")
    assert personal_paths("cwd: " + "/" + "home/user/Kicker-Analysis")
    assert personal_paths("cert: " + "/" + "root/.config/ca.crt")
    assert not personal_paths("C:" + sep + "Users" + sep + "someone" + sep + "site")
    assert not personal_paths("https://example.com" + "/" + "home/jane/page")
    assert not personal_paths("home/away splits and ~/Downloads stay quiet")


def test_no_personal_paths_in_committed_text():
    offenders = []
    for rel in COMMITTED:
        text = _read_text(rel)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for hit in personal_paths(line):
                offenders.append(f"{rel}:{lineno}: {hit}")
    assert not offenders, ("personal absolute paths in committed files; use a path "
                           "relative to the repo, an environment variable, or a "
                           "placeholder such as 'someone':\n  " + "\n  ".join(offenders))


def test_link_parser_finds_every_form():
    """The parser sees each link form the docs use, and skips code and URLs.

    The README has few relative links, so a parser that silently found none
    would pass the real check below on nothing.
    """
    md = "\n".join([
        "See [the method](METHOD.md#evaluation) and [notes][n].",
        "[![badge](docs/badge.svg)](https://example.com/ci)",
        "![chart](<docs/images/a b.png>)",
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="docs/d.png 1x, docs/d2.png 2x">',
        "  <img alt='x' src='docs/l.png'>",
        "</picture>",
        "<img src=docs/bare.png width=400>",
        '<a href="LICENSE">licence</a> <a href="#top">top</a>',
        "`[not](code.md)`",
        "```",
        "[also not](fenced.md)",
        "```",
        "<!-- [hidden](comment.md) -->",
        "[n]: notes/x.md \"title\"",
        "[^1]: A footnote, not a link.",
    ])
    targets = [local_target(t) for t in link_targets(md)]
    assert sorted(t for t in targets if t) == sorted([
        "METHOD.md", "notes/x.md", "docs/badge.svg", "docs/images/a b.png",
        "docs/d.png", "docs/d2.png", "docs/l.png", "docs/bare.png", "LICENSE"]), targets


@pytest.mark.parametrize("doc", LINKED_DOCS)
def test_relative_links_resolve(doc: str):
    assert (ROOT / doc).is_file(), f"{doc} is missing"
    problems = broken_links(doc)
    assert not problems, f"{doc} has links that will not resolve on GitHub:\n  " + \
        "\n  ".join(problems)
