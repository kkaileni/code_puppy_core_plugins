"""Discovery and chunking: turn a directory into judgeable source snippets.

Python files are split along the
AST (functions and methods stay whole; declarations longer than
``SPLIT_LINES`` are split into the blocks inside them so each snippet is one
behaviour). Every other file falls back to overlapping line windows. Line
coverage is exact: every non-blank line lands in at least one snippet.
"""

from __future__ import annotations

import ast
import os
import subprocess
from collections import Counter
from dataclasses import dataclass, field

from code_puppy.tools.ripgrep import find_ripgrep

# Discovery is bounded by files and bytes only. Snippet count is not capped:
# only the ranked shortlist is judged, and local BM25 over the ~24k snippets
# of a repo root takes well under a second.
MAX_FILES = 20_000
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_CHUNK_CHARS = 12_000
WHOLE_DECLARATION_LINES = 160
SPLIT_LINES = 24
WINDOW, OVERLAP = 60, 10


class LineTooLong(ValueError):
    """A single line is too long to judge (minified or generated source)."""


@dataclass(frozen=True)
class Chunk:
    path: str
    line: int
    end_line: int
    text: str
    symbol: str | None = None


@dataclass
class Discovery:
    chunks: list[Chunk] = field(default_factory=list)
    files: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    parsers: Counter = field(default_factory=Counter)


@dataclass
class _Range:
    start: int
    end: int
    symbol: str | None


def windows(
    lines: list[str],
    path: str,
    first_line: int = 1,
    size: int = WINDOW,
    overlap: int = OVERLAP,
    symbol: str | None = None,
) -> list[Chunk]:
    """Fixed-size line windows; giant windows are halved, never clipped."""
    out: list[Chunk] = []
    step = size - overlap
    for i in range(0, max(len(lines), 1), step):
        part = lines[i : i + size]
        text = "\n".join(part)
        if len(text) > MAX_CHUNK_CHARS:
            if size == 1:
                raise LineTooLong(
                    f"line exceeds {MAX_CHUNK_CHARS} characters: {path}:{first_line + i}"
                )
            out.extend(
                windows(part, path, first_line + i, max(1, size // 2), 0, symbol)
            )
        elif text.strip():
            out.append(
                Chunk(
                    path, first_line + i, first_line + i + len(part) - 1, text, symbol
                )
            )
        if i + size >= len(lines):
            break
    return out


def _begin(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _partition(node: ast.AST, start: int, end: int, symbol, depth: int = 0):
    """Split a long body into its blocks, covering every line of [start, end]."""
    body = getattr(node, "body", None)
    if not isinstance(body, list) or end - start < SPLIT_LINES:
        return None
    stmts = [s for s in body if hasattr(s, "end_lineno")]
    if not stmts:
        return None
    if len(stmts) == 1:
        return (
            _partition(stmts[0], start, end, symbol, depth + 1) if depth < 3 else None
        )

    out: list[_Range] = []
    cursor = start
    group: list[ast.stmt] = []

    def flush(last: bool) -> None:
        nonlocal cursor
        if not group:
            return
        stop = end if last else min(end, group[-1].end_lineno)
        if stop >= cursor:
            inner = None
            if len(group) == 1 and depth < 3:
                inner = _partition(group[0], cursor, stop, symbol, depth + 1)
            out.extend(inner or [_Range(cursor, stop, symbol)])
            cursor = stop + 1
        group.clear()

    for stmt in stmts:
        span = stmt.end_lineno - _begin(stmt) + 1
        if group and (span >= 6 or stmt.end_lineno - cursor + 1 > 20):
            flush(False)
        group.append(stmt)
        if span >= 6 and stmt is not stmts[-1]:
            flush(False)
    flush(True)
    return out if len(out) > 1 else None


def _python_ranges(text: str) -> list[_Range]:
    tree = ast.parse(text)
    ranges: list[_Range] = []

    def emit(node: ast.AST, start: int, symbol: str | None) -> None:
        ranges.extend(
            _partition(node, start, node.end_lineno, symbol)
            or [_Range(start, node.end_lineno, symbol)]
        )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            ranges.append(_Range(_begin(node), node.body[0].lineno - 1, node.name))
            for member in node.body:
                name = getattr(member, "name", "body")
                emit(member, _begin(member), f"{node.name}.{name}")
        else:
            emit(node, _begin(node), getattr(node, "name", None))
    return _merge_short_neighbours(ranges)


def _merge_short_neighbours(ranges: list[_Range]) -> list[_Range]:
    """Fold runs of imports/one-liners into one target (each costs a judgment)."""

    def mergeable(r: _Range) -> bool:
        return r.end - r.start <= 1 and "." not in (r.symbol or "")

    merged: list[_Range] = []
    for r in ranges:
        last = merged[-1] if merged else None
        if (
            last
            and mergeable(last)
            and mergeable(r)
            and r.start <= last.end + 1
            and r.end - last.start < 12
            and len(f"{last.symbol} {r.symbol}") < 60
        ):
            names = dict.fromkeys(s for s in (last.symbol, r.symbol) if s)
            last.end, last.symbol = r.end, ", ".join(names) or None
        else:
            merged.append(_Range(r.start, r.end, r.symbol))
    return merged


def _chunks_from_ranges(
    lines: list[str], path: str, ranges: list[_Range]
) -> list[Chunk]:
    out: list[Chunk] = []
    next_line = 1
    for r in ranges:
        start, end = min(r.start, next_line), min(r.end, len(lines))
        if end < start:
            continue
        part = lines[start - 1 : end]
        whole = (
            end - start < WHOLE_DECLARATION_LINES
            and len("\n".join(part)) <= MAX_CHUNK_CHARS
        )
        size, overlap = (len(part), 0) if whole else (WINDOW, OVERLAP)
        out.extend(windows(part, path, start, size, overlap, r.symbol))
        next_line = end + 1
    if next_line <= len(lines):
        out.extend(windows(lines[next_line - 1 :], path, next_line))
    return out


def source_chunks(text: str, path: str) -> tuple[list[Chunk], str]:
    """Chunk one file's text. Returns (chunks, parser_name)."""
    lines = text.splitlines()
    if path.endswith(".py"):
        try:
            ranges = _python_ranges(text)
        except (SyntaxError, ValueError, RecursionError):
            ranges = []
        if ranges:
            return _chunks_from_ranges(lines, path, ranges), "python"
    return windows(lines, path), "overlapping-lines"


def list_files(directory: str, glob: str | None = None) -> list[str]:
    """Files ripgrep would search: honours .gitignore, skips hidden files."""
    rg = find_ripgrep()
    if rg is None:
        raise RuntimeError("ripgrep (rg) is required for smart_grep.")
    args = [rg, "--files", "-0", *(["-g", glob] if glob else []), "--", directory]
    proc = subprocess.run(args, capture_output=True, timeout=15, check=False)
    if proc.returncode not in (0, 1):  # 1 == no files
        raise RuntimeError(
            f"ripgrep discovery failed: {proc.stderr.decode(errors='replace')[:300]}"
        )
    files = sorted({f for f in proc.stdout.decode(errors="replace").split("\0") if f})
    if len(files) > MAX_FILES:
        raise RuntimeError(
            f"Search exceeds {MAX_FILES} files. Narrow the directory or glob."
        )
    return files


def discover(directory: str, glob: str | None = None) -> Discovery:
    """Read and chunk every eligible file under ``directory``."""
    found = Discovery()
    files = list_files(directory, glob)
    found.files = len(files)
    total_bytes = 0
    for path in files:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            found.skipped.append((path, "larger than 1 MiB"))
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_BYTES:
            raise RuntimeError(
                "Search exceeds 32 MiB of source. Narrow the directory or glob."
            )
        if b"\0" in raw:
            found.skipped.append((path, "binary"))
            continue
        try:
            text = raw.decode("utf-8")
            chunks, parser = source_chunks(text, path)
        except UnicodeDecodeError:
            found.skipped.append((path, "not UTF-8"))
            continue
        except LineTooLong as exc:
            found.skipped.append((path, str(exc)))
            continue
        found.parsers[parser] += 1
        found.chunks.extend(chunks)
    return found
