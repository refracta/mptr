from __future__ import annotations

import html
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TexSpan:
    tex: str
    display: bool


LATEX_SPAN_RE = (
    r"\\\((?P<inline>.+?)\\\)"  # \( ... \)
    r"|\\\[(?P<display>.+?)\\\]"  # \[ ... \]
)


def split_text_with_tex(text: str) -> list[str | TexSpan]:
    import re

    if not text:
        return [""]

    pattern = re.compile(LATEX_SPAN_RE, flags=re.DOTALL)
    out: list[str | TexSpan] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            out.append(text[pos : m.start()])
        if m.group("inline") is not None:
            out.append(TexSpan(tex=m.group("inline"), display=False))
        elif m.group("display") is not None:
            out.append(TexSpan(tex=m.group("display"), display=True))
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


def iter_tex_spans(texts: Iterable[str]) -> list[TexSpan]:
    seen: set[tuple[str, bool]] = set()
    out: list[TexSpan] = []
    for t in texts:
        for part in split_text_with_tex(t):
            if isinstance(part, TexSpan):
                key = (part.tex, bool(part.display))
                if key in seen:
                    continue
                seen.add(key)
                out.append(part)
    return out


def _repo_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_mathjax_installed(repo_dir: Path) -> None:
    if (repo_dir / "node_modules" / "mathjax-full").exists():
        return
    raise RuntimeError(
        "MathJax (node dependency) not installed. Run `npm install` in the mptr repo root to enable LaTeX rendering."
    )


_SVG_CACHE: dict[tuple[str, bool, float], str] = {}


def tex_to_svg_batch(spans: list[TexSpan], *, math_scale: float = 1.0) -> dict[tuple[str, bool], str]:
    if not spans:
        return {}

    scale_key = round(float(math_scale), 4)
    want3 = {(s.tex, bool(s.display), scale_key) for s in spans}
    missing = [s for s in spans if (s.tex, bool(s.display), scale_key) not in _SVG_CACHE]
    if not missing:
        return {(tex, disp): _SVG_CACHE[(tex, disp, scale_key)] for (tex, disp, _s) in want3 if (tex, disp, scale_key) in _SVG_CACHE}

    repo_dir = _repo_dir()
    _ensure_mathjax_installed(repo_dir)

    script = repo_dir / "scripts" / "tex2svg.mjs"
    req = [{"tex": s.tex, "display": bool(s.display)} for s in missing]

    env = dict(os.environ)
    env["MPTR_MATH_SCALE"] = str(float(math_scale))
    proc = subprocess.run(
        ["node", str(script)],
        input=json.dumps(req).encode("utf-8"),
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"TeX→SVG conversion failed (node exit={proc.returncode}): {stderr}")

    try:
        items = json.loads(proc.stdout.decode("utf-8"))
    except Exception as e:
        out = proc.stdout.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"TeX→SVG conversion returned invalid JSON: {e}: {out}") from e

    out: dict[tuple[str, bool, float], str] = {}
    if not isinstance(items, list):
        return {}
    for it in items:
        if not isinstance(it, dict):
            continue
        tex = str(it.get("tex") or "")
        display = bool(it.get("display"))
        svg = it.get("svg")
        if isinstance(svg, str) and svg.strip():
            out[(tex, display, scale_key)] = svg

    _SVG_CACHE.update(out)
    return {(tex, disp): _SVG_CACHE[(tex, disp, scale_key)] for (tex, disp, _s) in want3 if (tex, disp, scale_key) in _SVG_CACHE}


def build_html_with_math(text: str, svg_map: dict[tuple[str, bool], str]) -> str:
    parts = split_text_with_tex(text)
    html_parts: list[str] = []
    for p in parts:
        if isinstance(p, TexSpan):
            svg = svg_map.get((p.tex, bool(p.display)))
            if svg:
                if p.display:
                    html_parts.append(f'<div class="mptr-math-block">{svg}</div>')
                else:
                    html_parts.append(svg)
            else:
                # Fallback: keep TeX visible if we couldn't render it.
                html_parts.append(html.escape(f"\\[{p.tex}\\]" if p.display else f"\\({p.tex}\\)"))
        else:
            html_parts.append(html.escape(p))
    return "".join(html_parts)
