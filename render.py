from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from latex import (
    TexSpan,
    iter_tex_spans,
    parse_mathjax_svg_metrics,
    split_text_with_tex,
    tex_to_svg_batch,
)
from sentences import Sentence
from util import normalize_path


@dataclass(frozen=True)
class RenderStats:
    inserted: int
    skipped: int
    overflow: int


def _has_tex(text: str) -> bool:
    return any(isinstance(p, TexSpan) for p in split_text_with_tex(text))


def _css_text_align(align: int) -> str:
    if int(align) == fitz.TEXT_ALIGN_CENTER:
        return "center"
    if int(align) == fitz.TEXT_ALIGN_RIGHT:
        return "right"
    if int(align) == fitz.TEXT_ALIGN_JUSTIFY:
        return "justify"
    return "left"


def _region_to_rect(
    region: dict,
    *,
    scale_x: float,
    scale_y: float,
    x_offset: float = 0.0,
    padding: float = 0.0,
) -> fitz.Rect:
    x0 = float(region["top_left_x"]) * scale_x + x_offset
    y0 = float(region["top_left_y"]) * scale_y
    x1 = (float(region["top_left_x"]) + float(region["width"])) * scale_x + x_offset
    y1 = (float(region["top_left_y"]) + float(region["height"])) * scale_y
    rect = fitz.Rect(x0, y0, x1, y1)
    if padding:
        rect.x0 += padding
        rect.y0 += padding
        rect.x1 -= padding
        rect.y1 -= padding
    return rect


@dataclass(frozen=True)
class _TextPiece:
    text: str
    width: float


@dataclass(frozen=True)
class _SpacePiece:
    width: float


@dataclass(frozen=True)
class _MathPiece:
    key: tuple[str, bool]
    width: float
    height: float
    valign: float
    display: bool


@dataclass(frozen=True)
class _Layout:
    fontsize: float
    ascent: float
    descent: float
    line_height: float
    lines: list[list[_TextPiece | _SpacePiece | _MathPiece]]


def _build_math_docs(svg_map: dict[tuple[str, bool], str]) -> dict[tuple[str, bool], fitz.Document]:
    out: dict[tuple[str, bool], fitz.Document] = {}
    for key, svg in svg_map.items():
        src = fitz.open("svg", svg.encode("utf-8"))
        try:
            pdf_bytes = src.convert_to_pdf()
        finally:
            src.close()
        out[key] = fitz.open("pdf", pdf_bytes)
    return out


def _layout_sentence(
    *,
    text: str,
    rect: fitz.Rect,
    font: fitz.Font,
    fontsize: float,
    svg_map: dict[tuple[str, bool], str],
    math_scale: float,
    math_inline_cap: float,
) -> _Layout | None:
    if rect.width <= 0 or rect.height <= 0:
        return None

    ascent = float(font.ascender) * float(fontsize)
    descent = -float(font.descender) * float(fontsize)
    line_height = float(fontsize) * 1.2

    ex_pt = (float(fontsize) * float(math_scale)) / 2.0
    em_h = (float(font.ascender) - float(font.descender)) * float(fontsize)
    inline_cap_h = float(em_h) * float(max(0.1, math_inline_cap))

    lines: list[list[_TextPiece | _SpacePiece | _MathPiece]] = []
    cur: list[_TextPiece | _SpacePiece | _MathPiece] = []
    cur_w = 0.0

    def flush() -> None:
        nonlocal cur, cur_w
        # Trim trailing spaces.
        while cur and isinstance(cur[-1], _SpacePiece):
            cur_w -= float(cur[-1].width)
            cur.pop()
        lines.append(cur)
        cur = []
        cur_w = 0.0

    parts = split_text_with_tex(text)
    for part in parts:
        if isinstance(part, TexSpan):
            key = (part.tex, bool(part.display))
            svg = svg_map.get(key)
            metrics = parse_mathjax_svg_metrics(svg or "")
            if not metrics:
                # If we can't parse the SVG metrics, treat as plain text fallback.
                fallback = f"\\[{part.tex}\\]" if part.display else f"\\({part.tex}\\)"
                parts2 = [fallback]
                # Re-run as plain text.
                for t in parts2:
                    for ch in t:
                        if ch == "\n":
                            if cur:
                                flush()
                            continue
                        if ch.isspace():
                            if not cur:
                                continue
                        w = float(font.text_length(ch, fontsize=float(fontsize)))
                        if (cur_w + w) > rect.width and cur:
                            flush()
                        if ch.isspace():
                            cur.append(_SpacePiece(width=w))
                        else:
                            if cur and isinstance(cur[-1], _TextPiece):
                                prev = cur[-1]
                                cur[-1] = _TextPiece(text=prev.text + ch, width=prev.width + w)
                            else:
                                cur.append(_TextPiece(text=ch, width=w))
                        cur_w += w
                continue

            w = float(metrics.width_ex) * ex_pt
            h = float(metrics.height_ex) * ex_pt
            valign = float(metrics.valign_ex) * ex_pt

            piece = _MathPiece(key=key, width=w, height=h, valign=valign, display=bool(part.display))

            if piece.display:
                # Force block math to its own line.
                if cur:
                    flush()
                cur.append(piece)
                cur_w = w
                flush()
                continue

            if not piece.display and piece.height > inline_cap_h and piece.height > 0:
                s = inline_cap_h / piece.height
                piece = _MathPiece(
                    key=piece.key,
                    width=piece.width * s,
                    height=piece.height * s,
                    valign=piece.valign * s,
                    display=piece.display,
                )

            if (cur_w + piece.width) > rect.width and cur:
                flush()
            if piece.width > rect.width and not cur:
                # Cannot fit even on an empty line.
                return None
            cur.append(piece)
            cur_w += piece.width
            continue

        s = str(part)
        for ch in s:
            if ch == "\n":
                if cur:
                    flush()
                continue
            if ch.isspace():
                if not cur:
                    continue
            w = float(font.text_length(ch, fontsize=float(fontsize)))
            if (cur_w + w) > rect.width and cur:
                flush()
            if ch.isspace():
                cur.append(_SpacePiece(width=w))
            else:
                if cur and isinstance(cur[-1], _TextPiece):
                    prev = cur[-1]
                    cur[-1] = _TextPiece(text=prev.text + ch, width=prev.width + w)
                else:
                    cur.append(_TextPiece(text=ch, width=w))
            cur_w += w

    if cur:
        flush()

    if not lines:
        lines = [[]]

    total_h = ascent + descent + max(0, len(lines) - 1) * line_height
    if total_h > rect.height + 1e-3:
        return None

    return _Layout(fontsize=float(fontsize), ascent=ascent, descent=descent, line_height=line_height, lines=lines)


def _choose_layout(
    *,
    text: str,
    rect: fitz.Rect,
    font: fitz.Font,
    svg_map: dict[tuple[str, bool], str],
    max_size: float,
    min_size: float,
    math_scale: float,
    math_inline_cap: float,
) -> _Layout | None:
    lo = float(min_size)
    hi = float(max_size)
    best: _Layout | None = None
    for _ in range(10):
        mid = (lo + hi) / 2.0
        layout = _layout_sentence(
            text=text,
            rect=rect,
            font=font,
            fontsize=mid,
            svg_map=svg_map,
            math_scale=math_scale,
            math_inline_cap=math_inline_cap,
        )
        if layout is not None:
            best = layout
            lo = mid
        else:
            hi = mid
    return best


def _render_layout(
    page: fitz.Page,
    *,
    rect: fitz.Rect,
    layout: _Layout,
    fontfile: Path,
    fontname: str,
    color: tuple[float, float, float],
    align: int,
    math_docs: dict[tuple[str, bool], fitz.Document],
) -> None:
    for line_index, line in enumerate(layout.lines):
        baseline_y = rect.y0 + layout.ascent + (line_index * layout.line_height)

        # Compute width and justify spacing.
        line_w = 0.0
        spaces = 0
        for p in line:
            if isinstance(p, _TextPiece):
                line_w += p.width
            elif isinstance(p, _SpacePiece):
                line_w += p.width
                spaces += 1
            else:
                line_w += p.width

        is_last = line_index == (len(layout.lines) - 1)
        extra_per_space = 0.0
        if int(align) == fitz.TEXT_ALIGN_JUSTIFY and (not is_last) and spaces > 0:
            extra = max(0.0, rect.width - line_w)
            extra_per_space = extra / float(spaces)

        # Center block-math-only lines.
        if (
            len(line) == 1
            and isinstance(line[0], _MathPiece)
            and bool(line[0].display)
            and line[0].width < rect.width
        ):
            x = rect.x0 + (rect.width - line[0].width) / 2.0
        else:
            x = rect.x0

        for p in line:
            if isinstance(p, _TextPiece):
                page.insert_text(
                    fitz.Point(x, baseline_y),
                    p.text,
                    fontsize=float(layout.fontsize),
                    fontname=fontname,
                    fontfile=str(fontfile),
                    color=color,
                    overlay=True,
                )
                x += p.width
            elif isinstance(p, _SpacePiece):
                x += p.width + extra_per_space
            else:
                math_doc = math_docs.get(p.key)
                if math_doc is None:
                    x += p.width
                    continue
                # CSS: negative vertical-align lowers the box (PDF y-axis downward),
                # so shift by -valign.
                top = baseline_y - p.height - p.valign
                mrect = fitz.Rect(x, top, x + p.width, top + p.height)
                page.show_pdf_page(mrect, math_doc, 0, overlay=True)
                x += p.width


def render_korean_pdf(
    *,
    pdf_path: str | Path,
    sentences: list[Sentence],
    translations: dict[str, str],
    font_path: str | Path,
    output_path: str | Path,
    whiteout: bool = True,
    padding: float = 1.0,
    align: int = fitz.TEXT_ALIGN_JUSTIFY,
    math_scale: float = 1.15,
    math_inline_cap: float = 1.0,
) -> RenderStats:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = normalize_path(str(font_path)).resolve()

    tex_spans = iter_tex_spans(translations.values())
    svg_map = tex_to_svg_batch(tex_spans, math_scale=float(math_scale)) if tex_spans else {}
    math_docs: dict[tuple[str, bool], fitz.Document] = {}
    if svg_map:
        math_docs = _build_math_docs(svg_map)

    by_page: dict[int, list[Sentence]] = {}
    for s in sentences:
        by_page.setdefault(int(s.page), []).append(s)

    doc = fitz.open(pdf_path)
    inserted = 0
    skipped = 0
    overflow = 0

    try:
        font = fitz.Font(fontfile=str(font_path))
        for page_index in range(doc.page_count):
            page_num = page_index + 1
            page_sentences = by_page.get(page_num, [])
            if not page_sentences:
                continue

            page = doc.load_page(page_index)
            page_rect = page.rect

            json_w = float(page_sentences[0].page_width or 0)
            json_h = float(page_sentences[0].page_height or 0)
            if not json_w or not json_h:
                continue
            scale_x = page_rect.width / json_w
            scale_y = page_rect.height / json_h

            for s in page_sentences:
                if str(s.meta.get("kind") or "text") != "text":
                    skipped += 1
                    continue
                text = translations.get(s.sentence_id, "").strip()
                if not text:
                    skipped += 1
                    continue

                r = _region_to_rect(s.region, scale_x=scale_x, scale_y=scale_y, padding=padding)
                if r.is_empty or r.is_infinite or r.width <= 2 or r.height <= 2:
                    skipped += 1
                    continue

                if whiteout:
                    page.draw_rect(
                        r,
                        color=None,
                        fill=(1, 1, 1),
                        width=0,
                        overlay=True,
                        fill_opacity=1,
                    )

                # Heuristic font size bounds based on box height.
                max_size = min(24.0, max(6.0, r.height * 0.9))
                min_size = max(4.0, min(8.0, r.height * 0.35))

                layout = _choose_layout(
                    text=text,
                    rect=r,
                    font=font,
                    svg_map=svg_map,
                    max_size=max_size,
                    min_size=min_size,
                    math_scale=float(math_scale),
                    math_inline_cap=float(math_inline_cap),
                )
                inserted += 1
                if layout is None:
                    overflow += 1
                    continue

                _render_layout(
                    page,
                    rect=r,
                    layout=layout,
                    fontfile=font_path,
                    fontname="mptr-font",
                    color=(0, 0, 0),
                    align=int(align),
                    math_docs=math_docs,
                )

        doc.save(output_path, garbage=4, deflate=True)
    finally:
        for d in math_docs.values():
            try:
                d.close()
            except Exception:
                pass
        doc.close()

    return RenderStats(inserted=inserted, skipped=skipped, overflow=overflow)


def render_side_by_side_pdf(
    *,
    pdf_path: str | Path,
    sentences: list[Sentence],
    translations: dict[str, str],
    font_path: str | Path,
    output_path: str | Path,
    padding: float = 1.0,
    separator: bool = True,
    align: int = fitz.TEXT_ALIGN_JUSTIFY,
    math_scale: float = 1.15,
    math_inline_cap: float = 1.0,
) -> RenderStats:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = normalize_path(str(font_path)).resolve()

    tex_spans = iter_tex_spans(translations.values())
    svg_map = tex_to_svg_batch(tex_spans, math_scale=float(math_scale)) if tex_spans else {}
    math_docs: dict[tuple[str, bool], fitz.Document] = {}
    if svg_map:
        math_docs = _build_math_docs(svg_map)

    by_page: dict[int, list[Sentence]] = {}
    for s in sentences:
        by_page.setdefault(int(s.page), []).append(s)

    src = fitz.open(pdf_path)
    out = fitz.open()
    inserted = 0
    skipped = 0
    overflow = 0

    try:
        font = fitz.Font(fontfile=str(font_path))
        for page_index in range(src.page_count):
            src_page = src.load_page(page_index)
            w = float(src_page.rect.width)
            h = float(src_page.rect.height)
            new_page = out.new_page(width=w * 2.0, height=h)

            # Left: original page as vector.
            new_page.show_pdf_page(fitz.Rect(0, 0, w, h), src, page_index)
            # Right: original page duplicated (so figures / diagrams remain visible).
            new_page.show_pdf_page(fitz.Rect(w, 0, w * 2.0, h), src, page_index)

            if separator:
                new_page.draw_line(
                    fitz.Point(w, 0),
                    fitz.Point(w, h),
                    color=(0.8, 0.8, 0.8),
                    width=0.5,
                    overlay=True,
                )

            page_num = page_index + 1
            page_sentences = by_page.get(page_num, [])
            if not page_sentences:
                continue

            json_w = float(page_sentences[0].page_width or 0)
            json_h = float(page_sentences[0].page_height or 0)
            if not json_w or not json_h:
                continue
            scale_x = w / json_w
            scale_y = h / json_h

            for s in page_sentences:
                if str(s.meta.get("kind") or "text") != "text":
                    skipped += 1
                    continue
                text = translations.get(s.sentence_id, "").strip()
                if not text:
                    skipped += 1
                    continue

                r = _region_to_rect(
                    s.region,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    x_offset=w,
                    padding=padding,
                )
                if r.is_empty or r.is_infinite or r.width <= 2 or r.height <= 2:
                    skipped += 1
                    continue

                # Whiteout original text on the right copy, then add translated text.
                new_page.draw_rect(
                    r,
                    color=None,
                    fill=(1, 1, 1),
                    width=0,
                    overlay=True,
                    fill_opacity=1,
                )

                max_size = min(24.0, max(6.0, r.height * 0.9))
                min_size = max(4.0, min(8.0, r.height * 0.35))

                layout = _choose_layout(
                    text=text,
                    rect=r,
                    font=font,
                    svg_map=svg_map,
                    max_size=max_size,
                    min_size=min_size,
                    math_scale=float(math_scale),
                    math_inline_cap=float(math_inline_cap),
                )
                inserted += 1
                if layout is None:
                    overflow += 1
                    continue

                _render_layout(
                    new_page,
                    rect=r,
                    layout=layout,
                    fontfile=font_path,
                    fontname="mptr-font",
                    color=(0, 0, 0),
                    align=int(align),
                    math_docs=math_docs,
                )

        out.save(output_path, garbage=4, deflate=True)
    finally:
        out.close()
        for d in math_docs.values():
            try:
                d.close()
            except Exception:
                pass
        src.close()

    return RenderStats(inserted=inserted, skipped=skipped, overflow=overflow)
