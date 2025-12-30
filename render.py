from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from latex import TexSpan, build_html_with_math, iter_tex_spans, split_text_with_tex, tex_to_svg_batch
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


def _fits_html(*, html_doc: str, css: str, archive: fitz.Archive, rect: fitz.Rect) -> bool:
    doc = fitz.open()
    try:
        page = doc.new_page(width=max(1.0, rect.width), height=max(1.0, rect.height))
        test_rect = fitz.Rect(0, 0, rect.width, rect.height)
        spare, _scale = page.insert_htmlbox(
            test_rect,
            html_doc,
            css=css,
            scale_low=1,
            archive=archive,
            overlay=True,
        )
        return spare >= 0
    finally:
        doc.close()


def _choose_html_font_size(
    *,
    html_body: str,
    css: str,
    archive: fitz.Archive,
    rect: fitz.Rect,
    max_size: float,
    min_size: float,
) -> float:
    lo = float(min_size)
    hi = float(max_size)
    best = lo
    # Binary search for the largest font size that fits without scaling.
    for _ in range(10):
        mid = (lo + hi) / 2.0
        html_doc = f'<div class="mptr" style="font-size:{mid:.4f}pt">{html_body}</div>'
        if _fits_html(html_doc=html_doc, css=css, archive=archive, rect=rect):
            best = mid
            lo = mid
        else:
            hi = mid
    return best


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
) -> RenderStats:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = normalize_path(str(font_path)).resolve()

    # Prepare TeX→SVG map once per render call.
    tex_spans = iter_tex_spans(translations.values())
    svg_map = tex_to_svg_batch(tex_spans, math_scale=float(math_scale)) if tex_spans else {}

    font_archive = fitz.Archive()
    font_archive.add((font_path.read_bytes(), "mptr-target.ttf"))
    css_align = _css_text_align(align)
    css = (
        '@font-face { font-family: MptrTarget; src: url("mptr-target.ttf"); }\n'
        f".mptr {{ font-family: MptrTarget; line-height: 1.2; white-space: pre-wrap; text-align: {css_align}; }}\n"
        ".mptr { word-break: break-all; overflow-wrap: anywhere; }\n"
        ".mptr svg { vertical-align: -0.2ex; }\n"
        ".mptr-math-block { text-align: center; margin: 0.2em 0; }\n"
    )

    by_page: dict[int, list[Sentence]] = {}
    for s in sentences:
        by_page.setdefault(int(s.page), []).append(s)

    doc = fitz.open(pdf_path)
    inserted = 0
    skipped = 0
    overflow = 0

    try:
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

                html_body = (
                    build_html_with_math(text, svg_map)
                    if _has_tex(text)
                    else html.escape(text)
                )
                fontsize = _choose_html_font_size(
                    html_body=html_body,
                    css=css,
                    archive=font_archive,
                    rect=r,
                    max_size=max_size,
                    min_size=min_size,
                )
                html_doc = f'<div class="mptr" style="font-size:{fontsize:.4f}pt">{html_body}</div>'
                spare, _scale = page.insert_htmlbox(
                    r,
                    html_doc,
                    css=css,
                    scale_low=1,
                    archive=font_archive,
                    overlay=True,
                )
                inserted += 1
                if spare < 0:
                    # Fallback: force-fit by scaling down below min_size if needed.
                    html_doc_min = f'<div class="mptr" style="font-size:{min_size:.4f}pt">{html_body}</div>'
                    spare2, _scale2 = page.insert_htmlbox(
                        r,
                        html_doc_min,
                        css=css,
                        scale_low=0,
                        archive=font_archive,
                        overlay=True,
                    )
                    if spare2 < 0:
                        overflow += 1

        doc.save(output_path, garbage=4, deflate=True)
    finally:
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
) -> RenderStats:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = normalize_path(str(font_path)).resolve()

    tex_spans = iter_tex_spans(translations.values())
    svg_map = tex_to_svg_batch(tex_spans, math_scale=float(math_scale)) if tex_spans else {}

    font_archive = fitz.Archive()
    font_archive.add((font_path.read_bytes(), "mptr-target.ttf"))
    css_align = _css_text_align(align)
    css = (
        '@font-face { font-family: MptrTarget; src: url("mptr-target.ttf"); }\n'
        f".mptr {{ font-family: MptrTarget; line-height: 1.2; white-space: pre-wrap; text-align: {css_align}; }}\n"
        ".mptr { word-break: break-all; overflow-wrap: anywhere; }\n"
        ".mptr svg { vertical-align: -0.2ex; }\n"
        ".mptr-math-block { text-align: center; margin: 0.2em 0; }\n"
    )

    by_page: dict[int, list[Sentence]] = {}
    for s in sentences:
        by_page.setdefault(int(s.page), []).append(s)

    src = fitz.open(pdf_path)
    out = fitz.open()
    inserted = 0
    skipped = 0
    overflow = 0

    try:
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

                html_body = (
                    build_html_with_math(text, svg_map)
                    if _has_tex(text)
                    else html.escape(text)
                )
                fontsize = _choose_html_font_size(
                    html_body=html_body,
                    css=css,
                    archive=font_archive,
                    rect=r,
                    max_size=max_size,
                    min_size=min_size,
                )
                html_doc = f'<div class="mptr" style="font-size:{fontsize:.4f}pt">{html_body}</div>'
                spare, _scale = new_page.insert_htmlbox(
                    r,
                    html_doc,
                    css=css,
                    scale_low=1,
                    archive=font_archive,
                    overlay=True,
                )
                inserted += 1
                if spare < 0:
                    html_doc_min = f'<div class="mptr" style="font-size:{min_size:.4f}pt">{html_body}</div>'
                    spare2, _scale2 = new_page.insert_htmlbox(
                        r,
                        html_doc_min,
                        css=css,
                        scale_low=0,
                        archive=font_archive,
                        overlay=True,
                    )
                    if spare2 < 0:
                        overflow += 1

        out.save(output_path, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()

    return RenderStats(inserted=inserted, skipped=skipped, overflow=overflow)
