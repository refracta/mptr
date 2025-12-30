from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from sentences import Sentence
from util import normalize_path


@dataclass(frozen=True)
class RenderStats:
    inserted: int
    skipped: int
    overflow: int


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


def _fits_text(*, text: str, rect: fitz.Rect, fontname: str, fontfile: Path, fontsize: float) -> bool:
    doc = fitz.open()
    try:
        page = doc.new_page(width=max(1.0, rect.width), height=max(1.0, rect.height))
        test_rect = fitz.Rect(0, 0, rect.width, rect.height)
        res = page.insert_textbox(
            test_rect,
            text,
            fontname=fontname,
            fontfile=str(fontfile),
            fontsize=float(fontsize),
            overlay=True,
        )
        return res >= 0
    finally:
        doc.close()


def _choose_font_size(
    *,
    text: str,
    rect: fitz.Rect,
    fontname: str,
    fontfile: Path,
    max_size: float,
    min_size: float,
) -> float:
    lo = float(min_size)
    hi = float(max_size)
    best = lo
    for _ in range(10):
        mid = (lo + hi) / 2.0
        if _fits_text(text=text, rect=rect, fontname=fontname, fontfile=fontfile, fontsize=mid):
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def render_korean_pdf(
    *,
    pdf_path: str | Path,
    sentences: list[Sentence],
    translations: dict[str, str],
    font_path: str | Path,
    output_path: str | Path,
    whiteout: bool = True,
    padding: float = 1.0,
) -> RenderStats:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = normalize_path(str(font_path)).resolve()

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
                fontname = "mptr-font"
                fontsize = _choose_font_size(
                    text=text,
                    rect=r,
                    fontname=fontname,
                    fontfile=font_path,
                    max_size=max_size,
                    min_size=min_size,
                )

                res = page.insert_textbox(
                    r,
                    text,
                    fontname=fontname,
                    fontfile=str(font_path),
                    fontsize=float(fontsize),
                    color=(0, 0, 0),
                    overlay=True,
                )
                inserted += 1
                if res < 0:
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
) -> RenderStats:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = normalize_path(str(font_path)).resolve()

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

                max_size = min(24.0, max(6.0, r.height * 0.9))
                min_size = max(4.0, min(8.0, r.height * 0.35))
                fontname = "mptr-font"
                fontsize = _choose_font_size(
                    text=text,
                    rect=r,
                    fontname=fontname,
                    fontfile=font_path,
                    max_size=max_size,
                    min_size=min_size,
                )
                res = new_page.insert_textbox(
                    r,
                    text,
                    fontname=fontname,
                    fontfile=str(font_path),
                    fontsize=float(fontsize),
                    color=(0, 0, 0),
                    overlay=True,
                )
                inserted += 1
                if res < 0:
                    overflow += 1

        out.save(output_path, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()

    return RenderStats(inserted=inserted, skipped=skipped, overflow=overflow)

