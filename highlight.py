from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from sentences import Sentence
from util import normalize_path


def _hash_color(key: str) -> tuple[float, float, float]:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    sat = 0.45
    val = 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (r, g, b)


def _region_to_rect(region: dict, *, scale_x: float, scale_y: float) -> fitz.Rect:
    x0 = float(region["top_left_x"]) * scale_x
    y0 = float(region["top_left_y"]) * scale_y
    x1 = (float(region["top_left_x"]) + float(region["width"])) * scale_x
    y1 = (float(region["top_left_y"]) + float(region["height"])) * scale_y
    return fitz.Rect(x0, y0, x1, y1)


def _add_rect(page: fitz.Page, rect: fitz.Rect, *, color: tuple[float, float, float], opacity: float) -> None:
    annot = page.add_rect_annot(rect)
    annot.set_border(width=0)
    annot.set_colors(stroke=color, fill=color)
    annot.set_opacity(max(0.0, min(1.0, float(opacity))))
    annot.update()


def create_sentence_highlight_pdf(
    *,
    pdf_path: str | Path,
    sentences: list[Sentence],
    output_path: str | Path,
    opacity: float = 0.25,
) -> Path:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    output_path = normalize_path(str(output_path)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Index by page
    by_page: dict[int, list[Sentence]] = {}
    for s in sentences:
        by_page.setdefault(int(s.page), []).append(s)

    doc = fitz.open(pdf_path)
    try:
        for page_index in range(doc.page_count):
            page_num = page_index + 1
            page_sentences = by_page.get(page_num, [])
            if not page_sentences:
                continue

            page = doc.load_page(page_index)
            rect = page.rect
            json_w = float(page_sentences[0].page_width or 0)
            json_h = float(page_sentences[0].page_height or 0)
            if not json_w or not json_h:
                continue

            scale_x = rect.width / float(json_w)
            scale_y = rect.height / float(json_h)

            for s in page_sentences:
                color = _hash_color(s.sentence_id)
                r = _region_to_rect(s.region, scale_x=scale_x, scale_y=scale_y)
                if r.is_empty or r.is_infinite:
                    continue
                if r.x1 <= 0 or r.y1 <= 0 or r.x0 >= rect.width or r.y0 >= rect.height:
                    continue
                _add_rect(page, r, color=color, opacity=opacity)

        doc.save(output_path, garbage=4, deflate=True)
    finally:
        doc.close()

    return output_path
