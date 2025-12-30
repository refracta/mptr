from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from sentences import Sentence
from util import normalize_path


def export_pdf_pages_to_png(
    *,
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 144,
) -> list[Path]:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    out_dir = normalize_path(str(out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    out_paths: list[Path] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=int(dpi))
            out_path = out_dir / f"page-{i + 1:03d}.png"
            pix.save(out_path)
            out_paths.append(out_path)
    finally:
        doc.close()
    return out_paths


def whiteout_sentence_regions(
    *,
    page_png: str | Path,
    sentences: list[Sentence],
    output_png: str | Path,
) -> Path:
    page_png = normalize_path(str(page_png)).resolve()
    output_png = normalize_path(str(output_png)).resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(img)

    if not sentences:
        img.save(output_png)
        return output_png

    page_w_px, page_h_px = img.size
    json_w = float(sentences[0].page_width or 1.0)
    json_h = float(sentences[0].page_height or 1.0)
    sx = page_w_px / json_w
    sy = page_h_px / json_h

    for s in sentences:
        r = s.region
        x0 = int(r["top_left_x"] * sx)
        y0 = int(r["top_left_y"] * sy)
        x1 = int((r["top_left_x"] + r["width"]) * sx)
        y1 = int((r["top_left_y"] + r["height"]) * sy)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

    img.save(output_png)
    return output_png

