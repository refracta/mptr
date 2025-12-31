from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from util import normalize_path, write_json

DEFAULT_BASE_URL = "https://api.mathpix.com/v3/pdf"

CONVERSION_FORMATS = {
    "md": True,
    "docx": True,
    "tex.zip": True,
    "html": True,
    "html.zip": True,
    "pdf": True,
    "latex.pdf": True,
    "pptx": True,
    "mmd.zip": True,
    "md.zip": True,
}

DOWNLOAD_SUFFIXES = [
    "html",
    "html.zip",
    "mmd",
    "md",
    "docx",
    "tex",
    "tex.zip",
    "pptx",
    "mmd.zip",
    "md.zip",
    "latex.pdf",
    "pdf",
    "lines.json",
    "lines.mmd.json",
]


@dataclass(frozen=True)
class MathpixCreds:
    app_id: str
    app_key: str


def _looks_like_math_only_text(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    # If it already contains TeX delimiters, treat as math-like.
    if "\\(" in t or "\\[" in t:
        return True

    # If it contains any "long" alphabetic word, assume it's normal prose.
    if re.search(r"[A-Za-z]{3,}", t):
        return False

    # Otherwise, if it contains common math operators/symbols (or looks like a formula),
    # treat as math-only to avoid translating it.
    if re.search(r"[=<>±×÷∑∫∂∇√≈≠≤≥∞]", t):
        return True
    if re.search(r"\b\d+(\.\d+)?\b", t) and re.search(r"[+\-*/^]", t):
        return True
    return False


def _lines_json_from_pdf_text(pdf_path: Path) -> dict:
    # Local fallback: create a minimal Mathpix-like lines.json using PyMuPDF text extraction.
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages: list[dict] = []
    try:
        for page_index in range(doc.page_count):
            page_num = page_index + 1
            page = doc.load_page(page_index)
            page_rect = page.rect
            page_w = float(page_rect.width)
            page_h = float(page_rect.height)

            data = page.get_text("dict")
            raw_lines: list[dict] = []
            line_idx = 0
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    bbox = line.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    spans = line.get("spans", []) or []
                    text = "".join(str(s.get("text") or "") for s in spans).strip()
                    if not text:
                        continue
                    raw_lines.append(
                        {
                            "id": f"p{page_num}-l{line_idx}",
                            "text": text,
                            "bbox": (x0, y0, x1, y1),
                            "region": {
                                "top_left_x": x0,
                                "top_left_y": y0,
                                "width": x1 - x0,
                                "height": y1 - y0,
                            },
                            "type": "math" if _looks_like_math_only_text(text) else "text",
                        }
                    )
                    line_idx += 1

            # Column assignment (best-effort): detect 2-column layouts.
            mid = page_w / 2.0
            for it in raw_lines:
                x0, _, x1, _ = it["bbox"]
                it["full_width"] = (x0 < page_w * 0.25 and x1 > page_w * 0.75) or ((x1 - x0) > page_w * 0.8)
                it["center_x"] = (x0 + x1) / 2.0

            non_full = [it for it in raw_lines if not it["full_width"]]
            left = [it for it in non_full if float(it["center_x"]) < mid]
            right = [it for it in non_full if float(it["center_x"]) >= mid]
            two_col = len(left) >= 10 and len(right) >= 10

            for it in raw_lines:
                if it["full_width"]:
                    it["column"] = -1
                elif two_col:
                    it["column"] = 0 if float(it["center_x"]) < mid else 1
                else:
                    it["column"] = 0

            pages.append(
                {
                    "page": page_num,
                    "page_width": page_w,
                    "page_height": page_h,
                    "lines": [
                        {
                            "id": it["id"],
                            "type": it["type"],
                            "text": it["text"],
                            "region": it["region"],
                            "column": int(it.get("column") or 0),
                        }
                        for it in raw_lines
                    ],
                }
            )
    finally:
        doc.close()

    return {"pages": pages}


def _extract_default_creds_from_convert_py(convert_py: Path) -> MathpixCreds | None:
    try:
        text = convert_py.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None

    # Compatible with the user's existing convert.py.
    m_id = re.search(r'APP_ID\s*=\s*os\.getenv\("MATHPIX_APP_ID",\s*"([^"]+)"\)', text)
    m_key = re.search(r'APP_KEY\s*=\s*os\.getenv\("MATHPIX_APP_KEY",\s*"([^"]+)"\)', text)
    if not m_id or not m_key:
        return None
    return MathpixCreds(app_id=m_id.group(1), app_key=m_key.group(1))


def load_mathpix_creds(config: dict) -> MathpixCreds:
    app_id = (config.get("mathpix_app_id") or "").strip()
    app_key = (config.get("mathpix_app_key") or "").strip()
    if app_id and app_key:
        return MathpixCreds(app_id=app_id, app_key=app_key)

    env_id = (os.getenv("MATHPIX_APP_ID") or "").strip()
    env_key = (os.getenv("MATHPIX_APP_KEY") or "").strip()
    if env_id and env_key:
        return MathpixCreds(app_id=env_id, app_key=env_key)

    fallback = config.get("mathpix_fallback_convert_py")
    if fallback:
        fb_path = normalize_path(str(fallback))
        if not fb_path.is_absolute():
            fb_path = (Path(__file__).resolve().parent / fb_path).resolve()
        else:
            fb_path = fb_path.resolve()
        creds = _extract_default_creds_from_convert_py(fb_path)
        if creds:
            return creds

    raise RuntimeError(
        "Missing Mathpix credentials. Set config.json mathpix_app_id/mathpix_app_key, "
        "or export MATHPIX_APP_ID/MATHPIX_APP_KEY, or provide mathpix_fallback_convert_py."
    )


def _submit_pdf(pdf_path: Path, *, creds: MathpixCreds, base_url: str) -> str:
    options = {"conversion_formats": CONVERSION_FORMATS}
    with pdf_path.open("rb") as f:
        resp = requests.post(
            base_url,
            headers={"app_id": creds.app_id, "app_key": creds.app_key},
            data={"options_json": json.dumps(options)},
            files={"file": f},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    pdf_id = data.get("pdf_id")
    if not pdf_id:
        raise RuntimeError(f"No pdf_id returned: {data}")
    return str(pdf_id)


def _wait_for_completion(
    pdf_id: str,
    *,
    creds: MathpixCreds,
    base_url: str,
    timeout_s: int,
    poll_interval_s: int,
) -> None:
    start = time.time()
    while True:
        resp = requests.get(
            f"{base_url}/{pdf_id}",
            headers={"app_id": creds.app_id, "app_key": creds.app_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "completed":
            return
        if status == "error":
            raise RuntimeError(f"Mathpix processing failed: {data}")
        if (time.time() - start) > timeout_s:
            raise TimeoutError("Timed out waiting for Mathpix.")
        time.sleep(poll_interval_s)


def _download(url: str, dest: Path, *, creds: MathpixCreds) -> bool:
    resp = requests.get(
        url,
        headers={"app_id": creds.app_id, "app_key": creds.app_key},
        stream=True,
        timeout=120,
    )
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return True


def conversion_dir_for(pdf_path: Path, *, base_dir: Path) -> Path:
    # folder name includes extension: e.g. conversion/RISE.pdf/
    return base_dir / pdf_path.name


def ensure_converted(
    pdf_path: str | Path,
    *,
    base_dir: str | Path,
    config: dict,
    force: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = 3600,
    poll_interval_s: int = 5,
) -> Path:
    pdf_path = normalize_path(str(pdf_path)).resolve()
    base_dir = normalize_path(str(base_dir)).resolve()
    out_dir = conversion_dir_for(pdf_path, base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    lines_path = out_dir / f"{stem}.lines.json"
    if lines_path.exists() and not force:
        return out_dir

    try:
        creds = load_mathpix_creds(config)
    except Exception:
        # If Mathpix creds are missing, fall back to local text extraction for selectable-text PDFs.
        lines_json = _lines_json_from_pdf_text(pdf_path)
        total_chars = sum(len(str(l.get("text") or "")) for p in lines_json.get("pages", []) for l in p.get("lines", []))
        total_lines = sum(len(p.get("lines", [])) for p in lines_json.get("pages", []))
        if total_chars < 20 and total_lines < 5:
            raise
        write_json(lines_path, lines_json)
        write_json(
            out_dir / "local.meta.json",
            {
                "source_pdf": str(pdf_path),
                "converter": "pymupdf-text-fallback",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        return out_dir

    pdf_id = _submit_pdf(pdf_path, creds=creds, base_url=base_url)
    _wait_for_completion(
        pdf_id,
        creds=creds,
        base_url=base_url,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )

    for suffix in DOWNLOAD_SUFFIXES:
        dest = out_dir / f"{stem}.{suffix}"
        _download(f"{base_url}/{pdf_id}.{suffix}", dest, creds=creds)

    write_json(
        out_dir / "mathpix.meta.json",
        {
            "pdf_id": pdf_id,
            "base_url": base_url,
            "source_pdf": str(pdf_path),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return out_dir
