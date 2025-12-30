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
        creds = _extract_default_creds_from_convert_py(normalize_path(str(fallback)).resolve())
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

    creds = load_mathpix_creds(config)
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

