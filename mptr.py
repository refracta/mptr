#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

import fitz  # PyMuPDF

from highlight import create_sentence_highlight_pdf
from images import export_pdf_pages_to_png, whiteout_sentence_regions
from mathpix import ensure_converted
from render import render_korean_pdf, render_side_by_side_pdf
from sentences import Sentence, sentences_from_lines_json
from translator import translate_sentences
from util import load_config, normalize_path, read_json, render_prompt_template, write_json


def _resolve_font_path(*, config: dict, repo_dir: Path) -> Path:
    raw = str(config.get("target_font") or "").strip()
    if not raw:
        raise RuntimeError("Missing target_font in config.json")

    # If it's a path, use it.
    if any(sep in raw for sep in ("/", "\\", ":")):
        p = normalize_path(raw).resolve()
        if p.exists():
            return p
    # Otherwise, treat as filename under fonts/
    p = (repo_dir / "fonts" / raw).resolve()
    if p.exists():
        return p

    raise FileNotFoundError(
        f"Target font not found: {raw}. Copy it into {repo_dir / 'fonts'} or set MPTR_target_font to a full path."
    )


def _load_prompt(*, config: dict, repo_dir: Path) -> str:
    prompt_name = str(config.get("target_prompt") or "translate.txt")
    prompt_path = repo_dir / "prompts" / prompt_name
    template = prompt_path.read_text(encoding="utf-8")
    return render_prompt_template(template, config)


def _resolve_align(config: dict) -> int:
    raw = str(config.get("render_align") or "justify").strip().lower()
    mapping = {
        "left": fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right": fitz.TEXT_ALIGN_RIGHT,
        "justify": fitz.TEXT_ALIGN_JUSTIFY,
    }
    return int(mapping.get(raw, fitz.TEXT_ALIGN_JUSTIFY))


def _next_run_dir(base: Path, docname: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base / f"{docname}.{n}.pdf"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        n += 1


def _docname_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    parts = name.split(".")
    if len(parts) >= 2 and parts[-1].isdigit():
        return ".".join(parts[:-1])
    return name


def _save_sentences(path: Path, sentences: list[Sentence]) -> None:
    write_json(
        path,
        [
            {
                "sentence_id": s.sentence_id,
                "page": s.page,
                "column": s.column,
                "page_width": s.page_width,
                "page_height": s.page_height,
                "region": s.region,
                "source_text": s.source_text,
                "line_ids": s.line_ids,
                "meta": s.meta,
            }
            for s in sentences
        ],
    )


def _load_sentences(path: Path) -> list[Sentence]:
    items = read_json(path)
    if not isinstance(items, list):
        raise ValueError(f"Invalid sentences file: {path}")
    out: list[Sentence] = []
    for it in items:
        out.append(
            Sentence(
                sentence_id=it["sentence_id"],
                page=int(it["page"]),
                column=int(it.get("column") or 0),
                page_width=float(it.get("page_width") or 1.0),
                page_height=float(it.get("page_height") or 1.0),
                region=it["region"],
                source_text=str(it.get("source_text") or ""),
                line_ids=list(it.get("line_ids") or []),
                meta=dict(it.get("meta") or {}),
            )
        )
    return out


def _extract_translations(cache_path: Path) -> dict[str, str]:
    cache = read_json(cache_path)
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid translation cache: {cache_path}")
    return {k: str(v.get("translated_text") or "") for k, v in cache.items() if isinstance(v, dict)}


def cmd_convert(args: argparse.Namespace) -> int:
    repo_dir = Path(__file__).resolve().parent
    config = load_config(normalize_path(args.config).resolve())
    pdf_path = normalize_path(args.pdf).resolve()
    ensure_converted(pdf_path, base_dir=repo_dir / "conversion", config=config, force=args.force)
    print("OK")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    repo_dir = Path(__file__).resolve().parent
    config_path = normalize_path(args.config).resolve()
    config = load_config(config_path)

    pdf_path = normalize_path(args.pdf).resolve()
    docname = pdf_path.stem

    # 1) Mathpix conversion (cached)
    conv_dir = ensure_converted(pdf_path, base_dir=repo_dir / "conversion", config=config, force=args.force_convert)

    # 2) Sentence grouping + highlight
    lines_json_path = conv_dir / f"{docname}.lines.json"
    lines_json = read_json(lines_json_path)
    sentences = sentences_from_lines_json(
        lines_json,
        indent_ratio=float(config.get("sentence_indent_ratio") or 0.015),
        spacing_factor=float(config.get("sentence_spacing_factor") or 1.6),
        min_text_len=int(config.get("sentence_min_text_len") or 1),
        include_math=bool(config.get("include_math_lines", True)),
    )

    run_dir = _next_run_dir(repo_dir / "translation", docname)
    artifacts_dir = run_dir / "artifacts"
    translations_dir = artifacts_dir / "translations"
    translations_dir.mkdir(parents=True, exist_ok=True)
    images_pages_dir = artifacts_dir / "images" / "pages"
    images_whiteout_dir = artifacts_dir / "images" / "whiteout"
    images_pages_dir.mkdir(parents=True, exist_ok=True)
    images_whiteout_dir.mkdir(parents=True, exist_ok=True)

    # Copy original
    shutil.copy2(pdf_path, run_dir / pdf_path.name)

    # Save resolved config for reproducibility
    write_json(artifacts_dir / "config_resolved.json", config)
    _save_sentences(artifacts_dir / "sentences.json", sentences)

    highlight_path = run_dir / f"{docname}.highlight.pdf"
    create_sentence_highlight_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        output_path=highlight_path,
        opacity=float(config.get("highlight_opacity") or 0.25),
    )

    # 3) Export page images + whiteout artifacts (for qualitative inspection)
    page_pngs = export_pdf_pages_to_png(
        pdf_path=pdf_path,
        out_dir=images_pages_dir,
        dpi=int(config.get("render_dpi") or 144),
    )
    by_page = {}
    for s in sentences:
        by_page.setdefault(int(s.page), []).append(s)
    for i, png in enumerate(page_pngs, start=1):
        whiteout_sentence_regions(
            page_png=png,
            sentences=by_page.get(i, []),
            output_png=images_whiteout_dir / png.name,
        )

    # 4-5) Translate sentences (cached)
    instructions = _load_prompt(config=config, repo_dir=repo_dir)
    tr_results = asyncio.run(
        translate_sentences(
            sentences,
            config=config,
            instructions=instructions,
            out_dir=translations_dir,
        )
    )
    translations = {k: v.translated_text for k, v in tr_results.items()}

    # 6) Render translation PDFs
    font_path = _resolve_font_path(config=config, repo_dir=repo_dir)
    align = _resolve_align(config)
    math_scale = float(config.get("math_scale") or 1.15)
    korean_out = run_dir / f"{docname}.korean.pdf"
    both_out = run_dir / f"{docname}.korean.both.pdf"

    stats_kr = render_korean_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        translations=translations,
        font_path=font_path,
        output_path=korean_out,
        whiteout=True,
        padding=float(config.get("render_padding") or 1.0),
        align=align,
        math_scale=math_scale,
    )
    stats_both = render_side_by_side_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        translations=translations,
        font_path=font_path,
        output_path=both_out,
        padding=float(config.get("render_padding") or 1.0),
        separator=True,
        align=align,
        math_scale=math_scale,
    )

    write_json(
        artifacts_dir / "render_stats.json",
        {
            "korean": stats_kr.__dict__,
            "both": stats_both.__dict__,
            "sentences": len(sentences),
        },
    )

    print(f"Run folder: {run_dir}")
    print(f"Korean PDF: {korean_out}")
    print(f"Both PDF: {both_out}")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    repo_dir = Path(__file__).resolve().parent
    config_path = normalize_path(args.config).resolve()
    config = load_config(config_path)

    pdf_path = normalize_path(args.pdf).resolve()
    docname = pdf_path.stem

    conv_dir = ensure_converted(pdf_path, base_dir=repo_dir / "conversion", config=config, force=args.force_convert)

    lines_json_path = conv_dir / f"{docname}.lines.json"
    lines_json = read_json(lines_json_path)
    sentences = sentences_from_lines_json(
        lines_json,
        indent_ratio=float(config.get("sentence_indent_ratio") or 0.015),
        spacing_factor=float(config.get("sentence_spacing_factor") or 1.6),
        min_text_len=int(config.get("sentence_min_text_len") or 1),
        include_math=bool(config.get("include_math_lines", True)),
    )

    run_dir = _next_run_dir(repo_dir / "translation", docname)
    artifacts_dir = run_dir / "artifacts"
    translations_dir = artifacts_dir / "translations"
    translations_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pdf_path, run_dir / pdf_path.name)
    write_json(artifacts_dir / "config_resolved.json", config)
    _save_sentences(artifacts_dir / "sentences.json", sentences)

    highlight_path = run_dir / f"{docname}.highlight.pdf"
    create_sentence_highlight_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        output_path=highlight_path,
        opacity=float(config.get("highlight_opacity") or 0.25),
    )

    instructions = _load_prompt(config=config, repo_dir=repo_dir)
    asyncio.run(
        translate_sentences(
            sentences,
            config=config,
            instructions=instructions,
            out_dir=translations_dir,
        )
    )
    print(f"Run folder: {run_dir}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    repo_dir = Path(__file__).resolve().parent
    run_dir = normalize_path(args.run).resolve()
    artifacts_dir = run_dir / "artifacts"

    config = load_config(normalize_path(args.config).resolve())
    font_path = _resolve_font_path(config=config, repo_dir=repo_dir)
    align = _resolve_align(config)
    math_scale = float(config.get("math_scale") or 1.15)

    docname = _docname_from_run_dir(run_dir)
    expected_pdf = run_dir / f"{docname}.pdf"
    if expected_pdf.exists():
        pdf_path = expected_pdf
    else:
        pdf_candidates = list(run_dir.glob("*.pdf"))
        if not pdf_candidates:
            raise FileNotFoundError(f"No PDF found in run dir: {run_dir}")
        # Fall back to a best-effort guess.
        pdf_path = sorted(pdf_candidates, key=lambda p: (".highlight" in p.stem, ".korean" in p.stem, p.name))[0]

    if bool(getattr(args, "rebuild_sentences", False)):
        conv_dir = (repo_dir / "conversion" / f"{docname}.pdf").resolve()
        lines_json_path = conv_dir / f"{docname}.lines.json"
        lines_json = read_json(lines_json_path)
        sentences = sentences_from_lines_json(
            lines_json,
            indent_ratio=float(config.get("sentence_indent_ratio") or 0.015),
            spacing_factor=float(config.get("sentence_spacing_factor") or 1.6),
            min_text_len=int(config.get("sentence_min_text_len") or 1),
            include_math=bool(config.get("include_math_lines", True)),
        )
        _save_sentences(artifacts_dir / "sentences.json", sentences)
    else:
        sentences = _load_sentences(artifacts_dir / "sentences.json")

    translations = _extract_translations(artifacts_dir / "translations" / "translations_cache.json")
    for s in sentences:
        if s.sentence_id in translations:
            continue
        if bool(s.meta.get("translate", True)) is False:
            translations[s.sentence_id] = s.source_text

    korean_out = run_dir / f"{docname}.korean.pdf"
    both_out = run_dir / f"{docname}.korean.both.pdf"
    render_korean_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        translations=translations,
        font_path=font_path,
        output_path=korean_out,
        whiteout=True,
        padding=float(config.get("render_padding") or 1.0),
        align=align,
        math_scale=math_scale,
    )
    render_side_by_side_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        translations=translations,
        font_path=font_path,
        output_path=both_out,
        padding=float(config.get("render_padding") or 1.0),
        separator=True,
        align=align,
        math_scale=math_scale,
    )
    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="mptr", description="Mathpix PDF translation pipeline")
    parser.add_argument("--config", default="config.json", help="Path to config.json")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser("convert", help="Mathpix conversion only")
    p_convert.add_argument("--pdf", required=True, help="Input PDF path")
    p_convert.add_argument("--force", action="store_true", help="Force re-convert even if cached")
    p_convert.set_defaults(func=cmd_convert)

    p_all = sub.add_parser("all", help="Run conversion + translation + render")
    p_all.add_argument("--pdf", required=True, help="Input PDF path")
    p_all.add_argument("--force-convert", action="store_true", help="Force re-convert")
    p_all.set_defaults(func=cmd_all)

    p_translate = sub.add_parser("translate", help="Translate only (use cached conversion)")
    p_translate.add_argument("--pdf", required=True, help="Input PDF path")
    p_translate.add_argument("--force-convert", action="store_true", help="Force re-convert")
    p_translate.set_defaults(func=cmd_translate)

    p_render = sub.add_parser("render", help="Render PDFs again from a run folder")
    p_render.add_argument("--run", required=True, help="Run folder path (translation/<DOC>.<N>.pdf)")
    p_render.add_argument(
        "--rebuild-sentences",
        action="store_true",
        help="Rebuild sentence regions from cached conversion/<DOC>.pdf/<DOC>.lines.json before rendering",
    )
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
