#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from highlight import create_sentence_highlight_pdf
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


def _next_run_dir(base: Path, docname: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base / f"{docname}.{n}.pdf"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        n += 1


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
    )

    run_dir = _next_run_dir(repo_dir / "translation", docname)
    artifacts_dir = run_dir / "artifacts"
    translations_dir = artifacts_dir / "translations"
    translations_dir.mkdir(parents=True, exist_ok=True)

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

    # 3-5) Translate sentences (cached)
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
    )
    stats_both = render_side_by_side_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        translations=translations,
        font_path=font_path,
        output_path=both_out,
        padding=float(config.get("render_padding") or 1.0),
        separator=True,
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


def cmd_render(args: argparse.Namespace) -> int:
    repo_dir = Path(__file__).resolve().parent
    run_dir = normalize_path(args.run).resolve()
    artifacts_dir = run_dir / "artifacts"

    config = load_config(normalize_path(args.config).resolve())
    font_path = _resolve_font_path(config=config, repo_dir=repo_dir)

    sentences = _load_sentences(artifacts_dir / "sentences.json")
    translations = _extract_translations(artifacts_dir / "translations" / "translations_cache.json")

    pdf_candidates = list(run_dir.glob("*.pdf"))
    # Prefer original copied into run folder.
    if pdf_candidates:
        pdf_path = pdf_candidates[0]
    else:
        raise FileNotFoundError(f"No PDF found in run dir: {run_dir}")
    docname = pdf_path.stem

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
    )
    render_side_by_side_pdf(
        pdf_path=pdf_path,
        sentences=sentences,
        translations=translations,
        font_path=font_path,
        output_path=both_out,
        padding=float(config.get("render_padding") or 1.0),
        separator=True,
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

    p_render = sub.add_parser("render", help="Render PDFs again from a run folder")
    p_render.add_argument("--run", required=True, help="Run folder path (translation/<DOC>.<N>.pdf)")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

