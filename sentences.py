from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _line_bbox(line: dict) -> tuple[float, float, float, float]:
    region = line["region"]
    x0 = float(region["top_left_x"])
    y0 = float(region["top_left_y"])
    x1 = x0 + float(region["width"])
    y1 = y0 + float(region["height"])
    return x0, y0, x1, y1


def _looks_like_new_item(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return bool(
        re.match(r"^\[\d+\]\s+", t)  # [16] Foo...
        or re.match(r"^\(?\d+\)?[.)]\s+", t)  # 1. Foo / (1) Foo / 1) Foo
        or re.match(r"^[•◦▪●◆\-*]\s+", t)  # bullets
    )


def _union_region(lines: list[dict]) -> dict:
    x0 = float("inf")
    y0 = float("inf")
    x1 = float("-inf")
    y1 = float("-inf")
    for line in lines:
        lx0, ly0, lx1, ly1 = _line_bbox(line)
        x0 = min(x0, lx0)
        y0 = min(y0, ly0)
        x1 = max(x1, lx1)
        y1 = max(y1, ly1)
    return {
        "top_left_x": x0,
        "top_left_y": y0,
        "width": max(0.0, x1 - x0),
        "height": max(0.0, y1 - y0),
    }


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    page: int
    column: int
    page_width: float
    page_height: float
    region: dict
    source_text: str
    line_ids: list[str]
    meta: dict[str, Any]


def build_sentence_groups(
    page_lines: Iterable[dict],
    *,
    page_num: int,
    page_width: float,
    indent_ratio: float = 0.015,
    spacing_factor: float = 1.6,
    min_text_len: int = 1,
) -> list[list[dict]]:
    """
    Merge wrapped text lines into sentence-like blocks.

    Notes:
    - Groups only `type=text` lines with a region and non-empty text.
    - Uses bbox vertical gap to avoid merging across whitespace.
    - Splits on indentation shifts (right indent => new sentence; left shift + list item => new sentence).
    """
    candidates: list[dict] = []
    for line in page_lines:
        if line.get("type") != "text":
            continue
        if not isinstance(line.get("region"), dict):
            continue
        text = (line.get("text") or "").strip()
        if len(text) < min_text_len:
            continue
        candidates.append(line)

    by_col: dict[int, list[dict]] = defaultdict(list)
    for line in candidates:
        by_col[int(line.get("column") or 0)].append(line)

    groups: list[list[dict]] = []
    for col, col_lines in sorted(by_col.items(), key=lambda kv: kv[0]):
        col_lines.sort(key=lambda l: (_line_bbox(l)[1], _line_bbox(l)[0]))

        x0s = [_line_bbox(l)[0] for l in col_lines]
        heights = [(_line_bbox(l)[3] - _line_bbox(l)[1]) for l in col_lines]
        baseline_x0 = _median(x0s) or 0.0
        median_h = _median(heights) or 1.0

        # Estimate typical y spacing from y0 deltas.
        y0s = [_line_bbox(l)[1] for l in col_lines]
        deltas = [y0s[i + 1] - y0s[i] for i in range(len(y0s) - 1)]
        typical_dy = _median([d for d in deltas if d > 0]) or median_h

        indent_thresh = max(page_width * indent_ratio, typical_dy * 0.6)
        break_dy = typical_dy * spacing_factor
        gap_thresh = max(2.0, median_h * 0.25)

        cur: list[dict] = []
        prev: dict | None = None
        for line in col_lines:
            if prev is None:
                cur = [line]
                prev = line
                continue

            prev_x0, prev_y0, _, prev_y1 = _line_bbox(prev)
            x0, y0, _, y1 = _line_bbox(line)
            dy = y0 - prev_y0
            gap = y0 - prev_y1

            start_new = False
            if gap > gap_thresh:
                start_new = True
            elif dy > break_dy:
                start_new = True
            else:
                if (x0 - prev_x0) > indent_thresh and x0 > (baseline_x0 + indent_thresh * 0.5):
                    start_new = True
                elif (prev_x0 - x0) > indent_thresh and _looks_like_new_item((line.get("text") or "")):
                    start_new = True

            if start_new:
                groups.append(cur)
                cur = [line]
            else:
                cur.append(line)
            prev = line

        if cur:
            groups.append(cur)

    return groups


def sentences_from_lines_json(
    lines_json: dict,
    *,
    indent_ratio: float = 0.015,
    spacing_factor: float = 1.6,
    min_text_len: int = 1,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    pages = lines_json.get("pages") or []
    for page in pages:
        page_num = int(page.get("page") or (len(sentences) + 1))
        page_width = float(page.get("page_width") or 1.0)
        page_height = float(page.get("page_height") or 1.0)

        groups = build_sentence_groups(
            page.get("lines", []),
            page_num=page_num,
            page_width=page_width,
            indent_ratio=indent_ratio,
            spacing_factor=spacing_factor,
            min_text_len=min_text_len,
        )

        for idx, group in enumerate(groups):
            text = " ".join((l.get("text") or "").strip() for l in group).strip()
            line_ids = [str(l.get("id") or "") for l in group]
            region = _union_region(group)
            # Stable-ish id: page/col/first line id.
            col = int(group[0].get("column") or 0)
            seed = line_ids[0] if line_ids and line_ids[0] else f"g{idx}"
            sentence_id = f"p{page_num}-c{col}-{seed}"
            sentences.append(
                Sentence(
                    sentence_id=sentence_id,
                    page=page_num,
                    column=col,
                    page_width=page_width,
                    page_height=page_height,
                    region=region,
                    source_text=text,
                    line_ids=line_ids,
                    meta={
                        "kind": "text",
                        "group_index": idx,
                    },
                )
            )

    return sentences
