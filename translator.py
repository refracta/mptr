from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from sentences import Sentence
from util import normalize_path, read_json, write_json


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def load_openai_api_key(config: dict) -> str:
    explicit = (config.get("openai_api_key") or "").strip()
    if explicit:
        return explicit

    env = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env:
        return env

    key_file = Path("~/.codex/secrets/openai_api_key").expanduser()
    key = _read_text_file(key_file)
    if key:
        return key

    auth_file = Path("~/.codex/auth.json").expanduser()
    try:
        auth = json.loads(auth_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        auth = None
    except Exception:
        auth = None

    if isinstance(auth, dict):
        key = auth.get("OPENAI_API_KEY")
        if isinstance(key, str) and key.strip():
            return key.strip()

    raise RuntimeError("Missing OpenAI API key (set OPENAI_API_KEY or ~/.codex/secrets/openai_api_key).")


@dataclass(frozen=True)
class TranslationResult:
    sentence_id: str
    source_text: str
    translated_text: str
    model: str
    response_id: str | None
    usage: dict[str, Any] | None
    created_at: str


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return None
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return None
    return str(obj)


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


async def _translate_one(
    client: AsyncOpenAI,
    *,
    sentence: Sentence,
    model: str,
    instructions: str,
    temperature: float,
    max_retries: int,
) -> TranslationResult:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.responses.create(
                model=model,
                instructions=instructions,
                input=sentence.source_text,
                temperature=temperature,
            )
            return TranslationResult(
                sentence_id=sentence.sentence_id,
                source_text=sentence.source_text,
                translated_text=(resp.output_text or "").strip(),
                model=model,
                response_id=getattr(resp, "id", None),
                usage=_jsonable(getattr(resp, "usage", None)),
                created_at=_now_utc(),
            )
        except Exception as e:
            last_err = e
            # Simple exponential backoff.
            await asyncio.sleep(min(30.0, 1.5**attempt))

    raise RuntimeError(f"Translation failed after {max_retries} retries: {last_err}")


async def translate_sentences(
    sentences: list[Sentence],
    *,
    config: dict,
    instructions: str,
    out_dir: str | Path,
) -> dict[str, TranslationResult]:
    out_dir = normalize_path(str(out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "translations_cache.json"
    cache = _load_cache(cache_path)

    model = str(config.get("model") or "gpt-5.2")
    parallelism = int(config.get("parallelism") or 4)
    temperature = float(config.get("temperature") or 0.2)
    max_retries = int(config.get("max_retries") or 6)

    key = load_openai_api_key(config)
    client = AsyncOpenAI(api_key=key)

    sem = asyncio.Semaphore(max(1, parallelism))
    lock = asyncio.Lock()
    results: dict[str, TranslationResult] = {}

    async def run_one(s: Sentence) -> None:
        if s.sentence_id in cache and isinstance(cache[s.sentence_id], dict):
            entry = cache[s.sentence_id]
            results[s.sentence_id] = TranslationResult(
                sentence_id=s.sentence_id,
                source_text=entry.get("source_text", s.source_text),
                translated_text=entry.get("translated_text", ""),
                model=entry.get("model", model),
                response_id=entry.get("response_id"),
                usage=entry.get("usage"),
                created_at=entry.get("created_at", ""),
            )
            return

        async with sem:
            tr = await _translate_one(
                client,
                sentence=s,
                model=model,
                instructions=instructions,
                temperature=temperature,
                max_retries=max_retries,
            )

        async with lock:
            results[s.sentence_id] = tr
            cache[s.sentence_id] = {
                "sentence_id": tr.sentence_id,
                "source_text": tr.source_text,
                "translated_text": tr.translated_text,
                "model": tr.model,
                "response_id": tr.response_id,
                "usage": tr.usage,
                "created_at": tr.created_at,
            }
            write_json(cache_path, cache)

    await asyncio.gather(*(run_one(s) for s in sentences))
    write_json(cache_path, cache)
    return results
