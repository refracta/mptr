from __future__ import annotations

import asyncio
import json
import os
import shutil
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


def _cache_hit(entry: Any, sentence: Sentence) -> bool:
    if not isinstance(entry, dict):
        return False
    return str(entry.get("source_text") or "") == sentence.source_text


def _chunked(items: list[Sentence], chunk_size: int) -> list[list[Sentence]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _build_codex_prompt(*, instructions: str, sentences: list[Sentence]) -> str:
    input_obj = [
        {
            "sentence_id": s.sentence_id,
            "source_text": s.source_text,
        }
        for s in sentences
    ]
    return (
        instructions.strip()
        + "\n\n"
        "Output override:\n"
        "- Ignore any previous output-format rules.\n"
        "- Output ONLY valid JSON: {\"translations\": [{\"sentence_id\": string, \"translated_text\": string}, ...]}\n"
        "- `translations` must have exactly the same number of items as the input sentences.\n"
        "- Each output item must use the same `sentence_id`.\n"
        "- Do not output code fences.\n\n"
        "INPUT JSON (list of sentences):\n"
        + json.dumps(input_obj, ensure_ascii=False)
    )


async def _codex_translate_chunk(
    sentences: list[Sentence],
    *,
    config: dict,
    instructions: str,
    out_dir: Path,
    schema_path: Path,
    chunk_index: int,
) -> dict[str, TranslationResult]:
    model = str(config.get("codex_model") or "gpt-5.2")

    prompt = _build_codex_prompt(instructions=instructions, sentences=sentences)

    logs_dir = (out_dir / "codex_logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    base = f"chunk{chunk_index:04d}.pid{os.getpid()}"
    stdout_path = logs_dir / f"{base}.stdout.log"
    stderr_path = logs_dir / f"{base}.stderr.log"

    max_retries = int(config.get("max_retries") or 6)
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            attempt_base = f"{base}.attempt{attempt + 1:02d}"
            last_path = logs_dir / f"{attempt_base}.last.json"

            codex_args = [
                "codex",
                "exec",
                "--color",
                "never",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(last_path),
                "--model",
                model,
                "-",
            ]

            with stdout_path.open("ab") as out, stderr_path.open("ab") as err:
                proc = await asyncio.create_subprocess_exec(
                    *codex_args,
                    cwd=str(out_dir),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=out,
                    stderr=err,
                )
                await proc.communicate(input=prompt.encode("utf-8"))
            if proc.returncode != 0:
                raise RuntimeError(f"codex exec failed (exit={proc.returncode}); see {stderr_path}")

            raw = last_path.read_text(encoding="utf-8").strip()
            output_obj = json.loads(raw)
            translations = output_obj.get("translations")
            if not isinstance(translations, list):
                raise ValueError("Invalid codex output: missing translations list")

            by_id: dict[str, str] = {}
            for it in translations:
                if not isinstance(it, dict):
                    continue
                sid = str(it.get("sentence_id") or "").strip()
                tt = it.get("translated_text")
                if not sid or not isinstance(tt, str):
                    continue
                by_id[sid] = tt.strip()

            # Validate coverage (strict): every requested sentence_id must be present.
            missing = [s.sentence_id for s in sentences if s.sentence_id not in by_id]
            if missing:
                raise ValueError(f"Codex output missing {len(missing)} sentence_id(s)")

            now = _now_utc()
            out: dict[str, TranslationResult] = {}
            for s in sentences:
                out[s.sentence_id] = TranslationResult(
                    sentence_id=s.sentence_id,
                    source_text=s.source_text,
                    translated_text=by_id.get(s.sentence_id, "").strip(),
                    model=model,
                    response_id=None,
                    usage=None,
                    created_at=now,
                )
            return out
        except Exception as e:
            last_err = e
            await asyncio.sleep(min(60.0, 1.5**attempt))

    raise RuntimeError(f"Codex translation failed after {max_retries} retries: {last_err}")


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

    use_codex = bool(config.get("use_codex", False))
    model = str(config.get("model") or "gpt-5.2")
    parallelism = int(config.get("parallelism") or 4)
    temperature = float(config.get("temperature") or 0.2)
    max_retries = int(config.get("max_retries") or 6)

    if use_codex:
        schema_path = (Path(__file__).resolve().parent / "schemas" / "translate_sentences.schema.json").resolve()
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing Codex output schema: {schema_path}")
        if shutil.which("codex") is None:
            raise FileNotFoundError("codex executable not found in PATH")

        codex_chunk_size = int(config.get("codex_chunk_size") or 25)
        codex_parallelism_raw = config.get("codex_parallelism")
        if codex_parallelism_raw is None:
            codex_parallelism = max(1, parallelism)
        else:
            codex_parallelism = int(codex_parallelism_raw)

        sem = asyncio.Semaphore(max(1, codex_parallelism))
        lock = asyncio.Lock()
        results: dict[str, TranslationResult] = {}

        pending: list[Sentence] = []
        for s in sentences:
            entry = cache.get(s.sentence_id)
            if _cache_hit(entry, s):
                entry_dict = entry
                results[s.sentence_id] = TranslationResult(
                    sentence_id=s.sentence_id,
                    source_text=entry_dict.get("source_text", s.source_text),
                    translated_text=entry_dict.get("translated_text", ""),
                    model=entry_dict.get("model", model),
                    response_id=entry_dict.get("response_id"),
                    usage=entry_dict.get("usage"),
                    created_at=entry_dict.get("created_at", ""),
                )
                continue

            if bool(s.meta.get("translate", True)) is False:
                tr = TranslationResult(
                    sentence_id=s.sentence_id,
                    source_text=s.source_text,
                    translated_text=s.source_text,
                    model="skip",
                    response_id=None,
                    usage=None,
                    created_at=_now_utc(),
                )
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
                continue

            pending.append(s)

        chunks = _chunked(pending, codex_chunk_size)

        async def run_chunk(chunk: list[Sentence], idx: int) -> None:
            async with sem:
                chunk_results = await _codex_translate_chunk(
                    chunk,
                    config=config,
                    instructions=instructions,
                    out_dir=out_dir,
                    schema_path=schema_path,
                    chunk_index=idx,
                )
            async with lock:
                for sid, tr in chunk_results.items():
                    results[sid] = tr
                    cache[sid] = {
                        "sentence_id": tr.sentence_id,
                        "source_text": tr.source_text,
                        "translated_text": tr.translated_text,
                        "model": tr.model,
                        "response_id": tr.response_id,
                        "usage": tr.usage,
                        "created_at": tr.created_at,
                    }
                write_json(cache_path, cache)

        await asyncio.gather(*(run_chunk(c, i) for i, c in enumerate(chunks)))
        write_json(cache_path, cache)
        return results

    key = load_openai_api_key(config)
    client = AsyncOpenAI(api_key=key)

    sem = asyncio.Semaphore(max(1, parallelism))
    lock = asyncio.Lock()
    results: dict[str, TranslationResult] = {}

    async def run_one(s: Sentence) -> None:
        entry = cache.get(s.sentence_id)
        if _cache_hit(entry, s):
            entry_dict = entry
            results[s.sentence_id] = TranslationResult(
                sentence_id=s.sentence_id,
                source_text=entry_dict.get("source_text", s.source_text),
                translated_text=entry_dict.get("translated_text", ""),
                model=entry_dict.get("model", model),
                response_id=entry_dict.get("response_id"),
                usage=entry_dict.get("usage"),
                created_at=entry_dict.get("created_at", ""),
            )
            return

        if bool(s.meta.get("translate", True)) is False:
            tr = TranslationResult(
                sentence_id=s.sentence_id,
                source_text=s.source_text,
                translated_text=s.source_text,
                model="skip",
                response_id=None,
                usage=None,
                created_at=_now_utc(),
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
