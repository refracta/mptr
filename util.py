from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def normalize_path(path_str: str) -> Path:
    win_drive = re.match(r"^([A-Za-z]):[\\\\/](.*)$", path_str)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path_str).expanduser()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    # Environment override: MPTR_<config_key>=...
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("MPTR_"):
            continue
        key = env_key[len("MPTR_") :]
        if not key:
            continue
        if key not in config:
            # Allow adding new keys via env override.
            pass
        try:
            config[key] = json.loads(env_val)
        except Exception:
            config[key] = env_val

    return config


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


def render_prompt_template(template: str, config: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in config:
            return str(config.get(key, ""))
        return match.group(0)

    return _PLACEHOLDER_RE.sub(repl, template)

