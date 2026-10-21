"""Load repo ``.env`` into ``os.environ`` for evaluation scripts.

Evaluation configs name a provider variable such as ``DEEPSEEK_API_KEY``.
``LLM_API_KEY`` fills any of those names when the named variable is unset,
and ``LLM_BASE_URL`` does the same for the matching base URL. A named
provider variable still wins when it is set. This module:

1. Parses ``.env`` without overriding already-exported shell variables.
2. Applies those aliases.
3. Normalizes Yunwu base URLs to the OpenAI-compatible ``.../v1`` form.

Importing this module is side-effect free; call :func:`ensure_env_loaded`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_LOADED = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# Canonical name -> fallback names to copy from when the canonical name is unset.
# Order matters: first non-empty wins.
_ALIASES: dict[str, tuple[str, ...]] = {
    "DEEPSEEK_API_KEY": ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
    "DEEPSEEK_BASE_URL": ("DEEPSEEK_BASE_URL", "DEEPSEEK_URL", "LLM_BASE_URL"),
    "OPENROUTER_API_KEY": ("OPENROUTER_API_KEY", "LLM_API_KEY"),
    "OPENROUTER_BASE_URL": ("OPENROUTER_BASE_URL", "LLM_BASE_URL"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY", "LLM_API_KEY"),
    "YUNWU_API_KEY": ("YUNWU_API_KEY", "GEMINI_KEY", "LLM_API_KEY"),
    "YUNWU_BASE_URL": ("YUNWU_BASE_URL", "YUNWU_URL", "URL", "LLM_BASE_URL"),
}

# When a config block uses api_key_env X, prefer these env vars for base_url.
_BASE_URL_FOR_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "DEEPSEEK_API_KEY": ("DEEPSEEK_BASE_URL", "DEEPSEEK_URL", "LLM_BASE_URL"),
    "OPENROUTER_API_KEY": ("OPENROUTER_BASE_URL", "LLM_BASE_URL"),
    "OPENAI_API_KEY": ("OPENAI_BASE_URL", "LLM_BASE_URL"),
    "YUNWU_API_KEY": ("YUNWU_BASE_URL", "YUNWU_URL", "URL", "LLM_BASE_URL"),
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a dotenv file. Duplicate keys keep the last assignment."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator:
            continue
        key = name.strip()
        if not key:
            continue
        value = raw_value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            closing = value.find(quote, 1)
            if closing < 0:
                continue
            value = value[1:closing]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def _normalize_base_url(url: str, *, provider: str | None = None) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return url
    if provider == "yunwu" or "yunwu.ai" in url:
        if not url.endswith("/v1"):
            url = f"{url}/v1"
    return url


def ensure_env_loaded(env_file: str | Path | None = None) -> Path:
    """Load ``.env`` + aliases into ``os.environ`` once.

    Already-exported environment variables always win over ``.env`` values.
    Returns the path that was considered (may not exist).
    """
    global _LOADED
    path = Path(env_file).expanduser().resolve() if env_file else DEFAULT_ENV_FILE
    if _LOADED:
        return path

    file_values = _parse_dotenv(path)
    for key, value in file_values.items():
        if key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = value

    for canonical, sources in _ALIASES.items():
        if os.environ.get(canonical):
            continue
        for source in sources:
            if source == canonical:
                continue
            value = os.environ.get(source) or file_values.get(source)
            if value:
                if "BASE_URL" in canonical or canonical.endswith("_URL"):
                    provider = "yunwu" if canonical.startswith("YUNWU") else None
                    value = _normalize_base_url(value, provider=provider)
                os.environ[canonical] = value
                break

    # Ensure Yunwu URL has /v1 even when set directly as YUNWU_BASE_URL/URL.
    for key in ("YUNWU_BASE_URL", "YUNWU_URL"):
        if os.environ.get(key):
            os.environ[key] = _normalize_base_url(os.environ[key], provider="yunwu")

    _LOADED = True
    return path


_DATA_ROOT_TOKEN = "${MOSAIC_DATA_ROOT}"


def expand_mosaic_string(value: str) -> str:
    """Replace ``${MOSAIC_DATA_ROOT}`` using the ``MOSAIC_DATA_ROOT`` env var."""
    if _DATA_ROOT_TOKEN not in value:
        return value
    root = os.environ.get("MOSAIC_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        raise RuntimeError(
            "Set MOSAIC_DATA_ROOT to the released data directory "
            "(the folder that contains dataset/stage2_human_filtered/)."
        )
    return value.replace(_DATA_ROOT_TOKEN, root)


def expand_mosaic_paths(value: Any) -> Any:
    """Expand ``${MOSAIC_DATA_ROOT}`` in strings inside a JSON config."""
    if isinstance(value, str):
        return expand_mosaic_string(value)
    if isinstance(value, dict):
        return {key: expand_mosaic_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [expand_mosaic_paths(child) for child in value]
    return value


def resolve_api_key(api_key_env: str | None = None) -> str:
    """Return the named key, or ``LLM_API_KEY`` when that name is unset."""
    ensure_env_loaded()
    if api_key_env:
        named = os.getenv(str(api_key_env), "").strip()
        if named:
            return named
    return os.getenv("LLM_API_KEY", "").strip()


def resolve_base_url_for_api_key_env(api_key_env: str | None) -> str | None:
    """Return a provider base URL from the environment for a given api_key_env.

    ``LLM_BASE_URL`` is the last fallback, including for a name that has no
    provider-specific URL list.
    """
    ensure_env_loaded()
    names = list(_BASE_URL_FOR_API_KEY_ENV.get(str(api_key_env or ""), ()))
    if "LLM_BASE_URL" not in names:
        names.append("LLM_BASE_URL")
    for name in names:
        value = os.getenv(name)
        if value:
            provider = "yunwu" if str(api_key_env) == "YUNWU_API_KEY" and name != "LLM_BASE_URL" else None
            return _normalize_base_url(value, provider=provider)
    return None
