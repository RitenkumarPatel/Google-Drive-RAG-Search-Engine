"""Application configuration, loaded from environment / a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    chat_model: str
    embed_model: str
    embed_dims: int
    data_dir: Path


def load_settings(*, require_api_key: bool = True, load_env: bool = True) -> Settings:
    """Build Settings from environment variables (and .env if present).

    Args:
        require_api_key: raise ConfigError if GEMINI_API_KEY is missing/blank.
        load_env: read a local .env into the environment first. Tests pass False
            to stay isolated from the developer's real .env.
    """
    if load_env:
        load_dotenv()  # no-op if there is no .env; never overrides existing env vars

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Add it to .env — get a free key at "
            "https://aistudio.google.com (Get API key)."
        )

    return Settings(
        gemini_api_key=api_key,
        chat_model=os.environ.get("GEMINI_CHAT_MODEL", "gemini-flash-latest").strip(),
        embed_model=os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001").strip(),
        embed_dims=int(os.environ.get("GEMINI_EMBED_DIMS", "768")),
        data_dir=Path(os.environ.get("GDRIVE_RAG_DATA_DIR", "./data")).expanduser(),
    )
