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
    local_embed_model: str
    data_dir: Path
    credentials_path: Path
    token_path: Path
    embed_delay: float = 0.0  # kept as a no-op for backward compat with existing .env files


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
        local_embed_model=os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5").strip(),
        data_dir=Path(os.environ.get("GDRIVE_RAG_DATA_DIR", "./data")).expanduser(),
        credentials_path=Path(
            os.environ.get("GDRIVE_RAG_CREDENTIALS", "./credentials.json")
        ).expanduser(),
        token_path=Path(os.environ.get("GDRIVE_RAG_TOKEN", "./token.json")).expanduser(),
        embed_delay=float(os.environ.get("GEMINI_EMBED_DELAY", "0.0")),
    )
