"""Command-line interface for gdrive-rag."""

from __future__ import annotations

import click

from . import __version__
from .config import ConfigError, load_settings


@click.group()
@click.version_option(__version__, prog_name="gdrive-rag")
def cli() -> None:
    """Natural-language Q&A with citations over your Google Drive (Gemini)."""


@cli.command()
def config() -> None:
    """Print the loaded (non-secret) configuration."""
    try:
        s = load_settings(require_api_key=False)
    except ConfigError as e:  # pragma: no cover - defensive
        raise click.ClickException(str(e))
    click.echo(f"chat_model    : {s.chat_model}")
    click.echo(f"embed_model   : {s.embed_model}")
    click.echo(f"embed_dims    : {s.embed_dims}")
    click.echo(f"data_dir      : {s.data_dir}")
    click.echo(f"GEMINI_API_KEY: {'set' if s.gemini_api_key else 'MISSING'}")


@cli.command()
def ping() -> None:
    """Check Gemini connectivity: send a tiny prompt and print the reply."""
    try:
        settings = load_settings()
    except ConfigError as e:
        raise click.ClickException(str(e))

    try:
        from google import genai
        from google.genai import errors as genai_errors
    except ImportError as e:  # pragma: no cover
        raise click.ClickException(f"google-genai is not installed: {e}")

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        resp = client.models.generate_content(
            model=settings.chat_model,
            contents="Reply with the single word: pong",
        )
    except genai_errors.APIError as e:
        raise click.ClickException(
            f"Gemini API call failed ({type(e).__name__}): {e}. "
            "Check GEMINI_API_KEY, the model name, and your rate limits."
        )

    reply = (resp.text or "").strip()
    click.echo(f"✓ {settings.chat_model}: {reply}")


def main() -> None:  # pragma: no cover - thin wrapper
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
