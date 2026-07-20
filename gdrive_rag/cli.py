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


@cli.command()
def login() -> None:
    """Authorize read-only access to your Google Drive (one-time; prints a URL to approve)."""
    from .auth import run_manual_login
    from .drive import get_service, get_user_email

    settings = load_settings(require_api_key=False)
    try:
        creds = run_manual_login(settings)
    except ConfigError as e:
        raise click.ClickException(str(e))
    except Exception as e:  # oauthlib/network errors → friendly message
        raise click.ClickException(f"Login failed: {e}")

    email = get_user_email(get_service(creds)) or "your account"
    click.echo(f"✓ Authorized as {email} — token saved to {settings.token_path}")


@cli.command("list")
@click.option("--limit", default=20, show_default=True, help="Max number of files to show.")
def list_cmd(limit: int) -> None:
    """List your most recently modified Drive files."""
    from .auth import load_credentials
    from .drive import get_service, list_files

    settings = load_settings(require_api_key=False)
    creds = load_credentials(settings)
    if creds is None:
        raise click.ClickException("Not authorized yet. Run `gdrive-rag login` first.")

    try:
        files = list_files(get_service(creds), limit=limit)
    except Exception as e:
        raise click.ClickException(
            f"Drive request failed: {e}\n"
            "If you're behind an HTTP proxy, set https_proxy/http_proxy (e.g. in .env)."
        )
    if not files:
        click.echo("(no files found)")
        return
    click.echo(f"{len(files)} file(s):\n")
    for f in files:
        name = (f.get("name") or "")[:48]
        click.echo(
            f"  {name:<50} {f.get('mimeType', ''):<44} "
            f"{f.get('modifiedTime', '')}  {f.get('id', '')}"
        )


def main() -> None:  # pragma: no cover - thin wrapper
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
