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


@cli.command()
@click.argument("file_id")
@click.option("--head", default=40, show_default=True, help="Lines of extracted text to preview.")
def fetch(file_id: str, head: int) -> None:
    """Fetch + parse a single Drive file; print its format, headings/pages, and a text preview."""
    from . import parsers
    from .auth import load_credentials
    from .drive import get_file_metadata, get_service

    settings = load_settings(require_api_key=False)
    creds = load_credentials(settings)
    if creds is None:
        raise click.ClickException("Not authorized yet. Run `gdrive-rag login` first.")

    session = get_service(creds)
    try:
        meta = get_file_metadata(session, file_id)
        doc = parsers.fetch_document(session, meta)
    except parsers.UnsupportedFormat as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(
            f"Drive request failed: {e}\n"
            "If you're behind an HTTP proxy, set https_proxy/http_proxy (e.g. in .env)."
        )

    click.echo(f"name    : {doc.name}")
    click.echo(f"format  : {doc.fmt}  ({doc.mime_type})")
    click.echo(f"chars   : {len(doc.text)}")
    if doc.pages:
        click.echo(f"pages   : {len(doc.pages)}")
    if doc.sections:
        click.echo(f"headings: {len(doc.sections)}")
        for s in doc.sections[:10]:
            click.echo(f"    - {s.path}")
    click.echo(f"\n--- text (first {head} lines) ---")
    for line in doc.text.splitlines()[:head]:
        click.echo(line)


@cli.command()
@click.option("--limit", default=50, show_default=True, help="Max files to index this run.")
def index(limit: int) -> None:
    """Index your Drive: fetch → parse → chunk → embed → store (idempotent)."""
    from . import parsers
    from .auth import load_credentials
    from .chunker import chunk_document
    from .drive import get_service, list_all_ids, list_files
    from .embed import embed_texts
    from .store import Store, content_version

    try:
        settings = load_settings()  # API key required — embeddings hit Gemini
    except ConfigError as e:
        raise click.ClickException(str(e))
    creds = load_credentials(settings)
    if creds is None:
        raise click.ClickException("Not authorized yet. Run `gdrive-rag login` first.")

    session = get_service(creds)
    store = Store(settings)
    indexed = chunk_total = unchanged = unsupported = empty = failed = 0
    try:
        files = list_files(session, limit=limit)
        for f in files:
            if parsers.classify(f.get("mimeType")) is None:
                unsupported += 1
                continue
            if store.get_indexed_version(f["id"]) == content_version(f):
                unchanged += 1
                continue
            try:
                doc = parsers.fetch_document(session, f)
                if not doc.text.strip():
                    empty += 1
                    continue
                chunks = chunk_document(doc)
                vectors = embed_texts(settings, [c.text for c in chunks])
                store.replace_file(f, chunks, vectors)
            except Exception as e:  # one bad file shouldn't abort the whole run
                failed += 1
                click.echo(f"  ! skipped {f.get('name', f['id'])}: {e}", err=True)
                continue
            indexed += 1
            chunk_total += len(chunks)
        purged = store.reconcile(list_all_ids(session))
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(
            f"Index failed: {e}\n"
            "If you're behind an HTTP proxy, set https_proxy/http_proxy (e.g. in .env)."
        )
    finally:
        store.close()

    click.echo(
        f"indexed {indexed} file(s), {chunk_total} chunk(s); "
        f"unchanged {unchanged}, unsupported {unsupported}, empty {empty}, "
        f"failed {failed}, purged {len(purged)}"
    )


@cli.command()
def stats() -> None:
    """Show how many files and chunks are currently indexed."""
    from .store import Store

    settings = load_settings(require_api_key=False)
    store = Store(settings)
    try:
        s = store.stats()
    finally:
        store.close()
    click.echo(f"files : {s['files']}")
    click.echo(f"chunks: {s['chunks']}")
    click.echo(f"data  : {settings.data_dir}")


def main() -> None:  # pragma: no cover - thin wrapper
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
