"""Command-line interface for gdrive-rag."""

from __future__ import annotations

import datetime
import time

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
    click.echo(f"embed_delay   : {s.embed_delay}s")
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
        if hasattr(client, "chats") and hasattr(client.chats, "create"):
            chat = client.chats.create(model=settings.chat_model)
            resp = chat.send_message("Reply with the single word: pong")
        else:
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


def _format_time_str(raw: str | float | None) -> str:
    if not raw:
        return "—"
    if isinstance(raw, (int, float)):
        return datetime.datetime.fromtimestamp(raw, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    s = str(raw).replace("T", " ").rstrip("Z")
    return s.split(".")[0][:16]


@cli.command()
@click.option("--limit", default=50, show_default=True, help="Max new/updated files to index (0 for all).")
@click.option("--delay", default=None, type=float, help="Seconds to delay between embedding calls (rate limiting).")
def index(limit: int, delay: float | None) -> None:
    """Index your Drive: fetch → parse → chunk → embed → store (incremental delta sync)."""
    from . import parsers
    from .auth import load_credentials
    from .chunker import chunk_document
    from .drive import get_service, list_all_metadata
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
    failed_details: list[tuple[str, str]] = []

    try:
        all_drive_files = list_all_metadata(session)
        live_ids = [f["id"] for f in all_drive_files]
        purged = store.reconcile(live_ids)

        target_files: list[dict] = []
        for f in all_drive_files:
            if parsers.classify(f.get("mimeType")) is None:
                unsupported += 1
                continue
            rec = store.get_file_record(f["id"])
            if rec and rec["status"] == "indexed" and rec["content_version"] == content_version(f):
                unchanged += 1
                continue
            target_files.append(f)

        if not target_files:
            click.echo(
                f"All {unchanged} document(s) are up-to-date "
                f"({unsupported} unsupported, {len(purged)} purged)."
            )
            store.set_sync_meta("last_sync_completed_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
            return

        to_index = target_files if limit <= 0 else target_files[:limit]
        remaining = len(target_files) - len(to_index)
        click.echo(
            f"Found {len(target_files)} file(s) requiring indexing "
            f"({unchanged} unchanged, {unsupported} unsupported)."
        )
        if remaining > 0:
            click.echo(f"Indexing {len(to_index)} file(s) this run (budget limit: {limit}).")

        with click.progressbar(
            to_index,
            label="Indexing documents",
            item_show_func=lambda f: (f.get("name") or f.get("id", ""))[:32] if f else "",
        ) as bar:
            for f in bar:
                try:
                    doc = parsers.fetch_document(session, f)
                    if not doc.text.strip():
                        empty += 1
                        continue
                    chunks = chunk_document(doc)
                    vectors = embed_texts(settings, [c.text for c in chunks], delay=delay)
                    store.replace_file(f, chunks, vectors)
                    indexed += 1
                    chunk_total += len(chunks)
                except Exception as e:
                    failed += 1
                    store.mark_file_failed(f, str(e))
                    doc_name = f.get("name") or f.get("id", "Unknown")
                    failed_details.append((doc_name, str(e)))
                    continue

        store.set_sync_meta("last_sync_completed_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
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
        f"\n✓ Indexed {indexed} file(s), {chunk_total} chunk(s); "
        f"unchanged {unchanged}, unsupported {unsupported}, empty {empty}, "
        f"failed {failed}, purged {len(purged)}"
    )
    if failed_details:
        click.echo(f"\nFailed document(s) ({len(failed_details)}):")
        for name, err in failed_details:
            click.echo(f"  • {name}: {err}")


@cli.command()
def status() -> None:
    """Show detailed status of all locally tracked documents in SQLite."""
    from .store import Store

    settings = load_settings(require_api_key=False)
    store = Store(settings)
    try:
        records = store.list_indexed_records()
        s = store.stats()
        last_sync = store.get_sync_meta("last_sync_completed_at")
    finally:
        store.close()

    if not records:
        click.echo("\nNo documents indexed yet. Run `gdrive-rag index` to begin.\n")
        return

    click.echo(f"\nIndexed Google Drive Documents ({len(records)} total):\n")
    click.echo(f"  {'Document Name':<46} {'Last Updated':<18} {'Status':<10} {'Chunks':<6}")
    click.echo("  " + "─" * 82)

    for r in records:
        name = (r.get("name") or "Untitled")[:44]
        updated = _format_time_str(r.get("modified_time") or r.get("indexed_at"))
        st = r.get("status", "indexed")
        chunks = r.get("chunk_count", 0)
        click.echo(f"  {name:<46} {updated:<18} {st:<10} {chunks:<6}")

    click.echo("  " + "─" * 82)
    click.echo(f"  Total Files Indexed : {s['files']}")
    click.echo(f"  Total Chunks Stored : {s['chunks']}")
    if last_sync:
        click.echo(f"  Last Sync Completed : {_format_time_str(last_sync)}")
    click.echo(f"  Data Directory      : {settings.data_dir}\n")


@cli.command()
def stats() -> None:
    """Show summary file and chunk counts currently indexed."""
    from .store import Store

    settings = load_settings(require_api_key=False)
    store = Store(settings)
    try:
        s = store.stats()
        last_sync = store.get_sync_meta("last_sync_completed_at")
    finally:
        store.close()
    click.echo(f"files    : {s['files']}")
    click.echo(f"chunks   : {s['chunks']}")
    if last_sync:
        click.echo(f"last_sync: {_format_time_str(last_sync)}")
    click.echo(f"data     : {settings.data_dir}")


def _locator_suffix(locator: dict) -> str:
    value = locator.get("value")
    if not value:
        return ""
    return f" · p.{value}" if locator.get("type") == "page" else f" · {value}"


def _snippet(text: str, width: int = 200) -> str:
    body = text.split("\n\n", 1)[-1]  # drop the "Name — locator" prefix line
    body = " ".join(body.split())
    return body[:width] + ("…" if len(body) > width else "")


@cli.command("search")
@click.argument("query")
@click.option("--k", default=6, show_default=True, help="Number of results to return.")
def search_cmd(query: str, k: int) -> None:
    """Semantic search over your indexed Drive (retrieval only — no LLM answer)."""
    from .retrieve import search
    from .store import Store

    try:
        settings = load_settings()  # API key required — the query is embedded
    except ConfigError as e:
        raise click.ClickException(str(e))

    store = Store(settings)
    try:
        hits = search(settings, store, query, k=k)
    except Exception as e:
        raise click.ClickException(
            f"Search failed: {e}\n"
            "If you're behind an HTTP proxy, set https_proxy/http_proxy (e.g. in .env)."
        )
    finally:
        store.close()

    if not hits:
        click.echo("(no results — is anything indexed? run `gdrive-rag index`)")
        return
    for rank, h in enumerate(hits, 1):
        click.echo(f"{rank}. [{h.score:.3f}] {h.name}{_locator_suffix(h.locator)}")
        if h.drive_url:
            click.echo(f"   {h.drive_url}")
        click.echo(f"   {_snippet(h.text)}\n")


@cli.command("ask")
@click.argument("query")
@click.option("--k", default=6, show_default=True, help="Number of retrieved chunks to ground on.")
def ask_cmd(query: str, k: int) -> None:
    """Ask a question over your indexed Google Drive (synthesized answer with citations)."""
    from .answer import answer_query
    from .store import Store

    try:
        settings = load_settings()  # API key required — embeddings + generation hit Gemini
    except ConfigError as e:
        raise click.ClickException(str(e))

    store = Store(settings)
    try:
        ans = answer_query(settings, store, query, k=k)
    except Exception as e:
        raise click.ClickException(
            f"Ask failed: {e}\n"
            "If you're behind an HTTP proxy, set https_proxy/http_proxy (e.g. in .env)."
        )
    finally:
        store.close()

    click.echo(f"\n{ans.text}\n")
    if ans.citations:
        click.echo("Sources:")
        for c in ans.citations:
            loc = _locator_suffix(c.locator)
            url_part = f" — {c.drive_url}" if c.drive_url else ""
            click.echo(f"  [{c.index}] {c.name}{loc}{url_part}")
        click.echo()


def main() -> None:  # pragma: no cover - thin wrapper
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
