# Google-Drive-RAG-Search-Engine (`gdrive-rag`)

A natural-language question-answering tool over your personal/work Google Drive documents powered by Google's Gemini models with format-adaptive, verifiable citations linked directly to your Google Drive files.

---

## Features

- **Multi-Format Parsing**: Extracts text and citation anchors from Google Docs, PDFs, Word documents (`.docx`), Markdown (`.md`), and Plain Text (`.txt`).
- **Format-Adaptive Citations**: Preserves PDF page numbers (`p. 4`) and heading breadcrumbs (`Processes > Scheduling`).
- **Smart Contextual Chunking**: Prefixes chunks with document title and section locator for improved semantic retrieval.
- **Asymmetric Gemini Embeddings**: Normalizes embeddings via `gemini-embedding-001` (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search).
- **Persistent Storage & Incremental Sync**: ChromaDB vector store + SQLite state tracking (`content_version` hash/timestamp matching) with automatic deletion reconciliation.
- **Grounded LLM Q&A**: Synthesizes accurate answers strictly grounded in your Drive files via Gemini chat models (`gemini-flash-latest`), with numbered inline citations `[1]`, `[2]` and source links.

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/RitenkumarPatel/Google-Drive-RAG-Search-Engine.git
cd Google-Drive-RAG-Search-Engine

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dependencies
pip install -e '.[dev]'
```

### 2. Configuration

Copy `.env.example` to `.env` and provide your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Google Drive OAuth Setup

Place your Google Cloud OAuth Client ID JSON file at `./credentials.json` (with `https://www.googleapis.com/auth/drive.readonly` scope enabled).

Run the interactive copy-paste login:
```bash
gdrive-rag login
```

---

## CLI Usage

### Check Configuration & Connectivity
```bash
# View configuration
gdrive-rag config

# Test Gemini API connectivity
gdrive-rag ping
```

### Index Your Google Drive
```bash
# List recent Drive files
gdrive-rag list --limit 20

# Index Drive documents (fetches, chunks, embeds, and stores)
gdrive-rag index --limit 50

# Check index statistics
gdrive-rag stats
```

### Search & Ask Questions
```bash
# Semantic dense retrieval only
gdrive-rag search "What is context switching?" --k 6

# Grounded natural language Q&A with citations
gdrive-rag ask "Explain context switching and CPU scheduling" --k 6
```

---

## Running Tests

```bash
pytest
```
