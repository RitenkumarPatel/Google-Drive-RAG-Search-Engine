# Google-Drive-RAG-Search-Engine (`gdrive-rag`)

A natural-language question-answering tool over your personal/work Google Drive documents powered by local sentence embeddings (`BAAI/bge-base-en-v1.5`) and Google Gemini models (`gemini-flash-latest`), featuring format-adaptive, verifiable citations linked directly to your Google Drive files.

---

## Features

- **Multi-Format Parsing**: Extracts text and citation anchors from Google Docs, PDFs, Word documents (`.docx`), Markdown (`.md`), and Plain Text (`.txt`).
- **Format-Adaptive Citations**: Preserves PDF page numbers (`p. 4`) and heading breadcrumbs (`Processes > Scheduling`).
- **Fast Local Embeddings**: Uses `sentence-transformers` locally (`BAAI/bge-base-en-v1.5`) with asymmetric query prefixing and zero external embedding API rate limits.
- **Incremental Sync & Deletion Reconciliation**: ChromaDB vector store + SQLite state tracking (`content_version` hash/timestamp matching) with automatic purging of deleted Drive files.
- **Grounded LLM Q&A**: Synthesizes accurate answers strictly grounded in your Drive files via Gemini chat models, with numbered citations `[1]`, `[2]` and direct Drive URLs.

---

## Setup Instructions

### 1. Installation

Clone the repository and install the package inside a virtual environment:

```bash
# Clone the repository
git clone https://github.com/RitenkumarPatel/Google-Drive-RAG-Search-Engine.git
cd Google-Drive-RAG-Search-Engine

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e '.[dev]'
```

---

### 2. Configure Environment (`.env`)

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set your Gemini API key (obtain a free key from [Google AI Studio](https://aistudio.google.com)):

```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

*(Optional settings such as `GEMINI_CHAT_MODEL=gemini-flash-latest` and `LOCAL_EMBED_MODEL=BAAI/bge-base-en-v1.5` are pre-configured with defaults).*

---

### 3. Google Drive OAuth Setup & Authentication

To allow `gdrive-rag` to access your Drive files in read-only mode:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project (or select an existing one) and enable the **Google Drive API**.
3. Configure the **OAuth consent screen** (User Type: **External**, add your email under Test Users).
4. Go to **Credentials** → **Create Credentials** → **OAuth client ID**.
   - Application type: **Desktop app** (or **Web application** with redirect URI `http://localhost:8765/`).
5. Download the JSON credential file and save it in the project root as:
   ```
   credentials.json
   ```
6. Run the interactive login command:
   ```bash
   gdrive-rag login
   ```
   Follow the printed URL in your browser, grant read-only Drive access, and complete authorization. Your access token will be saved to `./token.json`.

---

### 4. Verify Setup

Check that configuration and API access are working:

```bash
# View loaded settings
gdrive-rag config

# Test Gemini API connectivity
gdrive-rag ping

# List files available in your Google Drive
gdrive-rag list --limit 10
```

---

## Usage Guide

### Indexing Documents

Run the indexer to fetch, parse, chunk, embed, and store your Drive documents. Indexing uses incremental delta sync, only updating new or modified files and purging removed ones:

```bash
# Index all supported documents in your Google Drive
gdrive-rag index

# Or limit to indexing the first N documents
gdrive-rag index --limit 50
```

Check the indexing status and database summary at any time:

```bash
# View statistics (total files, total chunks, last sync time)
gdrive-rag stats

# View file-by-file tracking status
gdrive-rag status
```

---

### Querying Your Drive

#### 1. Ask Questions (LLM Grounded Synthesis + Citations)
Synthesize an answer using Gemini with inline citations `[1]`, `[2]` and clickable Drive links:

```bash
gdrive-rag ask "What were the key takeaways from the Q3 roadmap doc?"
```

Adjust the number of retrieved context chunks with `--k`:
```bash
gdrive-rag ask "Summarize the project timeline" --k 8
```

#### 2. Semantic Search (Dense Retrieval Only)
Retrieve raw matching chunks and similarity scores without calling Gemini LLM generation:

```bash
gdrive-rag search "process scheduling algorithms" --k 5
```

---

## Running Tests

All unit tests run offline with mocked network calls:

```bash
pytest
```
