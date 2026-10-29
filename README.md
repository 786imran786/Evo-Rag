# EvoRAG — Adaptive Evidence-First RAG

A full-stack demo of an adaptive, evidence-first Retrieval-Augmented Generation system. It ingests documents and URLs, detects knowledge gaps, retrieves semantically relevant chunks, asks Groq for grounded answers, checks claim support, and exposes evidence and evaluation metrics.

## Architecture

- **Frontend:** React + Vite + TypeScript + plain CSS dashboard
- **API:** FastAPI
- **LLM:** Groq API (free-tier availability/model limits depend on your Groq account)
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`); deterministic TF-IDF fallback if the model cannot load
- **Retrieval:** cosine similarity + metadata-aware scoring
- **Storage:** local JSON/JSONL files under `backend/data` (no database required for the demo)

## Quick start

### 1. Requirements
- Python 3.10+
- Node.js 18+
- A Groq API key from https://console.groq.com/keys

### 2. Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows CMD; on PowerShell use Copy-Item .env.example .env
# macOS/Linux: cp .env.example .env
```

Add `GROQ_API_KEY=...` to `backend/.env`, then run:

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 3. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually http://localhost:5173).

## How to use
1. Open **Knowledge Base** and upload `.txt`, `.md`, `.pdf`, or `.docx` files, or ingest a public URL.
2. Ask a question in **Ask EvoRAG**. The pipeline analyzes the query, retrieves evidence, and uses Groq to produce an answer with source citations.
3. Review evidence snippets and verification status. If the corpus is weak, EvoRAG flags the knowledge gap instead of pretending certainty.
4. Use **Evaluation** to run a small built-in benchmark comparing evidence-first retrieval/generation against a retrieval-only baseline. These are demo metrics, not a claim of scientific validation.

## Important notes
- Never commit `.env`; `.gitignore` excludes it. Do not expose the Groq key in frontend code.
- URL ingestion fetches public HTTP(S) pages only, blocks common private/local network targets, and has a timeout. For production, add a robust SSRF-safe fetch service, file size limits, auth, rate limiting, durable storage, and background jobs.
- Evidence verification here is a practical heuristic: lexical/semantic overlap plus an LLM support check when Groq is configured. It is not a formal proof of truth.
- Groq model IDs and free-tier limits can change. Configure `GROQ_MODEL` in `.env` if your account uses a different supported model.
