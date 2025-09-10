# Kortix kb-fusion

Hybrid local semantic search (SQLite FTS5 + OpenAI embeddings). Fast, lean, agent-friendly. Apache-2.0.

## Quick start (CLI)

No install needed (uses `uv` to resolve/run):

```bash
# Help
uv run -m kb_fusion -h

# Search a file (auto-ingests on first run)
uv run -m kb_fusion search docs/handbook.md "How do I reset my 2FA?" -k 5

# Batch queries (file or stdin)
uv run -m kb_fusion search docs/handbook.md -Q queries.txt
cat queries.txt | uv run -m kb_fusion search docs/handbook.md -Q -

# JSON output
uv run -m kb_fusion search docs/handbook.md "reset 2FA" --json

# Maintenance
uv run -m kb_fusion ls
uv run -m kb_fusion sweep --remove /abs/path/to/file.txt
uv run -m kb_fusion sweep --clear-embeddings 14
uv run -m kb_fusion sweep --clear-all
uv run -m kb_fusion db-path
````

## Programmatic use

```python
from kb_fusion import ensure_indexed, semantic_search

ensure_indexed("docs/handbook.md")
hits = semantic_search("docs/handbook.md", ["How do I reset my 2FA?"])[0]
print(hits[:5])
```

Run it with `uv`:

```bash
uv run python your_script.py
```

## Configure

`.env` can live in **CWD**, `$KB_DIR`, `~/.config/kb-fusion`, or the package dir. Minimal:

```
OPENAI_API_KEY=sk-...
KB_DIR=~/knowledge-base
OPENAI_MODEL=text-embedding-3-small
OPENAI_DIM=256
```

See `.env.example` for advanced knobs.

## Notes

* Content-addressed KB with delta reindex; 220-word spans (stride ≈200); Jaccard de-dup; RRF + gated MMR; PRF with drift guard; tight snippets.
* SQLite tuned (WAL/mmap), LRU + persisted embedding cache.
* Version key is **auto-derived**: `{model}:{dim}:p2:c4` (see `kb_store.VERSION_KEY`) — changing model/dim safely isolates caches.

## Layout

`kb_store.py` (ingest/index/sweep, FTS/cache) • `semantic_search.py` (retrieval/fusion) • `kb_fusion.py`/`__main__.py` (CLI)
