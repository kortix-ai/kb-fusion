# Contributing to kb-fusion

Thanks for helping make Kortix kb-fusion better.

## Workflow
1. Fork → branch: `git checkout -b feat/your-thing`
2. Commit: conventional-ish messages (`feat:`, `fix:`, etc.)
3. Push & open a PR
4. Keep PRs scoped to one change; add tests/docs as needed

## Dev Setup (uv)
```bash
# create venv & install deps (incl. linters/tests)
uv venv
uv sync -e dev
# or: uv pip install -e ".[dev]"
````

## Running

```bash
# CLI (uses your workspace copy)
uv run -m kb_fusion -h
uv run -m kb_fusion search docs/handbook.md "reset 2FA" -k 5
uv run -m kb_fusion sweep --remove /abs/path/to/file.txt
```

Programmatic:

```bash
uv run python -c 'from kb_fusion import ensure_indexed,semantic_search; \
ensure_indexed("docs/handbook.md"); print(semantic_search("docs/handbook.md",["reset 2FA"])[0][:3])'
```

> Put your `.env` in CWD / `$KB_DIR` / `~/.config/kb-fusion` / package dir. Need `OPENAI_API_KEY`.

## Quality Gate

```bash
uv run ruff check .
uv run black --check .
uv run mypy .
uv run pytest
# (optional) uv run pre-commit install
```

## Style & Guidelines

* Python ≥ 3.9; line length=100; no heavy deps
* Follow existing patterns; prefer small functions & pure helpers
* Add/adjust tests for new behavior
* Update docs if CLI/API changes

## Versioning/Caches

* `VERSION_KEY` is auto-derived: `{model}:{dim}:p{PREPROC_VER}:c{CHUNKER_VER}`
* If you change preprocessing or chunking logic, bump `PREPROC_VER`/`CHUNKER_VER` constants

## Reporting Issues

Include:

* Repro steps, expected vs. actual
* OS, Python, `uv --version`, `kb_fusion.__version__`
* Relevant logs/output (redact keys)
* Minimal file/query example when possible

## Optional: Single-Binary (maintainers)

```bash
uv pip install ".[build]"
uv run python -m nuitka \
  --onefile \
  --standalone \
  --static-libpython=no \
  --follow-imports \
  --enable-plugin=numpy \
  --output-filename=kb \
  kb_fusion.py

# then: ./kb search path/to/file "query"
```
