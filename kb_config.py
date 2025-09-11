# kb_config.py
import os, sys
from pathlib import Path

def _is_frozen() -> bool:
    return bool(
        getattr(sys, "frozen", False) or
        os.getenv("NUITKA_ONEFILE_PARENT") or
        getattr(sys, "_MEIPASS", None) or
        os.getenv("_MEIPASS2")
    )

# Mode: dev | test | prod | build
KB_ENV = (os.getenv("KB_ENV") or ("build" if _is_frozen() else "dev")).lower()

def _load_dotenv(paths):
    for base in paths:
        p = Path(base).expanduser() / ".env"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k not in os.environ:            # no overrides
                    os.environ[k] = v
            break

# Only load .env in dev/test
if KB_ENV in ("dev", "test"):
    kb_dir = os.getenv("KB_DIR", "~/knowledge-base")
    _load_dotenv([
        os.getcwd(),
        kb_dir,
        "~/.config/kb-fusion",
        os.path.dirname(__file__),
    ])

# Shared config
KB_DIR = Path(os.getenv("KB_DIR", "~/knowledge-base")).expanduser()
KB_DB = KB_DIR / "knbase.db"
SUPPORTED_EXTS = {".txt", ".md", ".json", ".jsonl"}

# Version key source of truth
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
OPENAI_DIM = int(os.getenv("OPENAI_DIM", "256"))
PREPROC_VER = 2
CHUNKER_VER = 4
VERSION_KEY = f"{OPENAI_MODEL}:{OPENAI_DIM}:p{PREPROC_VER}:c{CHUNKER_VER}"

# Global debug flag (set by CLI)
DEBUG = False

def set_debug(enabled: bool):
    global DEBUG
    DEBUG = enabled