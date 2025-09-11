import os, time, sqlite3, hashlib, json, re
from pathlib import Path
from collections import OrderedDict

from kb_config import KB_DIR, KB_DB, SUPPORTED_EXTS, VERSION_KEY
import kb_config

# Fast LRU cache for file stats to avoid disk reads
_file_cache = OrderedDict()
_cache_limit = 512

def _content_sig(file_path):
    """Content-based signature (path-agnostic)"""
    p = Path(file_path)
    size = p.stat().st_size
    
    if size <= 64 * 1024 * 1024:  # ≤64MB: full hash
        return hashlib.md5(p.read_bytes()).hexdigest()
    
    # Large file: composite hash
    with open(p, 'rb') as f:
        # First 64KB
        first = f.read(65536)
        
        # Middle 64KB
        f.seek(size // 2)
        middle = f.read(65536)
        
        # Last 64KB  
        f.seek(max(0, size - 65536))
        last = f.read(65536)
    
    hasher = hashlib.md5()
    hasher.update(first + middle + last + str(size).encode())
    return hasher.hexdigest()

def _text_hash(text):
    """Hash for text chunks - 16-byte binary MD5 for space efficiency"""
    return hashlib.md5(text.encode('utf-8')).digest()  # Binary instead of hex

def _get_file_cache_key(path):
    """Fast cache key for file stats"""
    stat = path.stat()
    return (str(path), stat.st_size, int(stat.st_mtime), VERSION_KEY)

def _check_file_cached(file_path):
    """Fast check if file is already indexed - avoids expensive hashing"""
    path = Path(file_path).resolve()
    
    # Check in-memory cache first
    cache_key = _get_file_cache_key(path)
    if cache_key in _file_cache:
        _file_cache.move_to_end(cache_key)  # LRU update
        return _file_cache[cache_key]  # Returns file_uid
    
    # Check DB with size+mtime guard (no hashing yet)
    con = _init_db()
    cur = con.cursor()
    
    try:
        stat = path.stat()
        row = cur.execute(
            "SELECT file_uid, content_sig FROM files WHERE path = ? AND size = ? AND mtime = ? AND version = ?",
            (str(path), stat.st_size, stat.st_mtime, VERSION_KEY)
        ).fetchone()
        
        if row:
            file_uid = row[0]
            # Cache the result
            _file_cache[cache_key] = file_uid
            if len(_file_cache) > _cache_limit:
                _file_cache.popitem(last=False)
            return file_uid
        
        return None  # Need indexing
        
    finally:
        con.close()

def _init_db():
    """Initialize KB database"""
    KB_DIR.mkdir(exist_ok=True)
    
    con = sqlite3.connect(KB_DB, check_same_thread=False, timeout=30)
    con.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA mmap_size=268435456;
        PRAGMA cache_size=-200000;
        PRAGMA busy_timeout=30000;
        PRAGMA wal_autocheckpoint=4000;
    """)
    
    # Files manifest
    con.execute("""CREATE TABLE IF NOT EXISTS files (
        file_uid TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        size INTEGER,
        mtime REAL,
        content_sig TEXT NOT NULL,
        version TEXT NOT NULL,
        indexed_at REAL,
        status TEXT DEFAULT 'active'
    )""")
    
    # Simplified: Keep FTS as-is but with space optimizations
    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS fts 
                   USING fts5(id UNINDEXED, text, file_uid UNINDEXED, file_path UNINDEXED, text_hash UNINDEXED,
                              tokenize='porter', prefix='2')""")
    
    # Embeddings cache with binary keys and simpler schema
    con.execute("""CREATE TABLE IF NOT EXISTS embeddings (
        text_hash BLOB,
        model TEXT,
        dim INTEGER, 
        version TEXT,
        vector BLOB,
        created REAL DEFAULT (unixepoch()),
        PRIMARY KEY (text_hash, model, dim, version)
    )""")
    
    con.commit()
    return con

def _extract_text(file_path):
    """Extract text from supported file types"""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext in ['.txt', '.md']:
        return path.read_text(encoding='utf-8')
    elif ext == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return " ".join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
        elif isinstance(data, list):
            return " ".join(str(item) for item in data if isinstance(item, (str, int, float)))
        else:
            return str(data)
    elif ext == '.jsonl':
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                if 'text' in obj:
                    texts.append(obj['text'])
                elif 'title' in obj and 'text' in obj:
                    texts.append(f"{obj['title']} {obj['text']}")
                else:
                    texts.append(" ".join(str(v) for v in obj.values() if isinstance(v, (str, int, float))))
        return "\n".join(texts)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _normalize_text(text):
    """Normalize text"""
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', ' ', text).strip()

def _span_chunk(text, words_per_span=220, stride=200):
    """Chunk text into overlapping spans of words.
    
    Args:
        text: Input text to chunk
        words_per_span: Number of words per chunk (220 for best accuracy)
        stride: Number of words to advance between chunks
        
    Returns:
        List of (chunk_type, text) tuples
    """
    words = text.split()
    if not words:
        return []
    
    spans = []
    start = 0
    while start < len(words):
        end = min(start + words_per_span, len(words))
        chunk = ' '.join(words[start:end])
        spans.append(("simple", chunk))
        
        if end >= len(words):
            break
        start += stride
    
    return spans

def ensure_indexed(file_path, force_reindex=False):
    """Ensure file is indexed - main API with fast path"""
    path = Path(file_path).resolve()
    
    # Check extension
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {path.suffix}")
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # FAST PATH: check if already indexed without expensive operations
    if not force_reindex:
        cached_uid = _check_file_cached(path)
        if cached_uid:
            return cached_uid
    
    # SLOW PATH: need to index/reindex (only when actually needed)
    return _do_index(path)

def _do_index(path):
    """Actually perform the indexing work (separated from fast path)"""
    # Get file stats
    stat = path.stat()
    size, mtime = stat.st_size, stat.st_mtime
    content_sig = _content_sig(path)
    file_uid = f"f_{content_sig}"
    
    con = _init_db()
    cur = con.cursor()
    
    try:
        # Check if already indexed
        row = cur.execute(
            "SELECT content_sig, version FROM files WHERE file_uid = ?", 
            (file_uid,)
        ).fetchone()
        
        if row and row[0] == content_sig and row[1] == VERSION_KEY:
            # Update path in case of rename and cache
            cur.execute("UPDATE files SET path = ? WHERE file_uid = ?", (str(path), file_uid))
            con.commit()
            
            # Update cache
            cache_key = _get_file_cache_key(path)
            _file_cache[cache_key] = file_uid
            if len(_file_cache) > _cache_limit:
                _file_cache.popitem(last=False)
            
            return file_uid
        
        # Need to index/re-index
        if kb_config.DEBUG:
            print(f"Indexing: {path.name}")
        
        # Extract and chunk
        raw_text = _extract_text(path)
        text = _normalize_text(raw_text)
        spans = _span_chunk(text)
        
        # Get existing span hashes for delta
        existing_hashes = set()
        if row:  # re-index
            existing_rows = cur.execute(
                "SELECT text_hash FROM fts WHERE file_uid = ?", (file_uid,)
            ).fetchall()
            existing_hashes = {r[0] for r in existing_rows}
        
        # Compute new spans
        new_spans = []
        new_hashes = set()
        for i, (chunk_type, span_text) in enumerate(spans):
            span_id = f"{file_uid}::{i}" if len(spans) > 1 else file_uid
            text_hash = _text_hash(span_text)
            new_spans.append((span_id, span_text, file_uid, str(path), text_hash))
            new_hashes.add(text_hash)
        
        # Delta updates
        stale_hashes = existing_hashes - new_hashes
        fresh_hashes = new_hashes - existing_hashes
        
        # Delete stale spans and embeddings
        if stale_hashes:
            placeholders = ','.join(['?' for _ in stale_hashes])
            cur.execute(f"DELETE FROM fts WHERE file_uid = ? AND text_hash IN ({placeholders})", 
                       [file_uid] + list(stale_hashes))
            cur.execute(f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})", 
                       list(stale_hashes))
        
        # Insert fresh spans
        fresh_spans = [s for s in new_spans if s[4] in fresh_hashes]
        if fresh_spans:
            cur.executemany("INSERT INTO fts(id, text, file_uid, file_path, text_hash) VALUES(?,?,?,?,?)", 
                           fresh_spans)
        
        # Update manifest
        cur.execute("""INSERT OR REPLACE INTO files 
                      (file_uid, path, size, mtime, content_sig, version, indexed_at, status) 
                      VALUES (?,?,?,?,?,?,?,?)""",
                   (file_uid, str(path), size, mtime, content_sig, VERSION_KEY, time.time(), 'active'))
        
        con.commit()
        if kb_config.DEBUG:
            print(f"Indexed: {len(fresh_spans)} new spans, {len(stale_hashes)} removed")
        
        # Update cache
        cache_key = _get_file_cache_key(path)
        _file_cache[cache_key] = file_uid
        if len(_file_cache) > _cache_limit:
            _file_cache.popitem(last=False)
        
        return file_uid
        
    finally:
        con.close()

def ingest_many(paths_or_dir):
    """Ingest multiple files or directory"""
    if isinstance(paths_or_dir, (str, Path)):
        path = Path(paths_or_dir)
        if path.is_dir():
            paths = []
            for ext in SUPPORTED_EXTS:
                paths.extend(path.rglob(f"*{ext}"))
        else:
            paths = [path]
    else:
        paths = [Path(p) for p in paths_or_dir]
    
    file_uids = []
    for path in paths:
        try:
            file_uid = ensure_indexed(path)
            file_uids.append(file_uid)
        except Exception as e:
            if kb_config.DEBUG:
                print(f"Failed to index {path}: {e}")
    
    # Housekeeping after batch (separate connection)
    if file_uids:
        try:
            con = _init_db()
            con.execute("INSERT INTO fts(fts) VALUES('optimize')")
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
        except Exception as e:
            if kb_config.DEBUG:
                print(f"Housekeeping failed: {e}")
    
    return file_uids

def sweep(retention_days=30):
    """Clean up missing files and orphaned data; prune embeddings by age."""
    con = _init_db()
    cur = con.cursor()
    try:
        files = cur.execute("SELECT file_uid, path FROM files WHERE status = 'active'").fetchall()
        missing_uids = [file_uid for file_uid, path in files if not Path(path).exists()]

        if missing_uids:
            placeholders = ",".join(["?"] * len(missing_uids))
            cur.execute(f"UPDATE files SET status = 'missing' WHERE file_uid IN ({placeholders})", missing_uids)
            cur.execute(f"DELETE FROM fts WHERE file_uid IN ({placeholders})", missing_uids)

        cutoff_time = time.time() - (retention_days * 24 * 3600)
        cur.execute(
            """DELETE FROM embeddings
               WHERE text_hash NOT IN (SELECT DISTINCT text_hash FROM fts)
                  OR created < ?""",
            (cutoff_time,),
        )

        con.commit()
        if kb_config.DEBUG:
            print(f"Swept: {len(missing_uids)} missing files, orphaned embeddings cleaned")
    finally:
        con.close()
        
def get_db_path():
    """Get the KB database path"""
    return str(KB_DB)

def prep_file(file_path):
    """Lightweight prep: ensure file is indexed (for batch scenarios)"""
    return ensure_indexed(file_path, force_reindex=False)

def get_file_uid(file_path):
    """Fast lookup of file_uid without indexing (read-only)"""
    path = Path(file_path).resolve()
    
    # Check cache first
    try:
        cache_key = _get_file_cache_key(path)
        if cache_key in _file_cache:
            _file_cache.move_to_end(cache_key)
            return _file_cache[cache_key]
    except:
        pass  # File might not exist, etc.
    
    # Check DB
    con = _init_db()
    try:
        stat = path.stat()
        row = con.execute(
            "SELECT file_uid FROM files WHERE path = ? AND size = ? AND mtime = ? AND version = ?",
            (str(path), stat.st_size, stat.st_mtime, VERSION_KEY)
        ).fetchone()
        if row:
            return row[0]
    except:
        pass
    finally:
        con.close()
    
    return None
def remove_files(paths):
    """Remove one or more files from the index (fts, files, embeddings by span hashes)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [str(Path(p).resolve()) for p in paths]

    con = _init_db()
    cur = con.cursor()
    removed = 0
    try:
        for p in paths:
            rows = cur.execute("SELECT file_uid FROM files WHERE path = ?", (p,)).fetchall()
            for (file_uid,) in rows:
                hashes = [r[0] for r in cur.execute("SELECT text_hash FROM fts WHERE file_uid = ?", (file_uid,)).fetchall()]
                if hashes:
                    placeholders = ",".join("?" * len(hashes))
                    cur.execute(f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})", hashes)
                cur.execute("DELETE FROM fts WHERE file_uid = ?", (file_uid,))
                cur.execute("DELETE FROM files WHERE file_uid = ?", (file_uid,))
                removed += 1
        con.commit()
    finally:
        con.close()
        _file_cache.clear()
    return removed

def clear_embeddings(days=None):
    """Delete embeddings entirely or older than N days."""
    con = _init_db()
    try:
        if days is None:
            con.execute("DELETE FROM embeddings")
        else:
            cutoff = time.time() - days * 24 * 3600
            con.execute("DELETE FROM embeddings WHERE created < ?", (cutoff,))
        con.commit()
    finally:
        con.close()

def clear_all():
    """Drop the entire KB database (and WAL/SHM), then recreate empty schema."""
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(KB_DB) + suffix).unlink(missing_ok=True)
        except Exception:
            pass
    _file_cache.clear()
    con = _init_db()
    con.close()