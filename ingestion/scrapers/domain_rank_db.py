"""
domain_rank_db.py — DomCop Top-10M domain rank SQLite helper

Downloads the DomCop top-10M CSV (based on Open PageRank) and stores it
in a local SQLite database (.domain_rank.db in the project root) for fast
<1 ms lookups.

DomCop CSV URL (updated ~monthly):
  https://www.domcop.com/files/top/top10milliondomains.csv.zip

Schema:
  CREATE TABLE domains (
      domain  TEXT PRIMARY KEY,
      rank    INTEGER NOT NULL
  )

Public API:
  lookup(domain: str) -> int | None        # rank integer, or None if not found
  status() -> dict                         # {last_updated, domain_count, db_path}
  download_and_ingest(progress_cb=None)    # blocking; progress_cb(msg: str)
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import time
import zipfile
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root = two levels up from this file (ingestion/scrapers/domain_rank_db.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_PROJECT_ROOT, ".domain_rank.db")

# DomCop publishes a fresh top-10M file each month; URL is stable.
DOMCOP_ZIP_URL = "https://www.domcop.com/files/top/top10milliondomains.csv.zip"
DOMCOP_CSV_NAME = "top10milliondomains.csv"  # filename inside the ZIP

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Open (or create) the SQLite DB and ensure the schema exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent read performance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domains (
            domain  TEXT PRIMARY KEY,
            rank    INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))
    conn.commit()


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(domain: str) -> Optional[int]:
    """
    Return the DomCop rank (integer) for the given domain, or None if not found.
    Domain should be the bare root domain, e.g. "gymshark.com".
    Fast: single indexed primary-key lookup, typically <1 ms.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT rank FROM domains WHERE domain=?", (domain.lower().strip(),)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def status() -> dict:
    """
    Return DB status dict:
      {
        "db_path":      str,
        "last_updated": str | None,   # ISO 8601 UTC, e.g. "2026-01-15T10:30:00Z"
        "domain_count": int,
        "available":    bool,         # True if DB exists and has data
      }
    """
    if not os.path.exists(DB_PATH):
        return {
            "db_path": DB_PATH,
            "last_updated": None,
            "domain_count": 0,
            "available": False,
        }
    try:
        with _get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
            last_updated = _get_meta(conn, "last_updated")
        return {
            "db_path": DB_PATH,
            "last_updated": last_updated,
            "domain_count": count,
            "available": count > 0,
        }
    except Exception:
        return {
            "db_path": DB_PATH,
            "last_updated": None,
            "domain_count": 0,
            "available": False,
        }


def download_and_ingest(progress_cb: Optional[Callable[[str], None]] = None) -> None:
    """
    Download the DomCop top-10M CSV ZIP, parse it, and store all rows in SQLite.

    This is a blocking call that may take 1–3 minutes depending on network speed
    (the ZIP is typically 70–100 MB).

    progress_cb — optional callable that receives human-readable progress strings.

    Raises RuntimeError on download or parse failure.
    """

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _log(f"⬇ Downloading DomCop top-10M from {DOMCOP_ZIP_URL} …")
    _log("  (This may take 1–3 minutes depending on your connection speed)")

    # ── Download ──────────────────────────────────────────────────────────────
    try:
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            resp = client.get(DOMCOP_ZIP_URL)
            resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e

    zip_bytes = resp.content
    _log(f"  Downloaded {len(zip_bytes) / 1_048_576:.1f} MB")

    # ── Parse ZIP → CSV ───────────────────────────────────────────────────────
    _log("📂 Extracting and parsing CSV …")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Find the CSV file in the archive (name may vary slightly)
            csv_filename = None
            for name in zf.namelist():
                if name.endswith(".csv"):
                    csv_filename = name
                    break
            if not csv_filename:
                raise RuntimeError("No CSV found inside downloaded ZIP")
            csv_bytes = zf.read(csv_filename)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Bad ZIP file from DomCop: {e}") from e

    # ── Ingest into SQLite ─────────────────────────────────────────────────────
    _log("💾 Ingesting into SQLite (this takes ~30–60 s for 10M rows) …")

    rows: list[tuple[str, int]] = []
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8", errors="replace")))

    # DomCop CSV format:  rank,domain  (header line present)
    # Example row:  1,google.com
    header_skipped = False
    for row in reader:
        if not header_skipped:
            header_skipped = True
            # Validate header
            if row and row[0].strip().lower() in ("rank", "#"):
                continue   # skip header row
            # If first row looks like data already, process it
        if len(row) < 2:
            continue
        try:
            rank_val = int(row[0].strip())
            domain_val = row[1].strip().lower()
            if domain_val:
                rows.append((domain_val, rank_val))
        except (ValueError, IndexError):
            continue   # skip malformed rows

    _log(f"  Parsed {len(rows):,} domain rows")

    if not rows:
        raise RuntimeError("Parsed 0 rows — DomCop CSV format may have changed")

    # Write to SQLite in one transaction (fast batch insert)
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM domains")   # clear old data
        conn.executemany("INSERT INTO domains (domain, rank) VALUES (?,?)", rows)
        _set_meta(conn, "last_updated", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        _set_meta(conn, "domain_count", str(len(rows)))
        conn.commit()
    finally:
        conn.close()

    _log(f"✅ DomCop database updated — {len(rows):,} domains indexed in {DB_PATH}")
