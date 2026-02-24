#!/usr/bin/env python3
"""
Small keep-alive script for Qdrant Cloud.

What it does:
- Connects to Qdrant using your existing qdrant_utils.get_qdrant_connection().
- Performs a very cheap operation (list collections) so the cluster counts as "active".
- Logs the run locally AND shows a macOS notification so you know it ran.

Usage:
- From your venv:  source .venv/bin/activate && python keep_qdrant_alive.py
- Or schedule it (e.g. with cron or launchd) to run once a day.
"""

import datetime
import subprocess
import traceback

from qdrant_utils import get_qdrant_connection


LOG_FILE = "keep_qdrant_alive.log"


def notify_mac(message: str, title: str = "RAG / Qdrant keep-alive") -> None:
    """
    Show a macOS notification using AppleScript.
    If anything fails (e.g. not on macOS), we silently ignore it.
    """
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            check=False,
        )
    except Exception:
        # Non-macOS or osascript not available – ignore
        pass


def log_line(text: str) -> None:
    """Append a line to the local log file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        # Logging should never break the script
        pass


def ping_qdrant() -> None:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        conn = get_qdrant_connection()
        collections = conn.get_collections()
        names = [c.name for c in collections.collections]
        msg = f"[{now}] Qdrant keep-alive OK. Collections: {', '.join(names) or '(none)'}"
        print(msg)
        log_line(msg)
        notify_mac("Ping OK at " + now)
    except Exception:
        err = f"[{now}] Qdrant keep-alive FAILED:\n{traceback.format_exc()}"
        print(err)
        log_line(err)
        notify_mac("Ping FAILED at " + now)


if __name__ == "__main__":
    ping_qdrant()

