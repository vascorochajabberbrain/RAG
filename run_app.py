#!/usr/bin/env python3
"""
Launch the RAG web interface. Starts the server and opens the app in your browser.
Double-click this file or run: python run_app.py
"""
import os
import sys
import time
import webbrowser
import threading

# Run from project root (directory where this script lives)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Load .env file if present
_env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def _kill_port(port):
    """Kill any Python/uvicorn process listening on the given port (skips Chrome etc)."""
    import subprocess
    import signal
    import socket
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-n", "-P"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines()[1:]:  # skip header row
            parts = line.split()
            if len(parts) < 2:
                continue
            cmd = parts[0].lower()
            pid = parts[1]
            # Only kill Python/uvicorn processes — never Chrome, Safari, etc.
            if any(x in cmd for x in ("python", "uvicorn")):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        time.sleep(0.8)  # give OS time to start releasing the port
        # Wait for port to be free (up to 3 seconds)
        for _ in range(6):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    time.sleep(0.5)
            except OSError:
                break  # port is free
    except Exception:
        pass


def main():
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency. Install with: pip install uvicorn fastapi")
        sys.exit(1)

    host = "127.0.0.1"
    port = 8000

    # Kill anything already on our port, then wait for it to free up
    _kill_port(port)

    url = f"http://{host}:{port}"

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"Opening RAG interface at {url}")
    print("Close this window or press Ctrl+C to stop the server.")
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
