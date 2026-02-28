#!/usr/bin/env python3
"""
Headless server launcher — starts uvicorn only, no browser.
Used by launch_rag.sh; the shell script handles browser opening.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Load .env
_env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
dev_mode = os.environ.get("RAG_DEV_MODE", "") == "1"

if dev_mode:
    # Use uvicorn CLI for reload mode — avoids socket-binding conflict
    # that occurs when uvicorn.run(reload=True) is called from a script.
    import subprocess
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "web.app:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--reload",
    ])
else:
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=port)
