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


def main():
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency. Install with: pip install uvicorn fastapi")
        sys.exit(1)

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"Opening RAG interface at {url}")
    print("Close this window or press Ctrl+C to stop the server.")
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
