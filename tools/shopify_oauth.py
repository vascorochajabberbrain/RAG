#!/usr/bin/env python3
"""
Shopify OAuth token receiver.
Gets an Admin API access token (shpat_...) for a store via the OAuth2 flow.

Usage:
    python tools/shopify_oauth.py

Prerequisites (one-time setup in Dev Dashboard):
    1. Go to partners.shopify.com → Dev Dashboard → your app → Versions → edit version
    2. Add redirect URL: http://localhost:8888/callback
    3. Save & release the version

You will need:
    - Client ID     (Dev Dashboard → Settings → Client credentials)
    - Client Secret (same page — click "Reveal" to show it)
    - Shop domain   (e.g. www-supremenutrition-com.myshopify.com)
"""

import http.server
import json
import os
import sys
import threading
import urllib.parse
import webbrowser

import httpx

PORT = 8888
SCOPES = "read_products,read_content,read_publications"
REDIRECT_URI = f"http://localhost:{PORT}/callback"

# Shared state between main thread and HTTP handler
_state: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stores_path() -> str:
    """Path to .shopify_stores.json in project root (one level up from tools/)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".shopify_stores.json")


def _save_token_to_stores(shop: str, token: str) -> bool:
    """
    Update the matching store entry in .shopify_stores.json if it exists.
    Returns True if a matching store was found and updated.
    """
    path = _stores_path()
    if not os.path.exists(path):
        print("[oauth] .shopify_stores.json not found — token not auto-saved to file.")
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matched = False
    for store in data.get("stores", []):
        store_host = (
            store.get("shop_url", "")
            .replace("https://", "")
            .replace("http://", "")
            .rstrip("/")
        )
        # Match by exact hostname or by the first segment (store name)
        if store_host == shop or store_host.split(".")[0] == shop.split(".")[0]:
            store["access_token"] = token
            matched = True
            print(f"[oauth] ✅ Token saved to store '{store['display_name']}' in .shopify_stores.json")

    if not matched:
        print(f"[oauth] No matching store found for '{shop}' in .shopify_stores.json")
        print("[oauth] → Paste the token manually: 🛍 Shopify Stores tab → Edit → Access token field.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return matched


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _OAuthHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path != "/callback":
            self._respond(404, "<h1>404 — expected /callback</h1>")
            return

        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        shop = params.get("shop", [_state.get("shop", "")])[0]

        if not code:
            msg = "❌ No 'code' parameter received. Try reinstalling the app."
            print(f"\n[oauth] {msg}")
            self._respond(200, f"<h1>{msg}</h1>")
            return

        print(f"\n[oauth] Code received from Shopify for shop: {shop}")
        print("[oauth] Exchanging code for access token...")

        try:
            resp = httpx.post(
                f"https://{shop}/admin/oauth/access_token",
                json={
                    "client_id": _state["client_id"],
                    "client_secret": _state["client_secret"],
                    "code": code,
                },
                timeout=15,
            )
            resp.raise_for_status()
            token = resp.json().get("access_token", "")
            if not token:
                raise ValueError(f"No access_token in response: {resp.text}")

        except Exception as exc:
            print(f"[oauth] ❌ Token exchange failed: {exc}")
            self._respond(200, f"<h1>❌ Token exchange failed</h1><pre>{exc}</pre>")
            return

        # Print token prominently
        bar = "=" * 62
        print(f"\n{bar}")
        print(f"  ACCESS TOKEN:")
        print(f"  {token}")
        print(f"{bar}\n")

        # Auto-save to .shopify_stores.json
        _save_token_to_stores(shop, token)

        _state["token"] = token

        self._respond(
            200,
            "<h1>✅ Token captured!</h1>"
            "<p>Check your terminal — the token has been printed there.</p>"
            f"<p>Hint: <code>{token[:12]}...{token[-4:]}</code></p>"
            "<p>You can close this browser tab.</p>",
        )

        # Shut down the server from a background thread (can't block handler)
        threading.Thread(target=_state["server"].shutdown, daemon=True).start()

    def _respond(self, status: int, html: str):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress default request logging noise


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 62)
    print("  Shopify OAuth Token Receiver — jB RAG Builder")
    print("=" * 62)
    print()
    print("You need your app's Client ID and Client Secret from:")
    print("  partners.shopify.com → Dev Dashboard → your app → Settings")
    print()

    client_id = input("Client ID     : ").strip()
    client_secret = input("Client Secret : ").strip()
    shop = (
        input("Shop domain   : ")
        .strip()
        .replace("https://", "")
        .replace("http://", "")
        .rstrip("/")
    )

    if not client_id or not client_secret or not shop:
        print("\n❌ All three fields are required.")
        sys.exit(1)

    _state["client_id"] = client_id
    _state["client_secret"] = client_secret
    _state["shop"] = shop

    # Build the Shopify OAuth authorize URL
    install_url = (
        f"https://{shop}/admin/oauth/authorize"
        f"?client_id={urllib.parse.quote(client_id)}"
        f"&scope={urllib.parse.quote(SCOPES)}"
        f"&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
        f"&state=jbrag"
    )

    server = http.server.HTTPServer(("", PORT), _OAuthHandler)
    _state["server"] = server

    print(f"\n[oauth] Local server listening on http://localhost:{PORT}/callback")
    print("[oauth] Opening Shopify install page in your browser...")
    print()

    webbrowser.open(install_url)

    print("If the browser didn't open automatically, paste this URL:")
    print(f"  {install_url}")
    print()
    print("Waiting for Shopify to redirect back (click Install in the browser)...")
    print()

    server.serve_forever()

    token = _state.get("token")
    if token:
        print("[oauth] All done! You can now test the connection in the 🛍 Shopify Stores tab.")
    else:
        print("[oauth] No token captured. Check the output above for errors.")


if __name__ == "__main__":
    main()
