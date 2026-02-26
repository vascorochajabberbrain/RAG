# How to Get a Shopify Admin API Token

> Use this when you need to get an `shpat_...` Admin API token for a Shopify store
> that uses the new Dev Dashboard (required since Jan 2026).

---

## Why This Is Needed

Shopify's new Dev Dashboard uses OAuth2. The `shpat_...` token is **never shown in any UI**
after install — it must be captured during the OAuth redirect. The `tools/shopify_oauth.py`
script handles this automatically by running a local web server that catches the redirect.

---

## One-Time Setup (Dev Dashboard)

Do this once per app. You don't need to redo it for new stores.

1. Go to **partners.shopify.com** → Dev Dashboard → **"jB RAG Ingestion"**
2. Click **Versions** in the left sidebar
3. Click the active version → edit it
4. Under **Redirect URLs**, add:
   ```
   http://localhost:8888/callback
   ```
5. Save & release the version

---

## Getting the Token (per store)

### Step 1 — Get your app credentials

📍 Dev Dashboard → **"jB RAG Ingestion"** → **Settings** → scroll to **Client credentials**

Copy:
- **Client ID**
- **Client Secret** (click Reveal)

### Step 2 — Run the OAuth receiver script

```bash
cd /Users/johanahlund/PROGRAMMING/RAG
python3 tools/shopify_oauth.py
```

Enter when prompted:
- **Client ID** — from step above
- **Client Secret** — from step above
- **Shop domain** — e.g. `www-supremenutrition-com.myshopify.com`
  (not the vanity domain — use the `.myshopify.com` hostname)

### Step 3 — Install the app on the store

The script opens the Shopify install page in your browser automatically.

1. The store admin opens — click **Install**
2. Shopify redirects to `localhost:8888/callback`
3. The script exchanges the code for a token
4. Terminal prints:
   ```
   ==============================================================
     ACCESS TOKEN:
     shpat_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ==============================================================
   ```
5. Token is auto-saved to `.shopify_stores.json` if a matching store entry exists

### Step 4 — Verify in the app

1. Open **jB RAG Builder** → **🛍 Shopify Stores** tab
2. Find the store → click **Test Connection**
3. Should show: `🟢 Admin API · Shop Name · X products, Y pages`
4. Click **Fetch Now** to ingest products + pages + articles

---

## Finding the Shop's .myshopify.com Domain

If you only know the vanity domain (e.g. `supremenutrition.com`), find the myshopify.com domain by:
- Visiting `supremenutrition.com` → view page source → search for `myshopify.com`
- Or: `supremenutrition.com/admin` → look at the URL after login (shows the myshopify subdomain)

For Supreme Nutrition: `www-supremenutrition-com.myshopify.com`

---

## Scopes Configured

The script requests these scopes:
- `read_products` — products + variants + metafields
- `read_content` — pages + blog articles
- `read_publications` — sales channels

Customers, orders, and inventory are intentionally **excluded**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Browser opened but no redirect to localhost | Make sure `http://localhost:8888/callback` is in the app's redirect URLs (Dev Dashboard → Versions) |
| "Token exchange failed: 400" | Code expired (codes are single-use, ~10 min TTL) — uninstall app from store admin, run script again |
| "No matching store found" | Add the store in 🛍 Shopify Stores tab first, then run the script — or paste token manually via Edit |
| Port 8888 already in use | `lsof -i :8888` to find the process, kill it, then retry |
| Script exits immediately | Make sure you entered all three fields (Client ID, Client Secret, shop domain) |
