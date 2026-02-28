"""
tech_detector.py — Ecommerce Site Technology Detector

Analyses a URL and returns a SiteReport with detected:
  - Ecommerce platform (Shopify, WooCommerce, Magento, …)
  - Chatbot provider (Intercom, Zendesk, Gorgias, …)
  - CMS (WordPress, Drupal, …)
  - SSL, social links, payment signals, blog presence
  - Tranco global rank (free, no API key, top ~1M sites)

Two-pass detection:
  Pass 1 — Fast httpx fetch: parse HTML + response headers
  Pass 2 — Playwright JS eval: only when Pass 1 is inconclusive
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Fingerprint databases
# ---------------------------------------------------------------------------

PLATFORM_FINGERPRINTS: dict[str, dict] = {
    "Shopify": {
        "script_contains":  ["cdn.shopify.com", "shopify-analytics", "Shopify.theme"],
        "html_contains":    ["Shopify.shop", "window.ShopifyAnalytics", "shopify-features"],
        "meta_generator":   ["shopify"],
        "api_check":        "/products.json",   # 200 + {"products":[...]} → confirmed Shopify
    },
    "WooCommerce": {
        "script_contains":  ["woocommerce"],
        "html_contains":    ["wc_add_to_cart_params", "?wc-ajax=", "woocommerce-page"],
        "meta_generator":   ["woocommerce"],
        "css_contains":     [".woocommerce"],
    },
    "Magento": {
        "script_contains":  ["/pub/static/", "/static/version"],
        "html_contains":    ["Mage.Cookies", "text/x-magento-init", "mage/cookies"],
        "meta_generator":   ["magento"],
        "header_keys":      ["X-Magento-Cache-Control", "X-Magento-Tags"],
    },
    "BigCommerce": {
        "script_contains":  ["cdn11.bigcommerce.com", "cdn.bigcommerce.com"],
        "html_contains":    ["window.BigCommerce", "BigCommerce."],
        "meta_generator":   ["bigcommerce"],
    },
    "PrestaShop": {
        "script_contains":  [],
        "html_contains":    ["window.prestashop", "prestashop"],
        "meta_generator":   ["prestashop"],
        "css_contains":     ["prestashop"],
    },
    "Wix": {
        "script_contains":  ["static.parastorage.com", "wixstatic.com"],
        "html_contains":    ["window.wixData", "wix-warmup-data"],
        "meta_generator":   ["wix"],
        "header_keys":      ["X-Wix-Request-Id"],
    },
    "Squarespace": {
        "script_contains":  ["static.squarespace.com"],
        "html_contains":    ["Static.SQUARESPACE_CONTEXT", "squarespace.com"],
        "meta_generator":   ["squarespace"],
    },
    "WordPress": {
        "script_contains":  ["wp-content/", "wp-includes/", "wp-emoji-release"],
        "html_contains":    ["wp-content/", "wp-includes/"],
        "meta_generator":   ["wordpress"],
    },
    "Drupal": {
        "script_contains":  ["sites/default/files/", "misc/drupal.js"],
        "html_contains":    ["Drupal.settings", "drupal.org"],
        "meta_generator":   ["drupal"],
        "header_keys":      ["X-Generator"],   # value: Drupal
    },
}

CHATBOT_FINGERPRINTS: dict[str, dict] = {
    "Intercom":    {"script_contains": ["widget.intercom.io", "intercomcdn.com"], "js_global": "Intercom"},
    "Zendesk":     {"script_contains": ["static.zdassets.com", "zopim.com"],      "js_global": "zopim"},
    "Drift":       {"script_contains": ["js.driftt.com", "drift.js"],             "js_global": "drift"},
    "HubSpot":     {"script_contains": ["js.hs-scripts.com", "hs-analytics.net"], "js_global": "HubSpotConversations"},
    "Crisp":       {"script_contains": ["client.crisp.chat"],                     "js_global": "$crisp"},
    "Tidio":       {"script_contains": ["code.tidio.co"],                         "js_global": "tidioChatApi"},
    "LiveChat":    {"script_contains": ["cdn.livechatinc.com"],                   "js_global": "LC_API"},
    "Freshchat":   {"script_contains": ["wchat.freshchat.com"],                   "js_global": "fcWidget"},
    "Gorgias":     {"script_contains": ["config.gorgias.chat"],                   "js_global": "gorgias"},
    "Tawk.to":     {"script_contains": ["embed.tawk.to"],                         "js_global": "Tawk_API"},
    "Olark":       {"script_contains": ["static.olark.com"],                      "js_global": "olark"},
    "Jivochat":    {"script_contains": ["code.jivosite.com"],                     "js_global": "jivo_api"},
    "Jabberbrain": {"script_contains": ["jabberbrain"],                           "js_global": "jabberbrain"},
    # ── Extended ecommerce & enterprise providers ──────────────────────────────
    "Re:amaze":          {"script_contains": ["cdn.reamaze.com/assets/reamaze"],
                                                                                   "js_global": "reamaze"},
    "Help Scout Beacon": {"script_contains": ["beacon-v2.helpscout.net"],          "js_global": "Beacon"},
    "Freshdesk":         {"script_contains": ["widget.freshworks.com/widgets/",
                                              "assets.freshdesk.com/widget"],      "js_global": "FreshworksWidget"},
    "Kustomer":          {"script_contains": ["cdn.kustomerapp.com"],              "js_global": "Kustomer"},
    "Gladly":            {"script_contains": ["cdn.gladly.com/chat-sdk"],          "js_global": "gladlyConfig"},
    "Chatwoot":          {"script_contains": ["cdn.chatwoot.com"],                 "js_global": "chatwootSDK"},
    "Botpress":          {"script_contains": ["cdn.botpress.cloud/webchat",
                                              "botman-web-widget"],                "js_global": "botpressWebChat"},
    "Salesforce Chat":   {"script_contains": ["service.force.com",
                                              "salesforceliveagent",
                                              "esw.min.js"],                       "js_global": "embedded_svc"},
    "Kommunicate":       {"script_contains": ["widgetapi.kommunicate.io"],         "js_global": "kommunicateSettings"},
    "ManyChat":          {"script_contains": ["widget.manychat.com"],              "js_global": ""},
    "Userlike":          {"script_contains": ["userlike.com/widget/",
                                              "cdn.userlike.com"],                 "js_global": "userlike"},
    "LivePerson":        {"script_contains": ["lpcdn.lpsnmedia.net",
                                              "liveperson.net"],                   "js_global": "lpTag"},
    "Smartsupp":         {"script_contains": ["smartsupp.com/loader.js"],          "js_global": "smartsupp"},
    "Chatra":            {"script_contains": ["call.chatra.io"],                   "js_global": "Chatra"},
    "Pure Chat":         {"script_contains": ["app.purechat.com/le/"],             "js_global": ""},
    "SnapEngage":        {"script_contains": ["snapengage.com/build/js"],          "js_global": "SnapABug"},
    "Front Chat":        {"script_contains": ["chat-assets.frontapp.com"],         "js_global": "Front"},
    "Dixa":              {"script_contains": ["messenger.dixa.io"],                "js_global": "dixaMessenger"},
    "Zoho SalesIQ":      {"script_contains": ["salesiq.zoho.com/widget"],          "js_global": ""},
    "Pipedrive Chat":    {"script_contains": ["leadbooster-chat.pipedrive.com"],   "js_global": "LeadBooster"},
    "Richpanel":         {"script_contains": ["cdn.richpanel.com"],                "js_global": "richpanel"},
    "Podium":            {"script_contains": ["assets.podium.com/widget"],         "js_global": "podium"},
    "Octane AI":         {"script_contains": ["messenger.octaneai.com"],           "js_global": ""},
    "Gist":              {"script_contains": ["widget.getgist.com"],               "js_global": "gist"},
    "Landbot":           {"script_contains": ["cdn.landbot.io",
                                              "static.landbot.io"],                "js_global": "landbot"},
    "Userflow":          {"script_contains": ["userflow.js",
                                              "userflow.io"],                      "js_global": "userflow"},
    "Customerly":        {"script_contains": ["widget.customerly.io"],             "js_global": "customerly"},
    "Trengo":            {"script_contains": ["static.widget.trengo.eu"],          "js_global": "Trengo"},
    "Helpcrunch":        {"script_contains": ["widget.helpcrunch.com"],            "js_global": "HelpCrunch"},
    "Acquire":           {"script_contains": ["app.acquire.io/widget"],            "js_global": "Acquire"},
    "ChatBot.com":       {"script_contains": ["cdn.chatbot.com"],                  "js_global": "ChatBotWidget"},
}

PAYMENT_FINGERPRINTS: dict[str, list[str]] = {
    "Stripe":      ["js.stripe.com", "stripe.com/v3", "data-stripe"],
    "PayPal":      ["paypal.com/sdk", "paypalobjects.com", "data-paypal"],
    "Klarna":      ["klarna.com", "js.klarna.com"],
    "Mollie":      ["mollie.com"],
    "Adyen":       ["adyen.com"],
    "Braintree":   ["braintreegateway.com", "braintree-web"],
    "Square":      ["squareup.com", "square.com/payments"],
    "Afterpay":    ["afterpay.com", "clearpay.co.uk"],
    "Multibanco":  ["multibanco"],
    "MB Way":      ["mbway"],
}

SOCIAL_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "linkedin.com", "youtube.com", "pinterest.com",
    "snapchat.com", "threads.net",
]

BLOG_PATH_SIGNALS = ["/blog", "/news", "/articles", "/journal", "/magazine", "/insights"]

# Maps chatbot provider name → category (used for table badge in UI)
CHATBOT_CATEGORY: dict[str, str] = {
    # ── Primarily automated bots ────────────────────────────────────────────
    "Landbot":           "Bot",
    "Botpress":          "Bot",
    "ChatBot.com":       "Bot",
    "Octane AI":         "Bot",
    "ManyChat":          "Bot",
    "CustomGPT":         "Bot",
    "Jabberbrain":       "Bot",
    # ── Primarily live chat (human agent) ───────────────────────────────────
    "LiveChat":          "Live Chat",
    "Olark":             "Live Chat",
    "Pure Chat":         "Live Chat",
    "SnapEngage":        "Live Chat",
    "Gladly":            "Live Chat",
    "Kustomer":          "Live Chat",
    "Salesforce Chat":   "Live Chat",
    "LivePerson":        "Live Chat",
    "Help Scout Beacon": "Live Chat",
    "Front Chat":        "Live Chat",
    "Userlike":          "Live Chat",
    "Dixa":              "Live Chat",
    "Jivochat":          "Live Chat",
    # ── Hybrid (bot + live handoff) ─────────────────────────────────────────
    "Intercom":          "Hybrid",
    "Zendesk":           "Hybrid",
    "Drift":             "Hybrid",
    "HubSpot":           "Hybrid",
    "Crisp":             "Hybrid",
    "Tidio":             "Hybrid",
    "Freshchat":         "Hybrid",
    "Freshdesk":         "Hybrid",
    "Gorgias":           "Hybrid",
    "Tawk.to":           "Hybrid",
    "Chatwoot":          "Hybrid",
    "Kommunicate":       "Hybrid",
    "Re:amaze":          "Hybrid",
    "Richpanel":         "Hybrid",
    "Trengo":            "Hybrid",
    "Helpcrunch":        "Hybrid",
    "Smartsupp":         "Hybrid",
    "Chatra":            "Hybrid",
    "Pipedrive Chat":    "Hybrid",
    "Zoho SalesIQ":      "Hybrid",
    "Gist":              "Hybrid",
    "Customerly":        "Hybrid",
    "Acquire":           "Hybrid",
    "Podium":            "Hybrid",
    "Userflow":          "Hybrid",
    "Messagely":         "Hybrid",
    "Postscript":        "Hybrid",
    "Klaviyo Chat":      "Hybrid",
    # "Unknown chatbot" and "None detected" → not in dict → empty string (no badge)
}

# ---------------------------------------------------------------------------
# SiteReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class SiteReport:
    url: str
    ssl: bool = False
    platform: str = "Unknown"
    platform_confidence: str = "none"   # "high" | "medium" | "low" | "none"
    chatbot: str = "None detected"
    chatbot_signal: str = ""            # raw signal that triggered detection
    chatbot_category: str = ""          # "Bot" | "Live Chat" | "Hybrid" | ""
    cms: str = ""                       # separate CMS if different from platform
    payments: list[str] = field(default_factory=list)
    social_links: list[str] = field(default_factory=list)
    has_blog: bool = False
    # ── Contact signals ───────────────────────────────────────────────────────
    contact_form: bool = False                               # <form> with email input
    contact_mailto: list[str] = field(default_factory=list) # emails from mailto: links
    contact_whatsapp: list[str] = field(default_factory=list)  # numbers from wa.me links
    contact_phone: list[str] = field(default_factory=list)  # numbers from tel: links
    # ── Rank ──────────────────────────────────────────────────────────────────
    rank: str = "N/A"           # "#42,500 (Tranco)" | "#4,200,000 (DomCop)" | "Not ranked" | "N/A"
    rank_source: str = ""       # "Tranco" | "DomCop" | ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "url":                self.url,
            "ssl":                self.ssl,
            "platform":           self.platform,
            "platform_confidence": self.platform_confidence,
            "chatbot":            self.chatbot,
            "chatbot_signal":     self.chatbot_signal,
            "chatbot_category":   self.chatbot_category,
            "cms":                self.cms,
            "payments":           ", ".join(self.payments),
            "social_links":       ", ".join(self.social_links),
            "has_blog":           self.has_blog,
            "contact_form":       self.contact_form,
            "contact_mailto":     ", ".join(self.contact_mailto),
            "contact_whatsapp":   ", ".join(self.contact_whatsapp),
            "contact_phone":      ", ".join(self.contact_phone),
            "rank":               self.rank,
            "rank_source":        self.rank_source,
            "error":              self.error,
        }

# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers=_HEADERS,
        follow_redirects=True,
        timeout=15,
        verify=False,  # some sites have self-signed / expired certs — don't fail
    )

# ---------------------------------------------------------------------------
# Pass 1 — HTML fingerprinting
# ---------------------------------------------------------------------------

def _html_detect_platform(html: str, headers: dict) -> tuple[str, str]:
    """
    Returns (platform_name, confidence): "high" | "medium" | "low" | "none".
    Matches highest-score platform.
    """
    html_lower = html.lower()
    scores: dict[str, int] = {}

    for platform, fp in PLATFORM_FINGERPRINTS.items():
        score = 0

        for sig in fp.get("script_contains", []):
            if sig.lower() in html_lower:
                score += 2

        for sig in fp.get("html_contains", []):
            if sig.lower() in html_lower:
                score += 2

        for sig in fp.get("css_contains", []):
            if sig.lower() in html_lower:
                score += 1

        for gen in fp.get("meta_generator", []):
            # look for <meta name="generator" content="...{gen}...">
            if re.search(rf'generator["\s\w]*{re.escape(gen)}', html_lower):
                score += 3

        for hkey in fp.get("header_keys", []):
            if hkey.lower() in {k.lower() for k in headers}:
                score += 2

        if score > 0:
            scores[platform] = score

    if not scores:
        return "Unknown", "none"

    best = max(scores, key=lambda k: scores[k])
    s = scores[best]
    confidence = "high" if s >= 5 else "medium" if s >= 3 else "low"
    return best, confidence


def _html_detect_chatbot(html: str, soup: BeautifulSoup) -> tuple[str, str]:
    """
    Returns (chatbot_name, signal_that_matched).
    Detection order:
      1. Named provider CDN fingerprints (script src substrings)
      2. onclick / link-text scan (deferred widgets, "Chat with us" buttons)
      3. Generic DOM signals (class names, iframe src, window globals)
    """
    html_lower = html.lower()
    # ── 1. Named provider scan ────────────────────────────────────────────────
    for name, fp in CHATBOT_FINGERPRINTS.items():
        for sig in fp.get("script_contains", []):
            if sig.lower() in html_lower:
                return name, sig
    # ── 2. onclick / link-text scan (deferred/click-triggered widgets) ────────
    onclick_result = _html_detect_chat_links(soup)
    if onclick_result[0] != "None detected":
        return onclick_result
    # ── 3. Generic DOM signal fallback ────────────────────────────────────────
    return _html_detect_chatbot_generic(soup)


# Known chatbot JavaScript API calls that appear in onclick/href attributes.
# Order matters — more specific providers first.
_CHAT_ONCLICK_APIS: list[tuple[str, str]] = [
    # (provider_name, case-insensitive substring to find in onclick/href)
    ("Intercom",    "Intercom("),
    ("Tawk.to",     "Tawk_API."),
    ("Crisp",       "$crisp.push"),
    ("LiveChat",    "LC_API."),
    ("HubSpot",     "HubSpotConversations"),
    ("Drift",       "drift.api"),
    ("Freshchat",   "fcWidget."),
    ("Tidio",       "tidioChatApi."),
    ("Jabberbrain", "jabberbrain"),
    ("Zendesk",     "zE("),
    ("Gorgias",     "GorgiasChat."),
]

# Link/button text phrases that strongly indicate a chat trigger button.
# All lowercase for case-insensitive comparison.
_CHAT_LINK_PHRASES: list[str] = [
    "chat with us",
    "live chat",
    "chat now",
    "start chat",
    "open chat",
    "start a chat",
    "talk to us",
    "message us",
    "contact us via chat",
    "get support via chat",
    "chat to us",
    "chat with support",
]


def _html_detect_chat_links(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Detect chat-trigger links/buttons that open a widget on click (deferred loading).
    Checks onclick attributes for known provider API calls, then falls back to
    link text phrase matching on non-navigational links.
    Returns (provider_or_"Unknown chatbot", signal) or ("None detected", "").
    """
    for tag in soup.find_all(["a", "button", "div", "span"]):
        onclick = (tag.get("onclick") or "").strip()
        href    = (tag.get("href") or "").strip()

        # 1. onclick or javascript: href contains a known provider API call
        combined_js = onclick + " " + (href if href.startswith("javascript:") else "")
        for provider, api_call in _CHAT_ONCLICK_APIS:
            if api_call.lower() in combined_js.lower():
                return provider, f"onclick/href contains {api_call!r}"

        # 2. Link text matches a chat phrase AND the link is non-navigational
        text = tag.get_text(separator=" ").strip().lower()
        for phrase in _CHAT_LINK_PHRASES:
            if phrase in text:
                is_non_nav = href in ("#", "", "javascript:void(0)", "javascript:;") or bool(onclick)
                if is_non_nav:
                    return "Unknown chatbot", f"link text={phrase!r}"
                break  # phrase matched but link is navigational — skip

    return "None detected", ""


# Generic signal patterns (provider-agnostic, compound terms to reduce false positives)
_GENERIC_CHAT_SCRIPT_TERMS = [
    "livechat",         # e.g. cdn.example.com/livechat.js
    "/chat.js",         # e.g. /assets/chat.js
    "/chatbot.js",      # e.g. /js/chatbot.js
    "/chat-widget",     # e.g. /dist/chat-widget.min.js
    "chat-sdk",         # e.g. chat-sdk.bundle.js
    "chat_widget",      # e.g. chat_widget_v2.js
    "chatwidget",       # e.g. chatwidget.js
]

_GENERIC_CHAT_CLASS_TERMS = [
    "chat-widget",
    "chat-bubble",
    "chat-button",
    "chat-launcher",
    "chat-toggle",
    "chat-icon",
    "live-chat",
    "chatwidget",
]


def _html_detect_chatbot_generic(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Generic chatbot detection — fires "Unknown chatbot" when no named provider matched.
    Checks 4 signal types ordered from highest to lowest specificity.
    Returns ("Unknown chatbot", signal) or ("None detected", "").
    """
    # 1. Script src — compound path patterns that strongly imply a chat tool
    for tag in soup.find_all("script", src=True):
        src = tag["src"].lower()
        for term in _GENERIC_CHAT_SCRIPT_TERMS:
            if term in src:
                return "Unknown chatbot", f"script[src*={term!r}]"

    # 2. iframes — src containing "chat" is a strong signal (chat iframes are common)
    for tag in soup.find_all("iframe", src=True):
        src = tag["src"].lower()
        if "chat" in src or ("widget" in src and "support" in src):
            return "Unknown chatbot", f"iframe src={tag['src'][:60]!r}"

    # 3. Element id/class — compound multi-word class names used by many providers
    for tag in soup.find_all(True, id=True):
        id_val = tag["id"].lower()
        for term in _GENERIC_CHAT_CLASS_TERMS:
            if term in id_val:
                return "Unknown chatbot", f"id={tag['id']!r}"
    for tag in soup.find_all(True, class_=True):
        classes = " ".join(tag.get("class", [])).lower()
        for term in _GENERIC_CHAT_CLASS_TERMS:
            if term in classes:
                return "Unknown chatbot", f"class contains {term!r}"

    # 4. Inline script — window.*Chat* / window.*chat* variable assignments
    for tag in soup.find_all("script", src=False):
        text = tag.get_text()
        if re.search(r'window\.[a-zA-Z]*[Cc]hat[a-zA-Z]*\s*[=\(]', text):
            match = re.search(r'window\.([a-zA-Z]*[Cc]hat[a-zA-Z]*)', text)
            signal = f"window.{match.group(1)}" if match else "window.*Chat*"
            return "Unknown chatbot", signal

    return "None detected", ""


def _html_detect_payments(html: str) -> list[str]:
    html_lower = html.lower()
    found = []
    for name, signals in PAYMENT_FINGERPRINTS.items():
        for sig in signals:
            if sig.lower() in html_lower:
                found.append(name)
                break
    return found


def _html_detect_social(soup: BeautifulSoup) -> list[str]:
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        for domain in SOCIAL_DOMAINS:
            if domain in href:
                # Keep just the domain name as label
                found.append(domain.split(".")[0].capitalize())
                break
    # deduplicate, preserve order
    seen = set()
    result = []
    for s in found:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _html_detect_blog(html: str, soup: BeautifulSoup) -> bool:
    html_lower = html.lower()
    for path in BLOG_PATH_SIGNALS:
        if path in html_lower:
            return True
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").lower()
        for path in BLOG_PATH_SIGNALS:
            if path in href:
                return True
    return False


# ---------------------------------------------------------------------------
# Contact signal detection
# ---------------------------------------------------------------------------

def _html_detect_contact_form(soup: BeautifulSoup) -> bool:
    """
    Detect embedded email/contact form.
    Returns True if a <form> with an email-type input or an email-hinted field is found.
    """
    for form in soup.find_all("form"):
        # Signal 1: explicit <input type="email">
        if form.find("input", {"type": "email"}):
            return True
        # Signal 2: any input/textarea whose name/id/placeholder/aria-label hints at email
        for inp in form.find_all(["input", "textarea"]):
            attrs = " ".join([
                inp.get("name", ""),
                inp.get("id", ""),
                inp.get("placeholder", ""),
                inp.get("aria-label", ""),
            ]).lower()
            if "email" in attrs:
                return True
    return False


def _html_detect_mailto(soup: BeautifulSoup) -> list[str]:
    """Return deduplicated list of email addresses found in mailto: links."""
    emails: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().startswith("mailto:"):
            email = href[7:].split("?")[0].strip()
            if "@" in email and email not in seen:
                seen.add(email)
                emails.append(email)
    return emails


def _html_detect_whatsapp(soup: BeautifulSoup) -> list[str]:
    """Return deduplicated list of phone numbers from wa.me/ or api.whatsapp.com links."""
    numbers: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        number: str | None = None
        if "wa.me/" in href:
            m = re.search(r"wa\.me/(\d+)", href)
            if m:
                number = "+" + m.group(1)
        elif "api.whatsapp.com" in href:
            m = re.search(r"phone=(\d+)", href)
            if m:
                number = "+" + m.group(1)
        if number and number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def _html_detect_phone(soup: BeautifulSoup) -> list[str]:
    """Return deduplicated list of phone numbers from tel: links."""
    phones: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().startswith("tel:"):
            phone = href[4:].strip()
            if any(c.isdigit() for c in phone) and phone not in seen:
                seen.add(phone)
                phones.append(phone)
    return phones


# ---------------------------------------------------------------------------
# Pass 2 — Playwright JS globals (only when platform unknown)
# ---------------------------------------------------------------------------

async def _playwright_detect(url: str) -> tuple[str, str, str, str]:
    """
    Returns (platform, confidence, chatbot, chatbot_signal) via JS evaluation.
    Falls back to ("Unknown","none","None detected","") on any error.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        return "Unknown", "none", "None detected", ""

    platform, confidence, chatbot, chatbot_signal = "Unknown", "none", "None detected", ""
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)

            # Check known platform JS globals
            for plat, js_var in [
                ("Shopify",     "window.Shopify"),
                ("Magento",     "window.Mage"),
                ("BigCommerce", "window.BigCommerce"),
                ("Wix",         "window.wixData"),
                ("Squarespace", "window.Static && window.Static.SQUARESPACE_CONTEXT"),
                ("PrestaShop",  "window.prestashop"),
            ]:
                try:
                    val = await page.evaluate(f"typeof {js_var} !== 'undefined'")
                    if val:
                        platform = plat
                        confidence = "high"
                        break
                except Exception:
                    continue

            # Check chatbot JS globals
            for name, fp in CHATBOT_FINGERPRINTS.items():
                js_var = fp.get("js_global", "")
                if not js_var:
                    continue
                try:
                    val = await page.evaluate(f"typeof window.{js_var} !== 'undefined'")
                    if val:
                        chatbot = name
                        chatbot_signal = f"window.{js_var}"
                        break
                except Exception:
                    continue

            await browser.close()
    except Exception:
        pass

    return platform, confidence, chatbot, chatbot_signal

# ---------------------------------------------------------------------------
# Tranco rank lookup (free, no API key, 1 req/sec limit)
# ---------------------------------------------------------------------------

# Lock to serialise Tranco API calls — rate limit is 1 req/sec
_tranco_lock = threading.Lock()


def _fetch_tranco_rank(domain: str) -> str:
    """
    Look up Tranco rank for a domain. Free, no auth required.
    API: GET https://tranco-list.eu/api/ranks/domain/{domain}
    Rate limit: 1 req/sec (enforced via module-level lock + sleep).
    Returns: "#42,500" | "Not ranked" | "N/A"
    Only covers top ~1M sites (30-day rolling average).
    """
    with _tranco_lock:
        try:
            r = httpx.get(
                f"https://tranco-list.eu/api/ranks/domain/{domain}",
                timeout=8,
                headers={"User-Agent": "jabberbrain-site-analyzer/1.0"},
            )
            if r.status_code == 200:
                data = r.json()
                ranks = data.get("ranks", [])
                if ranks:
                    latest = ranks[0].get("rank")
                    if latest:
                        return f"#{latest:,}"
                return "Not ranked"
            elif r.status_code == 429:
                return "Rate limited"
        except Exception:
            pass
        finally:
            time.sleep(1.1)   # stay within 1 req/sec even on error
    return "N/A"


def _fetch_rank(domain: str) -> tuple[str, str]:
    """
    Two-source rank lookup.

    1. Tranco API (live, top ~1M, free, 1 req/sec)
    2. DomCop local SQLite fallback (top 10M, if DB is present)

    Returns (rank_str, source) where:
      rank_str: "#42,500" | "#4,200,000" | "Not ranked" | "N/A"
      source:   "Tranco" | "DomCop" | ""
    """
    # ── Step 1: Tranco ────────────────────────────────────────────────────────
    tranco = _fetch_tranco_rank(domain)
    if tranco not in ("Not ranked", "N/A", "Rate limited"):
        # Got a real Tranco rank — use it
        return tranco, "Tranco"

    # ── Step 2: DomCop local fallback ─────────────────────────────────────────
    try:
        from ingestion.scrapers.domain_rank_db import lookup as _domcop_lookup
        dc_rank = _domcop_lookup(domain)
        if dc_rank is not None:
            return f"#{dc_rank:,}", "DomCop"
    except ImportError:
        pass  # domain_rank_db not available — skip silently

    # Nothing found
    if tranco == "Rate limited":
        return "Rate limited", ""
    return "Not ranked", ""

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_site(url: str) -> SiteReport:
    """
    Synchronous detection — runs Pass 1 (httpx) always,
    Pass 2 (Playwright) only if platform is still unknown after Pass 1.
    Fetches Tranco rank (free, no API key needed).
    """
    import asyncio

    # Normalise URL
    url = url.strip().rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url

    report = SiteReport(url=url)
    report.ssl = url.startswith("https://")

    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path  # e.g. "store.peixefresco.com.pt"

    try:
        with _make_client() as client:
            # ── Pass 1: fetch homepage ──────────────────────────────────────
            try:
                resp = client.get(url, timeout=15)
                html = resp.text
                headers = dict(resp.headers)
                report.ssl = str(resp.url).startswith("https://")
            except Exception as e:
                report.error = f"Fetch failed: {e}"
                return report

            soup = BeautifulSoup(html, "html.parser")

            # Platform
            platform, confidence = _html_detect_platform(html, headers)
            report.platform = platform
            report.platform_confidence = confidence

            # Chatbot + category
            chatbot, chatbot_signal = _html_detect_chatbot(html, soup)
            report.chatbot = chatbot
            report.chatbot_signal = chatbot_signal
            report.chatbot_category = CHATBOT_CATEGORY.get(chatbot, "")

            # Payments, social, blog
            report.payments = _html_detect_payments(html)
            report.social_links = _html_detect_social(soup)
            report.has_blog = _html_detect_blog(html, soup)

            # Contact signals (form, mailto, WhatsApp, phone)
            report.contact_form     = _html_detect_contact_form(soup)
            report.contact_mailto   = _html_detect_mailto(soup)
            report.contact_whatsapp = _html_detect_whatsapp(soup)
            report.contact_phone    = _html_detect_phone(soup)

            # Separate CMS label if platform is an ecommerce one built on WP
            if platform == "WooCommerce":
                report.cms = "WordPress"
            elif platform == "WordPress":
                report.platform = "WordPress (generic)"
                report.cms = "WordPress"

            # ── Shopify API confirmation ────────────────────────────────────
            if platform in ("Shopify", "Unknown"):
                try:
                    r2 = client.get(f"{url}/products.json?limit=1", timeout=8)
                    if r2.status_code == 200:
                        data = r2.json()
                        if isinstance(data, dict) and "products" in data:
                            report.platform = "Shopify"
                            report.platform_confidence = "high"
                except Exception:
                    pass

    except Exception as e:
        report.error = f"Detection error: {e}"
        return report

    # ── Pass 2: Playwright JS globals (only if still unknown) ──────────────
    if report.platform_confidence in ("none", "low") or report.chatbot == "None detected":
        try:
            loop = asyncio.new_event_loop()
            pl_platform, pl_confidence, pl_chatbot, pl_chatbot_signal = loop.run_until_complete(
                _playwright_detect(url)
            )
            loop.close()

            if pl_platform != "Unknown" and report.platform_confidence in ("none", "low"):
                report.platform = pl_platform
                report.platform_confidence = pl_confidence

            if pl_chatbot != "None detected" and report.chatbot == "None detected":
                report.chatbot = pl_chatbot
                report.chatbot_signal = pl_chatbot_signal
                report.chatbot_category = CHATBOT_CATEGORY.get(pl_chatbot, "")

        except Exception:
            pass  # Playwright unavailable — not fatal

    # ── Rank lookup: Tranco (live) → DomCop (local fallback) ──────────────
    # Strip subdomains — both Tranco and DomCop use root domains
    root_domain = ".".join(domain.split(".")[-2:]) if domain.count(".") > 1 else domain
    report.rank, report.rank_source = _fetch_rank(root_domain)

    return report


def detect_site_sync(url: str) -> dict:
    """Convenience wrapper returning a plain dict (for JSON serialisation)."""
    return detect_site(url).to_dict()
