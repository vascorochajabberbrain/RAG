"""
jBKE (jabberBrain Knowledge Editor) HTTP client.

Uses AIM-style token auth: POST with user_name + version_id + auth_token
to authenticate without a browser session.

Env vars (from .env):
  JBKE_BASE_URL   — e.g. https://www.jabberbrain.com/jb
  JBKE_AUTH_TOKEN  — shared secret matching jBKE server's AUTH_TOKEN
  JBKE_USER_NAME   — audit-trail username (e.g. "RAG_BUILDER")
"""

import os
import logging
import requests

log = logging.getLogger(__name__)


class JBKEClient:
    """Thin HTTP wrapper around jBKE's rag_collection_process.php."""

    def __init__(
        self,
        base_url: str | None = None,
        auth_token: str | None = None,
        user_name: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("JBKE_BASE_URL", "")).rstrip("/")
        self.auth_token = auth_token or os.getenv("JBKE_AUTH_TOKEN", "")
        self.user_name = user_name or os.getenv("JBKE_USER_NAME", "RAG_BUILDER")

        if not self.base_url or not self.auth_token:
            raise ValueError(
                "jBKE connection not configured. "
                "Set JBKE_BASE_URL and JBKE_AUTH_TOKEN in your .env file."
            )

        # Cache language code → id mapping (loaded on first use)
        self._language_map: dict[str, int] | None = None

    # ── low-level helpers ──────────────────────────────────────────

    @staticmethod
    def _parse_response(resp) -> dict:
        """Parse jBKE response, handling PHP errors that return HTML instead of JSON."""
        try:
            data = resp.json()
        except Exception:
            body = resp.text.strip()
            # PHP fatal errors come back as HTML
            if "Fatal error" in body or "<b>" in body:
                import re
                msg = re.sub(r"<[^>]+>", "", body).strip()
                raise RuntimeError(f"jBKE PHP error: {msg[:300]}")
            raise RuntimeError(f"jBKE returned non-JSON response: {body[:300]}")
        # PHP empty array() serializes as [] — normalize to dict
        if isinstance(data, list):
            log.warning("jBKE returned array instead of object: %s", resp.url)
            return {"success": False, "data": data}
        return data

    def _auth_params(self, version_id: int) -> dict:
        return {
            "user_name": self.user_name,
            "version_id": str(version_id),
            "auth_token": self.auth_token,
        }

    def _endpoint(self, php_file: str) -> str:
        return f"{self.base_url}/{php_file}"

    # ── Language lookup ──────────────────────────────────────────

    def _load_languages(self, version_id: int) -> dict[str, int]:
        """Fetch language code → id mapping from jBKE's languages table."""
        url = self._endpoint("bo_language_process.php")
        params = {
            **self._auth_params(version_id),
            "UserAction": "GetLanguages",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        result = self._parse_response(resp)
        mapping = {}
        for lang in result.get("data", []):
            # Map language name to id; also build code→id from known names
            name = lang.get("name", "").strip()
            lid = int(lang.get("value", 0))
            if name and lid:
                mapping[name.upper()] = lid
        # Build a code→id lookup from common language names
        name_to_code = {
            "PORTUGUESE": "PT", "ENGLISH": "EN", "SPANISH": "ES",
            "FRENCH": "FR", "GERMAN": "DE", "ITALIAN": "IT",
            "DUTCH": "NL", "SWEDISH": "SV", "DANISH": "DA",
            "FINNISH": "FI", "NORWEGIAN": "NO", "POLISH": "PL",
            "ROMANIAN": "RO", "RUSSIAN": "RU", "HUNGARIAN": "HU",
            "JAPANESE": "JA", "SLOVAK": "SK", "SLOVENE": "SL",
            "GENERAL": "XX", "ALL": "ALL",
        }
        code_map = {}
        for name_upper, lid in mapping.items():
            code = name_to_code.get(name_upper)
            if code:
                code_map[code] = lid
        return code_map

    def get_language_id(self, lang_code: str, version_id: int) -> int | None:
        """Resolve a 2-letter language code (e.g. 'pt') to a jBKE language_id."""
        if self._language_map is None:
            self._language_map = self._load_languages(version_id)
        return self._language_map.get(lang_code.upper())

    # ── CRUD operations ────────────────────────────────────────────

    def list_collections(self, version_id: int) -> list[dict]:
        """Fetch all RAG collection records for this version."""
        url = self._endpoint("rag_collection_process.php")
        params = {
            **self._auth_params(version_id),
            "UserAction": "GetAll",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        result = self._parse_response(resp)
        if isinstance(result, dict) and result.get("success"):
            data = result.get("data", [])
            return data if isinstance(data, list) else []
        log.warning("list_collections: unexpected response: %s", type(result))
        return []

    def find_collection_by_name(
        self, version_id: int, collection_name: str
    ) -> dict | None:
        """Find a RAG collection by its rcd_name. Returns the record dict or None."""
        all_colls = self.list_collections(version_id)
        for coll in all_colls:
            if isinstance(coll, dict) and coll.get("rcd_name") == collection_name:
                return coll
        return None

    def _find_by_name_scan(
        self, version_id: int, collection_name: str
    ) -> dict | None:
        """Find a collection by name: try GetAll first, then scan IDs 1-100."""
        # Try GetAll (fast path)
        found = self.find_collection_by_name(version_id, collection_name)
        if found:
            return found
        # Fallback: scan individual IDs via GetData
        log.info("GetAll didn't find '%s', scanning IDs…", collection_name)
        for rcm_id in range(1, 101):
            rec = self.get_collection(version_id, rcm_id)
            if rec and isinstance(rec, dict) and rec.get("rcd_name") == collection_name:
                return rec
        return None

    def get_collection(self, version_id: int, rcm_id: int) -> dict | None:
        """Fetch a RAG collection record by rcm_id (main table id)."""
        url = self._endpoint("rag_collection_process.php")
        params = {
            **self._auth_params(version_id),
            "UserAction": "GetData",
            "rcm_id": str(rcm_id),
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        result = self._parse_response(resp)
        if result.get("success") and result.get("data"):
            return result["data"]
        return None

    def create_collection(self, version_id: int, fields: dict) -> dict:
        """
        Create a new RAG collection in jBKE.

        Returns the full response dict.  On success, the new record's
        rcm_id can be found via a subsequent GetData call (the Create
        response from jBKE's GeneralProcess returns the new detail id).
        """
        url = self._endpoint("rag_collection_process.php")
        payload = {
            **self._auth_params(version_id),
            "UserAction": "Create",
            **fields,
        }
        resp = requests.post(url, data=payload, timeout=30)
        resp.raise_for_status()
        return self._parse_response(resp)

    def update_collection(
        self, version_id: int, rcm_id: int, fields: dict
    ) -> dict:
        """Update routing / config fields on an existing RAG collection."""
        url = self._endpoint("rag_collection_process.php")
        payload = {
            **self._auth_params(version_id),
            "UserAction": "Update",
            "rcm_id": str(rcm_id),
            **fields,
        }
        resp = requests.post(url, data=payload, timeout=30)
        resp.raise_for_status()
        return self._parse_response(resp)

    # ── high-level: push routing metadata ──────────────────────────

    @staticmethod
    def _doc_type_to_rcd_type(doc_type: str) -> str:
        """Map solutions.yaml doc_type to jBKE rcd_type values."""
        mapping = {
            "product_catalog": "Product",
            "recipe_book": "Recipe",
            "faq": "FAQ",
            "manual": "Manual",
            "legal": "Legal",
            "general": "General",
        }
        return mapping.get(doc_type, "Other")

    def build_routing_payload(
        self,
        collection_name: str,
        collection_type: str,
        routing: dict,
        settings: dict,
        cbva_id: int | None = None,
        version_id: int | None = None,
    ) -> dict:
        """
        Transform solutions.yaml routing + settings into jBKE POST fields.

        Array fields (keywords, typical_questions, not_covered) are joined
        with comma or pipe separators to match jBKE's expected format.
        """
        def _join_comma(val):
            """Join list items with ', ' — for keywords and not_covered."""
            if isinstance(val, list):
                return ", ".join(str(v) for v in val)
            return str(val) if val else ""

        def _join_pipe(val):
            """Join list items with ' | ' — for typical_questions."""
            if isinstance(val, list):
                return " | ".join(str(v) for v in val)
            return str(val) if val else ""

        fields = {
            "rcd_name": collection_name,
            # rcd_type = content type (Product, FAQ, General, etc.)
            "rcd_type": self._doc_type_to_rcd_type(routing.get("doc_type", "")),
            "rcd_status": "enabled",
            # Routing metadata
            "rcd_routing_description": routing.get("description", ""),
            "rcd_routing_keywords": _join_comma(routing.get("keywords")),
            "rcd_routing_typical_questions": _join_pipe(
                routing.get("typical_questions")
            ),
            "rcd_routing_not_covered": _join_comma(routing.get("not_covered")),
            "rcd_routing_doc_type": routing.get("doc_type", ""),
            # Embedding config (crucial — always populate)
            "rcd_llm_model_embedding": settings.get("embedding_model", ""),
            "rcd_embedding_dimensions": "1536",
            # Sequence: controls processing order in jBKE
            "rcd_sequence": str(routing.get("sequence", 0)),
            # Additional prompt: appended to LLM prompt for RAG responses
            "rcd_prompt_rag": routing.get("additional_prompt", ""),
            "rcd_append_prompt": "1" if routing.get("additional_prompt") else "0",
            # Leave these empty so jBKE uses its defaults:
            # rcd_vector_store_url, rcd_llm_model_answer,
            # rcd_llm_model_query_rewrite, rcd_qdrant_collection_name
        }

        # Language: look up the numeric language_id from jBKE's languages table
        lang = routing.get("language", "")
        if lang and version_id:
            lang_id = self.get_language_id(lang, version_id)
            if lang_id:
                fields["rcd_routing_language_id"] = str(lang_id)

        if cbva_id:
            fields["rcd_cbva_id"] = str(cbva_id)

        return fields

    def push_routing(
        self,
        version_id: int,
        rcm_id: int | None,
        collection_name: str,
        collection_type: str,
        routing: dict,
        settings: dict,
        cbva_id: int | None = None,
    ) -> dict:
        """
        Create or update a RAG collection in jBKE with routing metadata.

        Returns {"success": bool, "action": "created"|"updated",
                 "rcm_id": int|None, "message": str}
        """
        fields = self.build_routing_payload(
            collection_name, collection_type, routing, settings, cbva_id,
            version_id=version_id,
        )

        if not rcm_id:
            # No rcm_id stored — check if collection already exists in jBKE
            existing = self.find_collection_by_name(version_id, collection_name)
            if existing:
                rcm_id = int(existing.get("rcm_id", 0)) or None
                if rcm_id:
                    log.info(
                        "Auto-detected existing jBKE collection '%s' → rcm_id=%s",
                        collection_name, rcm_id,
                    )

        if rcm_id:
            # UPDATE existing
            result = self.update_collection(version_id, rcm_id, fields)
            success = result.get("success", False)
            return {
                "success": success,
                "action": "updated",
                "rcm_id": rcm_id,
                "message": (
                    f"Updated RAG collection '{collection_name}' in jBKE (rcm_id={rcm_id})"
                    if success
                    else f"Failed to update: {result.get('errors', result)}"
                ),
            }
        else:
            # CREATE new
            result = self.create_collection(version_id, fields)
            success = result.get("success", False)

            # Handle "already exists" — find the existing record and update it
            if not success:
                errors = result.get("errors", {})
                err_vals = " ".join(str(v) for v in errors.values()) if isinstance(errors, dict) else str(errors)
                if "already exists" in err_vals.lower():
                    log.info(
                        "Collection '%s' already exists in jBKE, looking up rcm_id…",
                        collection_name,
                    )
                    found = self._find_by_name_scan(version_id, collection_name)
                    if found:
                        found_id = int(found.get("rcm_id", 0)) or None
                        if found_id:
                            log.info("Found existing rcm_id=%s, updating…", found_id)
                            upd = self.update_collection(version_id, found_id, fields)
                            upd_ok = upd.get("success", False)
                            return {
                                "success": upd_ok,
                                "action": "updated",
                                "rcm_id": found_id,
                                "message": (
                                    f"Found existing collection in jBKE (rcm_id={found_id}) and updated it"
                                    if upd_ok
                                    else f"Found rcm_id={found_id} but update failed: {upd.get('errors', upd)}"
                                ),
                            }

            # Try to extract the new rcm_id from the response
            new_rcm_id = None
            if success:
                rd = result.get("ReceivedData", {})
                new_rcm_id = rd.get("rcm_id") or rd.get("rcm_id")
                # Also try to get from the data key
                if not new_rcm_id and "data" in result:
                    new_rcm_id = result["data"].get("rcm_id")
            return {
                "success": success,
                "action": "created",
                "rcm_id": int(new_rcm_id) if new_rcm_id else None,
                "message": (
                    f"Created RAG collection '{collection_name}' in jBKE"
                    + (f" (rcm_id={new_rcm_id})" if new_rcm_id else "")
                    if success
                    else f"Failed to create: {result.get('errors', result)}"
                ),
            }
