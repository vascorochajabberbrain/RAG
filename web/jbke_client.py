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

    # ── low-level helpers ──────────────────────────────────────────

    def _auth_params(self, version_id: int) -> dict:
        return {
            "user_name": self.user_name,
            "version_id": str(version_id),
            "auth_token": self.auth_token,
        }

    def _endpoint(self, php_file: str) -> str:
        return f"{self.base_url}/{php_file}"

    # ── CRUD operations ────────────────────────────────────────────

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
        result = resp.json()
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
        return resp.json()

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
        return resp.json()

    # ── high-level: push routing metadata ──────────────────────────

    def build_routing_payload(
        self,
        collection_name: str,
        collection_type: str,
        routing: dict,
        settings: dict,
        cbva_id: int | None = None,
    ) -> dict:
        """
        Transform solutions.yaml routing + settings into jBKE POST fields.

        Array fields (keywords, typical_questions, not_covered) are joined
        with newlines — jBKE stores them as text blobs.
        """
        def _join(val):
            if isinstance(val, list):
                return "\n".join(str(v) for v in val)
            return str(val) if val else ""

        fields = {
            "rcd_name": collection_name,
            "rcd_type": collection_type or "scs",
            "rcd_status": "enabled",
            # Routing metadata
            "rcd_routing_description": routing.get("description", ""),
            "rcd_routing_keywords": _join(routing.get("keywords")),
            "rcd_routing_typical_questions": _join(
                routing.get("typical_questions")
            ),
            "rcd_routing_not_covered": _join(routing.get("not_covered")),
            "rcd_routing_doc_type": routing.get("doc_type", ""),
            # LLM / vector config from settings
            "rcd_vector_store_url": settings.get("qdrant_url", ""),
            "rcd_llm_model_answer": settings.get("llm_chat_model", ""),
            "rcd_llm_model_query_rewrite": settings.get(
                "llm_processing_model", ""
            ),
            "rcd_llm_model_embedding": settings.get("embedding_model", ""),
            "rcd_embedding_dimensions": "1536",
        }
        # Language: pass as-is (jBKE stores language_id as a number,
        # but the field accepts the value directly)
        lang = routing.get("language", "")
        if lang:
            fields["rcd_routing_language_id"] = lang

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
            collection_name, collection_type, routing, settings, cbva_id
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
