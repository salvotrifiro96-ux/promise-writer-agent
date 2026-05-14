"""Archivio Supabase dei brief del promise-writer.

Tabella `promise_briefs`:
    id            uuid primary key default gen_random_uuid()
    created_at    timestamptz default now()
    updated_at    timestamptz default now()
    title         text          -- auto-derivato dai primi 60 char del context o dal target
    brief         jsonb         -- {context, references, target_audience, brand_voice, n_headlines, extra_instructions}
    promises      jsonb         -- lista Promise serializzate

Se le env Supabase non sono settate, `BriefStore.from_env()` ritorna None e
l'app continua a funzionare senza archivio.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

TABLE = "promise_briefs"


@dataclass(frozen=True)
class BriefRow:
    id: str
    title: str
    brief: dict[str, Any]
    promises: list[dict[str, Any]]
    created_at: str
    updated_at: str


class BriefStore:
    def __init__(self, url: str, secret_key: str) -> None:
        if not url or not secret_key:
            raise ValueError("SUPABASE_URL e SUPABASE_SECRET_KEY obbligatori")
        self.url = url.rstrip("/")
        self.secret_key = secret_key
        self._rest = f"{self.url}/rest/v1"
        self._h_read = {
            "apikey": secret_key,
            "Authorization": f"Bearer {secret_key}",
        }
        self._h_write = {
            **self._h_read,
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    @classmethod
    def from_env(cls) -> "BriefStore | None":
        try:
            import streamlit as st
            url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
            key = (
                os.getenv("SUPABASE_SECRET_KEY")
                or os.getenv("SUPABASE_SERVICE_KEY")
                or st.secrets.get("SUPABASE_SECRET_KEY", "")
                or st.secrets.get("SUPABASE_SERVICE_KEY", "")
            )
        except Exception:
            url = os.getenv("SUPABASE_URL", "")
            key = (
                os.getenv("SUPABASE_SECRET_KEY", "")
                or os.getenv("SUPABASE_SERVICE_KEY", "")
            )
        if not url or not key:
            return None
        return cls(url=url, secret_key=key)

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _derive_title(brief: dict[str, Any]) -> str:
        target = (brief.get("target_audience") or "").strip()
        if target:
            return target[:80]
        ctx = (brief.get("context") or "").strip()
        first_line = ctx.split("\n", 1)[0].strip()
        return (first_line or "Brief senza titolo")[:80]

    @staticmethod
    def _row_to_brief(row: dict[str, Any]) -> BriefRow:
        return BriefRow(
            id=str(row["id"]),
            title=row.get("title", "") or "(senza titolo)",
            brief=row.get("brief", {}) or {},
            promises=row.get("promises", []) or [],
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )

    # ── CRUD ──────────────────────────────────────────────────────────
    def insert(self, brief: dict[str, Any], promises: list[dict[str, Any]]) -> BriefRow:
        row = {
            "title": self._derive_title(brief),
            "brief": brief,
            "promises": promises,
        }
        r = requests.post(
            f"{self._rest}/{TABLE}",
            data=json.dumps(row),
            headers=self._h_write,
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Insert brief fallito {r.status_code}: {r.text[:300]}")
        data = r.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Risposta inattesa: {data!r}")
        return self._row_to_brief(data[0])

    def update(self, brief_id: str, brief: dict[str, Any], promises: list[dict[str, Any]]) -> BriefRow:
        body = {
            "title": self._derive_title(brief),
            "brief": brief,
            "promises": promises,
            "updated_at": "now()",
        }
        r = requests.patch(
            f"{self._rest}/{TABLE}",
            params={"id": f"eq.{brief_id}"},
            data=json.dumps(body),
            headers=self._h_write,
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Update brief fallito {r.status_code}: {r.text[:300]}")
        data = r.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Update senza risposta: {data!r}")
        return self._row_to_brief(data[0])

    def delete(self, brief_id: str) -> None:
        r = requests.delete(
            f"{self._rest}/{TABLE}",
            params={"id": f"eq.{brief_id}"},
            headers=self._h_read,
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Delete brief fallito {r.status_code}: {r.text[:300]}")

    def list_recent(self, limit: int = 50) -> list[BriefRow]:
        r = requests.get(
            f"{self._rest}/{TABLE}",
            params={
                "select": "*",
                "order": "updated_at.desc",
                "limit": str(limit),
            },
            headers=self._h_read,
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"List briefs fallito {r.status_code}: {r.text[:300]}")
        rows = r.json() or []
        return [self._row_to_brief(row) for row in rows]

    def get(self, brief_id: str) -> BriefRow | None:
        r = requests.get(
            f"{self._rest}/{TABLE}",
            params={"select": "*", "id": f"eq.{brief_id}", "limit": "1"},
            headers=self._h_read,
            timeout=30,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Get brief fallito {r.status_code}: {r.text[:300]}")
        rows = r.json() or []
        if not rows:
            return None
        return self._row_to_brief(rows[0])
