"""Durable single-use TEE nonce consumption (atomic, restart-safe)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import aiosqlite

from prism_challenge.db import open_sqlite

TEE_NONCE_LEDGER_DDL = (
    "CREATE TABLE IF NOT EXISTS tee_nonce_ledger ("
    "nonce_digest TEXT PRIMARY KEY,"
    "provider TEXT NOT NULL,"
    "work_unit_id TEXT NOT NULL,"
    "evidence_digest TEXT NOT NULL,"
    "consumed_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_tee_nonce_ledger_unit "
    "ON tee_nonce_ledger(work_unit_id, consumed_at);"
)

TEE_DECISION_DDL = (
    "CREATE TABLE IF NOT EXISTS tee_verification_decisions ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "work_unit_id TEXT NOT NULL,"
    "evidence_digest TEXT NOT NULL,"
    "provider TEXT NOT NULL,"
    "classification TEXT NOT NULL,"
    "reason TEXT NOT NULL,"
    "effective_tier INTEGER NOT NULL,"
    "claimed_tier INTEGER NOT NULL,"
    "trust_root_fingerprint TEXT,"
    "gpu_key_fingerprint TEXT,"
    "image_digest TEXT,"
    "nonce_digest TEXT,"
    "validated_claims TEXT NOT NULL,"
    "created_at TEXT NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_tee_decisions_unit "
    "ON tee_verification_decisions(work_unit_id, created_at);"
)


def nonce_digest(nonce: str) -> str:
    return hashlib.sha256(nonce.encode("utf-8")).hexdigest()


class NonceStore(Protocol):
    async def is_consumed(self, nonce: str) -> bool: ...

    async def try_consume(
        self,
        *,
        nonce: str,
        provider: str,
        work_unit_id: str,
        evidence_digest: str,
    ) -> bool:
        """Atomically consume ``nonce``. Returns False if already consumed."""
        ...


class InMemoryNonceStore:
    """Process-local nonce store for pure unit tests (not restart-safe)."""

    def __init__(self) -> None:
        import asyncio

        self._seen: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    async def is_consumed(self, nonce: str) -> bool:
        async with self._lock:
            return nonce_digest(nonce) in self._seen

    async def try_consume(
        self,
        *,
        nonce: str,
        provider: str,
        work_unit_id: str,
        evidence_digest: str,
    ) -> bool:
        digest = nonce_digest(nonce)
        async with self._lock:
            if digest in self._seen:
                return False
            self._seen[digest] = {
                "provider": provider,
                "work_unit_id": work_unit_id,
                "evidence_digest": evidence_digest,
                "consumed_at": datetime.now(UTC).isoformat(),
            }
            return True


class DurableNonceStore:
    """SQLite-backed TEE nonce ledger shared across Prism restarts."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path

    async def ensure_schema(self) -> None:
        conn = await open_sqlite(self._path)
        try:
            await conn.executescript(TEE_NONCE_LEDGER_DDL)
            await conn.executescript(TEE_DECISION_DDL)
            await conn.commit()
        finally:
            await conn.close()

    async def is_consumed(self, nonce: str) -> bool:
        digest = nonce_digest(nonce)
        conn = await open_sqlite(self._path)
        try:
            row = await conn.execute_fetchall(
                "SELECT 1 FROM tee_nonce_ledger WHERE nonce_digest=?",
                (digest,),
            )
            return bool(list(row))
        finally:
            await conn.close()

    async def try_consume(
        self,
        *,
        nonce: str,
        provider: str,
        work_unit_id: str,
        evidence_digest: str,
    ) -> bool:
        digest = nonce_digest(nonce)
        conn = await open_sqlite(self._path)
        try:
            try:
                await conn.execute(
                    "INSERT INTO tee_nonce_ledger("
                    "nonce_digest, provider, work_unit_id, evidence_digest, consumed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        digest,
                        provider,
                        work_unit_id,
                        evidence_digest,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                await conn.commit()
                return True
            except aiosqlite.IntegrityError:
                await conn.rollback()
                return False
        finally:
            await conn.close()

    async def record_decision(
        self,
        *,
        work_unit_id: str,
        evidence_digest: str,
        provider: str,
        classification: str,
        reason: str,
        effective_tier: int,
        claimed_tier: int,
        trust_root_fingerprint: str | None,
        gpu_key_fingerprint: str | None,
        image_digest: str | None,
        nonce_digest_value: str | None,
        validated_claims: str,
    ) -> None:
        conn = await open_sqlite(self._path)
        try:
            await conn.execute(
                "INSERT INTO tee_verification_decisions("
                "work_unit_id, evidence_digest, provider, classification, reason,"
                "effective_tier, claimed_tier, trust_root_fingerprint, gpu_key_fingerprint,"
                "image_digest, nonce_digest, validated_claims, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    work_unit_id,
                    evidence_digest,
                    provider,
                    classification,
                    reason,
                    int(effective_tier),
                    int(claimed_tier),
                    trust_root_fingerprint,
                    gpu_key_fingerprint,
                    image_digest,
                    nonce_digest_value,
                    validated_claims,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await conn.commit()
        finally:
            await conn.close()
