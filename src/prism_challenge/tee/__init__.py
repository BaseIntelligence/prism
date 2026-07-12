"""Prism-owned TEE evidence verification (fail-closed, provider-scoped).

Only Prism may interpret TEE evidence. Base carries opaque attestation dictionaries
and ordinary proof signatures. Real Lium/Targon validation remains BLOCKED until
authoritative contracts and trust material exist; fully verified local fixtures are
labeled ``LOCAL-FIXTURE PASS`` only.
"""

from __future__ import annotations

from .config import TeeVerifierConfig, tee_config_from_settings
from .nonce_store import DurableNonceStore, InMemoryNonceStore, NonceStore
from .types import (
    TeeClassification,
    TeeDecision,
    TeeProviderKind,
    TeeReasonCode,
)
from .verifier import TeeVerifier, verify_proof_tee

__all__ = [
    "DurableNonceStore",
    "InMemoryNonceStore",
    "NonceStore",
    "TeeClassification",
    "TeeDecision",
    "TeeProviderKind",
    "TeeReasonCode",
    "TeeVerifier",
    "TeeVerifierConfig",
    "tee_config_from_settings",
    "verify_proof_tee",
]
