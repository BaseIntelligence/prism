"""Prism-owned TEE evidence verification (fail-closed, provider-scoped).

Only Prism may interpret TEE evidence. Base carries opaque attestation dictionaries
and ordinary proof signatures. Real Lium/Targon validation remains BLOCKED until
authoritative contracts and trust material exist; fully verified local fixtures are
labeled ``LOCAL-FIXTURE PASS`` only.
"""

from __future__ import annotations

from .adapters import LiumAdapter, TargonAdapter, select_adapter
from .config import TeeVerifierConfig, tee_config_from_settings
from .nonce_store import DurableNonceStore, InMemoryNonceStore, NonceStore
from .readiness import (
    HARD_GATE_ITEMS,
    WATCHTOWER_UNBOUND,
    ProviderReadinessReport,
    SafeProbeReport,
    WatchtowerEvaluation,
    classify_safe_probe,
    evaluate_provider_readiness,
    evaluate_watchtower_digest,
    real_provider_pass_is_possible,
)
from .score_gate import (
    TEE_REQUIRED_REASON,
    ScoreAuthorization,
    decision_authorizes_score,
    reject_message,
    require_for_score_enabled,
)
from .types import (
    TeeClassification,
    TeeDecision,
    TeeProviderKind,
    TeeReasonCode,
)
from .verifier import TeeVerifier, verify_proof_tee

__all__ = [
    "HARD_GATE_ITEMS",
    "TEE_REQUIRED_REASON",
    "WATCHTOWER_UNBOUND",
    "DurableNonceStore",
    "InMemoryNonceStore",
    "LiumAdapter",
    "NonceStore",
    "ProviderReadinessReport",
    "SafeProbeReport",
    "ScoreAuthorization",
    "TargonAdapter",
    "TeeClassification",
    "TeeDecision",
    "TeeProviderKind",
    "TeeReasonCode",
    "TeeVerifier",
    "TeeVerifierConfig",
    "WatchtowerEvaluation",
    "classify_safe_probe",
    "decision_authorizes_score",
    "evaluate_provider_readiness",
    "evaluate_watchtower_digest",
    "real_provider_pass_is_possible",
    "reject_message",
    "require_for_score_enabled",
    "select_adapter",
    "tee_config_from_settings",
    "verify_proof_tee",
]
