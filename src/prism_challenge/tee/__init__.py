"""Prism-owned TEE evidence verification (fail-closed, provider-scoped).

Only Prism may interpret TEE evidence. Base carries opaque attestation dictionaries
and ordinary proof signatures. Real Lium/Targon validation remains BLOCKED until
authoritative contracts and trust material exist; fully verified local fixtures are
labeled ``LOCAL-FIXTURE PASS`` only.
"""

from __future__ import annotations

from .adapters import LiumAdapter, TargonAdapter, select_adapter
from .classification import (
    LOCAL_FIXTURE_PASS_LABEL,
    REAL_PROVIDER_PASS_LABEL,
    ClassificationHonestyError,
    assert_honest_classification_surface,
    assert_not_real_provider_pass,
    coerce_accepted_fixture_classification,
    decision_public_surface,
    decision_validation_source,
    human_summary_line,
    smoke_deploy_labels,
)
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
    "LOCAL_FIXTURE_PASS_LABEL",
    "REAL_PROVIDER_PASS_LABEL",
    "TEE_REQUIRED_REASON",
    "WATCHTOWER_UNBOUND",
    "ClassificationHonestyError",
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
    "assert_honest_classification_surface",
    "assert_not_real_provider_pass",
    "classify_safe_probe",
    "coerce_accepted_fixture_classification",
    "decision_authorizes_score",
    "decision_public_surface",
    "decision_validation_source",
    "evaluate_provider_readiness",
    "evaluate_watchtower_digest",
    "human_summary_line",
    "real_provider_pass_is_possible",
    "reject_message",
    "require_for_score_enabled",
    "select_adapter",
    "smoke_deploy_labels",
    "tee_config_from_settings",
    "verify_proof_tee",
]
