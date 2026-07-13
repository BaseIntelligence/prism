"""Fail-closed production scoring authorization for TEE-required mode.

When ``require_for_score`` is enabled, Prism must not finalize a production score,
leaderboard/architecture-family row, or emission-ready weight contribution unless a
verifier-accepted TEE decision authorizes it. Ordinary ExecutionProof tiers 0/1,
watchtower digest matches, and legacy broker/base_gpu re-exec alone never satisfy
the gate. LOCAL-FIXTURE PASS may authorize lab scoring only while verifier mode is
``local_fixture``; REAL-PROVIDER PASS remains the only production elevation class
and stays BLOCKED until external provider contracts exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import TeeClassification, TeeDecision, TeeReasonCode

#: Stable machine reason used by ingestion/HTTP when TEE-required scoring rejects.
TEE_REQUIRED_REASON = "tee_required"

#: Sub-reasons embedded in detail for diagnostics (never secret-bearing).
SUBREASON_MISSING_DECISION = "missing_tee_decision"
SUBREASON_NOT_ACCEPTED = "tee_not_accepted"
SUBREASON_LOW_TIER = "effective_tier_insufficient"
SUBREASON_WRONG_CLASS = "classification_not_score_authorizing"
SUBREASON_MODE_MISMATCH = "local_fixture_not_allowed_in_production_mode"
SUBREASON_WATCHTOWER = "watchtower_not_attestation"
SUBREASON_CONFIG = "tee_config_incomplete"
SUBREASON_LEGACY_PATH = "legacy_path_without_accepted_tee"


@dataclass(frozen=True)
class ScoreAuthorization:
    """Result of applying the TEE-required score gate."""

    authorized: bool
    reason: str | None = None
    subreason: str | None = None
    detail: str = ""
    decision: TeeDecision | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "reason": self.reason,
            "subreason": self.subreason,
            "detail": self.detail,
            "decision_accepted": (None if self.decision is None else bool(self.decision.accepted)),
            "decision_classification": (
                None if self.decision is None else self.decision.classification.value
            ),
            "decision_effective_tier": (
                None if self.decision is None else int(self.decision.effective_tier)
            ),
        }


def require_for_score_enabled(
    *,
    require_for_score: bool | None = None,
    tee_config: Any | None = None,
    settings: Any | None = None,
) -> bool:
    """Resolve whether TEE-required scoring is active from explicit or nested config."""

    if require_for_score is not None:
        return bool(require_for_score)
    if tee_config is not None and hasattr(tee_config, "require_for_score"):
        return bool(tee_config.require_for_score)
    if settings is not None:
        tee = getattr(settings, "tee", None)
        if tee is not None and hasattr(tee, "require_for_score"):
            return bool(tee.require_for_score)
        top = getattr(settings, "tee_require_for_score", None)
        if top is not None:
            return bool(top)
    return False


def decision_authorizes_score(
    decision: TeeDecision | None,
    *,
    require_for_score: bool,
    mode: str = "local_fixture",
) -> ScoreAuthorization:
    """Authorize production score finalization under the TEE-required policy.

    When ``require_for_score`` is false the gate is a no-op (authorized=True) so existing
    legacy/test paths keep working unless the product flag is on.
    """

    if not require_for_score:
        return ScoreAuthorization(authorized=True, decision=decision)

    if decision is None:
        return ScoreAuthorization(
            authorized=False,
            reason=TEE_REQUIRED_REASON,
            subreason=SUBREASON_MISSING_DECISION,
            detail="TEE-required scoring has no verifier decision",
            decision=None,
        )

    # Watchtower / metadata-only never authorizes, even if mis-labeled accepted.
    if decision.reason is TeeReasonCode.METADATA_NOT_ATTESTATION:
        return ScoreAuthorization(
            authorized=False,
            reason=TEE_REQUIRED_REASON,
            subreason=SUBREASON_WATCHTOWER,
            detail="watchtower digest match is never TEE evidence",
            decision=decision,
        )
    if (
        decision.classification
        in (
            TeeClassification.BLOCKED,
            TeeClassification.FAIL,
        )
        or not decision.accepted
    ):
        # Incomplete config often lands as VERIFIER_MISCONFIGURED / BLOCKED.
        sub = SUBREASON_NOT_ACCEPTED
        if decision.reason in (
            TeeReasonCode.VERIFIER_MISCONFIGURED,
            TeeReasonCode.VERIFIER_DISABLED,
            TeeReasonCode.MEASUREMENT_POLICY_EMPTY,
            TeeReasonCode.ADAPTER_NOT_READY,
        ):
            sub = SUBREASON_CONFIG
        if decision.reason is TeeReasonCode.METADATA_NOT_ATTESTATION:
            sub = SUBREASON_WATCHTOWER
        return ScoreAuthorization(
            authorized=False,
            reason=TEE_REQUIRED_REASON,
            subreason=sub,
            detail=(
                f"TEE decision not accepted "
                f"(classification={decision.classification.value}, "
                f"reason={decision.reason.value})"
            ),
            decision=decision,
        )

    if int(decision.effective_tier) < 2:
        return ScoreAuthorization(
            authorized=False,
            reason=TEE_REQUIRED_REASON,
            subreason=SUBREASON_LOW_TIER,
            detail=(
                f"effective tier {decision.effective_tier} insufficient for "
                "TEE-required score finalization (tier 0/1 never authorize)"
            ),
            decision=decision,
        )

    if decision.classification is TeeClassification.LOCAL_FIXTURE_PASS:
        if str(mode) != "local_fixture":
            return ScoreAuthorization(
                authorized=False,
                reason=TEE_REQUIRED_REASON,
                subreason=SUBREASON_MODE_MISMATCH,
                detail=("LOCAL-FIXTURE PASS cannot authorize production-mode TEE-required scoring"),
                decision=decision,
            )
        return ScoreAuthorization(authorized=True, decision=decision)

    if decision.classification is TeeClassification.REAL_PROVIDER_PASS:
        return ScoreAuthorization(authorized=True, decision=decision)

    return ScoreAuthorization(
        authorized=False,
        reason=TEE_REQUIRED_REASON,
        subreason=SUBREASON_WRONG_CLASS,
        detail=(f"classification {decision.classification.value} is not score-authorizing"),
        decision=decision,
    )


def reject_message(auth: ScoreAuthorization) -> str:
    """Stable human detail string for ingestion/HTTP error surfaces."""

    parts = [auth.detail or "TEE-required scoring rejected"]
    if auth.subreason:
        parts.append(f"subreason={auth.subreason}")
    return "; ".join(parts)


# Convenience set (kept for tests/docs; gate uses enum comparisons above).
SCORE_AUTHORIZING_CLASSIFICATIONS = frozenset(
    {
        TeeClassification.LOCAL_FIXTURE_PASS,
        TeeClassification.REAL_PROVIDER_PASS,
    }
)

__all__ = [
    "SCORE_AUTHORIZING_CLASSIFICATIONS",
    "SUBREASON_CONFIG",
    "SUBREASON_LEGACY_PATH",
    "SUBREASON_LOW_TIER",
    "SUBREASON_MISSING_DECISION",
    "SUBREASON_MODE_MISMATCH",
    "SUBREASON_NOT_ACCEPTED",
    "SUBREASON_WATCHTOWER",
    "SUBREASON_WRONG_CLASS",
    "ScoreAuthorization",
    "TEE_REQUIRED_REASON",
    "decision_authorizes_score",
    "reject_message",
    "require_for_score_enabled",
]
