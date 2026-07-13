"""Honest TEE classification surfaces for APIs, CLI, audit, and lab strings.

Contract outcomes are exactly ``LOCAL-FIXTURE PASS``, ``REAL-PROVIDER PASS``,
``BLOCKED``, or ``FAIL``. Local cryptographic fixtures may only ever be labeled
``LOCAL-FIXTURE PASS`` (VAL-TEEREQ-003, VAL-TEEREQ-010). Lium/Targon adapters never
emit ``REAL-PROVIDER PASS`` while readiness ``would_grant_real_provider_pass`` is
false (VAL-TEEREQ-004, VAL-TEEREQ-005). Guessing or rewriting a local fixture decision
into a real-provider badge is forbidden with a hard error rather than a silent
upgrade.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import TeeClassification, TeeDecision, TeeProviderKind, TeeReasonCode

#: Canonical real-provider PASS string (must never appear for local fixtures).
REAL_PROVIDER_PASS_LABEL = TeeClassification.REAL_PROVIDER_PASS.value
#: Canonical local-fixture PASS string.
LOCAL_FIXTURE_PASS_LABEL = TeeClassification.LOCAL_FIXTURE_PASS.value

#: Badges / synonyms that would smuggle fixture success into production mine authority.
_FORBIDDEN_PRODUCTION_MINE_BADGES = frozenset(
    {
        REAL_PROVIDER_PASS_LABEL,
        "REAL PROVIDER PASS",
        "REAL_PROVIDER_PASS",
        "REALPROVIDERPASS",
        "PRODUCTION MINE",
        "PRODUCTION-MINE",
        "LIVE-EMISSION",
        "LIVE EMISSION AUTHORITY",
        "LIVE_EMISSION_AUTHORITY",
    }
)

#: Source labels always required when the classification is LOCAL-FIXTURE PASS.
_LOCAL_SOURCE_MARKERS = frozenset(
    {
        "local_fixture",
        "LOCAL-FIXTURE",
        "LOCAL_FIXTURE",
        "local-fixture",
        "validation_source=local_fixture",
    }
)


class ClassificationHonestyError(ValueError):
    """Raised when a caller attempts to promote or smuggle TEE classifications."""


def decision_validation_source(decision: TeeDecision) -> str:
    """Stable source label for public surfaces (never secret-bearing)."""

    meta = decision.metadata or {}
    raw = meta.get("validation_source")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if decision.provider is TeeProviderKind.LOCAL_FIXTURE:
        return "local_fixture"
    if decision.provider is TeeProviderKind.LIUM:
        return "lium"
    if decision.provider is TeeProviderKind.TARGON:
        return "targon"
    return "unknown"


def human_summary_line(decision: TeeDecision) -> str:
    """Single-line human summary that forbids Real-Provider language for fixtures.

    Format is stable for CLI/lab dashboards:
    ``TEE <classification> · source=<source> · tier=<n> · accepted=<bool> · reason=<code>``
    """

    source = decision_validation_source(decision)
    classification = decision.classification.value
    # Defensive: never mint a REAL-PROVIDER line from a local fixture decision object.
    if (
        decision.provider is TeeProviderKind.LOCAL_FIXTURE
        or source == "local_fixture"
        or decision.reason is TeeReasonCode.ACCEPTED_LOCAL_FIXTURE
    ):
        if classification == REAL_PROVIDER_PASS_LABEL:
            classification = LOCAL_FIXTURE_PASS_LABEL
        if decision.accepted and classification not in {
            LOCAL_FIXTURE_PASS_LABEL,
            TeeClassification.BLOCKED.value,
            TeeClassification.FAIL.value,
        }:
            classification = LOCAL_FIXTURE_PASS_LABEL
    return (
        f"TEE {classification} · source={source} · "
        f"tier={int(decision.effective_tier)} · "
        f"accepted={bool(decision.accepted)} · "
        f"reason={decision.reason.value}"
    )


def decision_public_surface(decision: TeeDecision) -> dict[str, Any]:
    """Serialize a decision for APIs/CLI/lab with honesty guards baked in.

    Guarantees:
    * Local-fixture accepted decisions report ``LOCAL-FIXTURE PASS`` only.
    * ``real_provider_pass`` is true only when classification is REAL-PROVIDER PASS
      AND provider is not local_fixture.
    * ``summary`` always includes an explicit source label for local fixtures.
    * No production-mine or live-emission authority flags for fixtures.
    """

    source = decision_validation_source(decision)
    classification = decision.classification
    # Coerce any mislabeled local acceptance back to LOCAL-FIXTURE PASS.
    if (
        decision.provider is TeeProviderKind.LOCAL_FIXTURE
        or source == "local_fixture"
        or decision.reason is TeeReasonCode.ACCEPTED_LOCAL_FIXTURE
    ) and classification is TeeClassification.REAL_PROVIDER_PASS:
        classification = TeeClassification.LOCAL_FIXTURE_PASS

    is_local = (
        classification is TeeClassification.LOCAL_FIXTURE_PASS
        or source == "local_fixture"
        or decision.provider is TeeProviderKind.LOCAL_FIXTURE
    )
    is_real_provider = (
        classification is TeeClassification.REAL_PROVIDER_PASS
        and not is_local
        and decision.provider
        in {
            TeeProviderKind.LIUM,
            TeeProviderKind.TARGON,
        }
        and bool(decision.accepted)
        and int(decision.effective_tier) >= 2
    )

    surface = {
        "accepted": bool(decision.accepted) and classification is not TeeClassification.FAIL,
        "classification": classification.value,
        "reason": decision.reason.value,
        "provider": decision.provider.value,
        "effective_tier": int(decision.effective_tier),
        "validation_source": source,
        "real_provider_pass": is_real_provider,
        "local_fixture_pass": classification is TeeClassification.LOCAL_FIXTURE_PASS
        and bool(decision.accepted),
        "production_mine_badge": False if is_local else is_real_provider,
        "live_emission_authority": False if is_local else is_real_provider,
        "would_grant_real_provider_pass": bool(
            (decision.metadata or {}).get("would_grant_real_provider_pass", False)
        )
        and not is_local,
        "summary": human_summary_line(
            TeeDecision(
                accepted=decision.accepted,
                classification=classification,
                reason=decision.reason,
                provider=decision.provider,
                effective_tier=decision.effective_tier,
                detail=decision.detail,
                metadata=dict(decision.metadata or {}),
                evidence_digest=decision.evidence_digest,
                trust_root_fingerprint=decision.trust_root_fingerprint,
                gpu_key_fingerprint=decision.gpu_key_fingerprint,
                image_digest=decision.image_digest,
                nonce_digest=decision.nonce_digest,
                work_unit_id=decision.work_unit_id,
                validated_claims=decision.validated_claims,
            )
        ),
        "detail": decision.detail,
        "evidence_digest": decision.evidence_digest,
        "work_unit_id": decision.work_unit_id,
    }
    assert_honest_classification_surface(surface)
    return surface


def assert_honest_classification_surface(surface: Mapping[str, Any]) -> None:
    """Fail closed if a public surface smuggles REAL-PROVIDER over a local fixture.

    Used by serialization helpers and targeted tests (VAL-TEEREQ-003, VAL-TEEREQ-010).
    """

    classification = str(surface.get("classification", ""))
    source = str(surface.get("validation_source", "")).lower()
    provider = str(surface.get("provider", "")).lower()
    summary = str(surface.get("summary", ""))
    combined = " ".join(
        [
            classification,
            source,
            provider,
            summary,
            str(surface.get("detail", "")),
            str(surface.get("badge", "")),
            str(surface.get("label", "")),
        ]
    ).upper()

    is_local_context = (
        source in {"local_fixture", "local-fixture"}
        or provider in {"local_fixture", "local-fixture"}
        or classification == LOCAL_FIXTURE_PASS_LABEL
        or "LOCAL-FIXTURE" in combined
        or "LOCAL_FIXTURE" in combined
    )

    if is_local_context:
        if classification == REAL_PROVIDER_PASS_LABEL:
            raise ClassificationHonestyError(
                "LOCAL-FIXTURE decision must not be labeled REAL-PROVIDER PASS"
            )
        if surface.get("real_provider_pass") is True:
            raise ClassificationHonestyError(
                "local fixture surface cannot set real_provider_pass=true"
            )
        if surface.get("production_mine_badge") is True:
            raise ClassificationHonestyError("local fixture cannot carry a production mine badge")
        if surface.get("live_emission_authority") is True:
            raise ClassificationHonestyError("local fixture cannot claim live-emission authority")
        if REAL_PROVIDER_PASS_LABEL in combined and "LOCAL-FIXTURE" in combined:
            # Mixed language (local + REAL-PROVIDER) is smuggling.
            raise ClassificationHonestyError(
                "summary smuggles REAL-PROVIDER language over a LOCAL-FIXTURE decision"
            )
        if REAL_PROVIDER_PASS_LABEL in combined or "REAL PROVIDER PASS" in combined:
            raise ClassificationHonestyError(
                "LOCAL-FIXTURE surface must not contain REAL-PROVIDER PASS language"
            )
        # Require an explicit local source marker in summary or source field.
        has_marker = any(
            marker.lower() in (summary + " " + source).lower() for marker in _LOCAL_SOURCE_MARKERS
        )
        if classification == LOCAL_FIXTURE_PASS_LABEL and not has_marker:
            raise ClassificationHonestyError(
                "LOCAL-FIXTURE PASS surface requires explicit local_fixture source label"
            )

    # Independent of fixture context: production-mine badges still cannot coexist with
    # a LOCALs classification value.
    if classification == LOCAL_FIXTURE_PASS_LABEL and any(
        badge in combined
        for badge in _FORBIDDEN_PRODUCTION_MINE_BADGES
        if badge != REAL_PROVIDER_PASS_LABEL
    ):
        # Allow intended REAL provider only when classification itself is REAL.
        if classification != REAL_PROVIDER_PASS_LABEL:
            # production mine / live emission next to LOCAL-FIXTURE is always smuggling.
            if (
                "PRODUCTION MINE" in combined
                or "LIVE-EMISSION" in combined
                or "LIVE EMISSION" in combined
            ):
                raise ClassificationHonestyError(
                    "LOCAL-FIXTURE PASS cannot claim production mine or live emission authority"
                )


def assert_not_real_provider_pass(
    *,
    classification: TeeClassification | str,
    provider: TeeProviderKind | str | None = None,
    would_grant_real_provider_pass: bool | None = None,
    surface: Mapping[str, Any] | None = None,
) -> None:
    """Hard check used by adapters/lab reports: current matrix never grants real PASS."""

    class_value = (
        classification.value
        if isinstance(classification, TeeClassification)
        else str(classification)
    )
    provider_value = (
        provider.value
        if isinstance(provider, TeeProviderKind)
        else (str(provider) if provider is not None else "")
    )
    if class_value == REAL_PROVIDER_PASS_LABEL:
        if provider_value in {"local_fixture", TeeProviderKind.LOCAL_FIXTURE.value}:
            raise ClassificationHonestyError("local fixture cannot produce REAL-PROVIDER PASS")
        # Present mission surface forbids real provider PASS even for Lium/Targon.
        raise ClassificationHonestyError(
            f"{provider_value or 'provider'} cannot produce REAL-PROVIDER PASS while "
            "would_grant_real_provider_pass is false"
        )
    if would_grant_real_provider_pass is True and provider_value in {
        "lium",
        "targon",
        TeeProviderKind.LIUM.value,
        TeeProviderKind.TARGON.value,
    }:
        # Mission hard gate: adapters report this false always.
        raise ClassificationHonestyError(
            f"{provider_value} readiness still reports would_grant_real_provider_pass-"
            "true which is mission-forbidden without authoritative contracts"
        )
    if surface is not None:
        assert_honest_classification_surface(surface)


def coerce_accepted_fixture_classification(
    classification: TeeClassification | str,
    *,
    provider: TeeProviderKind | str,
    reason: TeeReasonCode | str | None = None,
) -> TeeClassification:
    """Map an accepted local-fixture path onto LOCAL-FIXTURE PASS exclusively.

    Callers that attempt to pass REAL-PROVIDER for a local fixture are corrected
    rather than elevated (exception available via ``assert_not_real_provider_pass``).
    """

    provider_value = provider.value if isinstance(provider, TeeProviderKind) else str(provider)
    class_value = (
        classification
        if isinstance(classification, TeeClassification)
        else TeeClassification(str(classification))
    )
    reason_value = (
        reason.value if isinstance(reason, TeeReasonCode) else (str(reason) if reason else "")
    )
    if (
        provider_value == TeeProviderKind.LOCAL_FIXTURE.value
        or reason_value == TeeReasonCode.ACCEPTED_LOCAL_FIXTURE.value
    ):
        if class_value is TeeClassification.REAL_PROVIDER_PASS:
            return TeeClassification.LOCAL_FIXTURE_PASS
        return class_value
    return class_value


def smoke_deploy_labels(*, deploy_ok: bool) -> dict[str, str]:
    """Dual labels for paid Lium deploy smoke (never conflates with crypto PASS).

    Returns independent ``DEPLOY SMOKE`` and ``REAL-PROVIDER TEE`` labels so lab
    reports cannot smuggle infra success into cryptographic real-provider PASS.
    """

    return {
        "deploy_smoke": "DEPLOY SMOKE PASS" if deploy_ok else "DEPLOY SMOKE FAIL",
        "real_provider_tee": "BLOCKED",
        "real_provider_pass": "BLOCKED",
        "note": (
            "Paid Lium provision proves worker deploy reachability only; "
            "REAL-PROVIDER TEE PASS remains BLOCKED until HARD_GATE_ITEMS are "
            "authoritatively satisfied."
        ),
    }


__all__ = [
    "LOCAL_FIXTURE_PASS_LABEL",
    "REAL_PROVIDER_PASS_LABEL",
    "ClassificationHonestyError",
    "assert_honest_classification_surface",
    "assert_not_real_provider_pass",
    "coerce_accepted_fixture_classification",
    "decision_public_surface",
    "decision_validation_source",
    "human_summary_line",
    "smoke_deploy_labels",
]
