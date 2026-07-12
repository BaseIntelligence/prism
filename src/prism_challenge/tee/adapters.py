"""Provider-scoped TEE adapters (explicit selection, fail-closed readiness)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import TeeVerifierConfig
from .types import TeeClassification, TeeProviderKind, TeeReasonCode, fail_decision


class ProviderAdapter(Protocol):
    name: TeeProviderKind

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]: ...


@dataclass(frozen=True)
class LocalFixtureAdapter:
    name: TeeProviderKind = TeeProviderKind.LOCAL_FIXTURE

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]:
        if config.mode != "local_fixture" and config.expected_provider != "local_fixture":
            return False, TeeReasonCode.ADAPTER_NOT_READY, "local adapter not selected"
        gaps = config.readiness_gaps()
        if gaps:
            return (
                False,
                TeeReasonCode.VERIFIER_MISCONFIGURED,
                f"missing: {','.join(gaps)}",
            )
        return True, TeeReasonCode.ACCEPTED_LOCAL_FIXTURE, "local fixture ready"


@dataclass(frozen=True)
class LiumAdapter:
    """Prepared Lium adapter — real PASS remains BLOCKED without full gate."""

    name: TeeProviderKind = TeeProviderKind.LIUM

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]:
        if config.lium_ready:
            # Hard gate: even if an operator flips the flag, every dependency must exist.
            gaps = list(config.readiness_gaps())
            if not gaps:
                # Still blocked for REAL PASS until external authoritative contract exists.
                return (
                    False,
                    TeeReasonCode.PROVIDER_BLOCKED,
                    "lium contract not authoritative for real attestation",
                )
            return False, TeeReasonCode.ADAPTER_NOT_READY, f"lium incomplete: {','.join(gaps)}"
        return (
            False,
            TeeReasonCode.PROVIDER_BLOCKED,
            "lium real-provider validation blocked pending authoritative contract",
        )

    def classify_probe(self) -> dict[str, str]:
        return {
            "provider_api_reachable": "unknown",
            "tee_validation": TeeClassification.BLOCKED.value,
        }


@dataclass(frozen=True)
class TargonAdapter:
    """Future/blocked Targon surface — never confers tier 2."""

    name: TeeProviderKind = TeeProviderKind.TARGON

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]:
        if config.targon_ready:
            gaps = list(config.readiness_gaps())
            if gaps:
                return (
                    False,
                    TeeReasonCode.ADAPTER_NOT_READY,
                    f"targon incomplete: {','.join(gaps)}",
                )
            return (
                False,
                TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
                "targon future enablement still lacks authoritative contract",
            )
        return (
            False,
            TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
            "targon future/blocked by default",
        )


def select_adapter(
    provider: TeeProviderKind | str, config: TeeVerifierConfig
) -> ProviderAdapter | None:
    """Return the exact provider-scoped adapter or None for unknown names."""

    name = provider.value if isinstance(provider, TeeProviderKind) else str(provider)
    if name in {"local_fixture", TeeProviderKind.LOCAL_FIXTURE.value}:
        return LocalFixtureAdapter()
    if name in {"lium", TeeProviderKind.LIUM.value}:
        return LiumAdapter()
    if name in {"targon", TeeProviderKind.TARGON.value}:
        return TargonAdapter()
    return None


def blocked_for_provider(provider: TeeProviderKind, config: TeeVerifierConfig):
    adapter = select_adapter(provider, config)
    if adapter is None:
        return fail_decision(
            reason=TeeReasonCode.EVIDENCE_UNKNOWN_PROVIDER,
            provider=provider,
            classification=TeeClassification.FAIL,
            detail=f"no adapter for provider {provider}",
        )
    ready, reason, detail = adapter.readiness(config)
    if ready:
        return None
    classification = (
        TeeClassification.BLOCKED
        if reason
        in {
            TeeReasonCode.PROVIDER_BLOCKED,
            TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
            TeeReasonCode.ADAPTER_NOT_READY,
        }
        else TeeClassification.FAIL
    )
    return fail_decision(
        reason=reason,
        provider=provider,
        classification=classification,
        detail=detail,
    )
