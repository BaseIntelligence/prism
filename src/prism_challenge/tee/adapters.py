"""Provider-scoped TEE adapters (explicit selection, fail-closed readiness)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import TeeVerifierConfig
from .readiness import (
    ProviderReadinessReport,
    SafeProbeReport,
    WatchtowerEvaluation,
    classify_safe_probe,
    evaluate_provider_readiness,
    evaluate_watchtower_digest,
)
from .types import TeeClassification, TeeProviderKind, TeeReasonCode, fail_decision


class ProviderAdapter(Protocol):
    name: TeeProviderKind

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]: ...

    def readiness_report(self, config: TeeVerifierConfig) -> ProviderReadinessReport: ...


@dataclass(frozen=True)
class LocalFixtureAdapter:
    name: TeeProviderKind = TeeProviderKind.LOCAL_FIXTURE

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]:
        report = self.readiness_report(config)
        return report.ready, report.reason, report.detail

    def readiness_report(self, config: TeeVerifierConfig) -> ProviderReadinessReport:
        return evaluate_provider_readiness(TeeProviderKind.LOCAL_FIXTURE, config)


@dataclass(frozen=True)
class LiumAdapter:
    """Prepared Lium adapter — real PASS remains BLOCKED without full gate."""

    name: TeeProviderKind = TeeProviderKind.LIUM

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]:
        report = self.readiness_report(config)
        # ready is always False for real Lium in this mission.
        return report.ready, report.reason, report.detail

    def readiness_report(self, config: TeeVerifierConfig) -> ProviderReadinessReport:
        return evaluate_provider_readiness(TeeProviderKind.LIUM, config)

    def classify_probe(
        self,
        *,
        api_reachable: bool | None = None,
        http_status: int | None = None,
        path: str | None = None,
        method: str = "GET",
        config: TeeVerifierConfig | None = None,
    ) -> dict[str, Any]:
        readiness = self.readiness_report(config) if config is not None else None
        report = classify_safe_probe(
            TeeProviderKind.LIUM,
            api_reachable=api_reachable,
            http_status=http_status,
            path=path,
            method=method,
            readiness=readiness,
        )
        return report.as_dict()

    def evaluate_watchtower(
        self,
        payload: dict[str, Any] | None,
        *,
        config: TeeVerifierConfig | None = None,
        expected_image_digest: str | None = None,
        now: Any | None = None,
        max_age_seconds: int | None = None,
    ) -> WatchtowerEvaluation:
        digest = expected_image_digest
        if digest is None and config is not None:
            digest = config.expected_image_digest
        age = max_age_seconds
        if age is None and config is not None:
            age = int(config.max_age_seconds)
        if age is None:
            age = 3_600
        return evaluate_watchtower_digest(
            payload,
            expected_image_digest=digest,
            max_age_seconds=age,
            now=now,
        )


@dataclass(frozen=True)
class TargonAdapter:
    """Future/blocked Targon surface — never confers tier 2."""

    name: TeeProviderKind = TeeProviderKind.TARGON

    def readiness(self, config: TeeVerifierConfig) -> tuple[bool, TeeReasonCode, str]:
        report = self.readiness_report(config)
        return report.ready, report.reason, report.detail

    def readiness_report(self, config: TeeVerifierConfig) -> ProviderReadinessReport:
        return evaluate_provider_readiness(TeeProviderKind.TARGON, config)

    def classify_probe(
        self,
        *,
        api_reachable: bool | None = None,
        http_status: int | None = None,
        path: str | None = None,
        method: str = "GET",
        config: TeeVerifierConfig | None = None,
    ) -> dict[str, Any]:
        readiness = self.readiness_report(config) if config is not None else None
        report = classify_safe_probe(
            TeeProviderKind.TARGON,
            api_reachable=api_reachable,
            http_status=http_status,
            path=path,
            method=method,
            readiness=readiness,
        )
        return report.as_dict()


def select_adapter(
    provider: TeeProviderKind | str, config: TeeVerifierConfig
) -> LocalFixtureAdapter | LiumAdapter | TargonAdapter | None:
    """Return the exact provider-scoped adapter or None for unknown names."""

    _ = config  # selection is name-scoped; readiness still takes the config
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
    report = adapter.readiness_report(config)
    if report.ready:
        return None
    classification = (
        TeeClassification.BLOCKED
        if report.reason
        in {
            TeeReasonCode.PROVIDER_BLOCKED,
            TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
            TeeReasonCode.ADAPTER_NOT_READY,
            TeeReasonCode.VERIFIER_MISCONFIGURED,
        }
        else TeeClassification.FAIL
        if report.classification is TeeClassification.FAIL
        else TeeClassification.BLOCKED
    )
    return fail_decision(
        reason=report.reason,
        provider=provider,
        classification=classification,
        detail=report.detail,
    )


__all__ = [
    "LiumAdapter",
    "LocalFixtureAdapter",
    "ProviderAdapter",
    "ProviderReadinessReport",
    "SafeProbeReport",
    "TargonAdapter",
    "WatchtowerEvaluation",
    "blocked_for_provider",
    "classify_safe_probe",
    "evaluate_provider_readiness",
    "evaluate_watchtower_digest",
    "select_adapter",
]
