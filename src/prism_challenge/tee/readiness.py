"""Authoritative real-provider readiness gates (Lium/Targon).

Real ``REAL-PROVIDER PASS`` remains impossible until every hard-gate input is
independently consumable from provider documentation. Credentials, inventory
reachability, non-empty quotes/JWTs, watchtower digests, paid pod metadata, and
operator name flags never satisfy the gate.

``HARD_GATE_ITEMS`` (11 authoritative dependencies) is still in force for any
future real-provider unlock. Adapters expose ``would_grant_real_provider_pass=false``
for Lium and Targon on every path in this mission surface
(VAL-TEEREQ-004/005, VAL-TEE-043/047/051/052).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import _DIGEST_RE, TeeVerifierConfig
from .types import TeeClassification, TeeProviderKind, TeeReasonCode

# Hard-gate checklist IDs shared by real-provider adapters (VAL-TEE-043, 047, 051, 052).
HARD_GATE_ITEMS: tuple[str, ...] = (
    "authoritative_evidence_endpoint",
    "authoritative_evidence_format",
    "authoritative_issuer_audience",
    "authoritative_trust_roots",
    "freshness_and_clock_policy",
    "nonce_semantics",
    "digest_pinned_public_worker_image",
    "measurement_policy",
    "gpu_claim_policy",
    "cross_binding_semantics",
    "real_workload_evidence_artifact",
)

# Properties a watchtower digest can never establish for a specific execution.
WATCHTOWER_UNBOUND: tuple[str, ...] = (
    "nonce",
    "workload_identity",
    "tdx_measurement",
    "gpu_identity",
    "manifest_binding",
    "execution_freshness",
    "cross_component_binding",
)


@dataclass(frozen=True)
class GateItem:
    """One hard-gate dependency with explicit available/missing state."""

    item_id: str
    available: bool
    source: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderReadinessReport:
    """Machine-readable readiness for a provider adapter (never secret-bearing)."""

    provider: TeeProviderKind
    ready: bool
    classification: TeeClassification
    reason: TeeReasonCode
    detail: str
    checklist: tuple[GateItem, ...] = ()
    missing_items: tuple[str, ...] = ()
    # Caps any TEE elevation claim from this adapter path alone.
    max_effective_tier: int = 0
    would_grant_real_provider_pass: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "ready": self.ready,
            "classification": self.classification.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "checklist": [item.as_dict() for item in self.checklist],
            "missing_items": list(self.missing_items),
            "max_effective_tier": self.max_effective_tier,
            "would_grant_real_provider_pass": self.would_grant_real_provider_pass,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WatchtowerEvaluation:
    """Evaluation of Lium watchtower-shaped metadata (never attestation)."""

    accepted_as_tier1_input: bool
    effective_tier: int
    reason: TeeReasonCode
    detail: str
    unbound_properties: tuple[str, ...] = WATCHTOWER_UNBOUND
    image_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_as_tier1_input": self.accepted_as_tier1_input,
            "effective_tier": self.effective_tier,
            "reason": self.reason.value,
            "detail": self.detail,
            "unbound_properties": list(self.unbound_properties),
            "image_digest": self.image_digest,
            "metadata": dict(self.metadata),
            "tee_validation": TeeClassification.BLOCKED.value,
            "grants_tier_2": False,
        }


@dataclass(frozen=True)
class SafeProbeReport:
    """Classification of safe read-only provider probes (no mutation, no PASS)."""

    provider: TeeProviderKind
    provider_api_reachable: str  # "true" | "false" | "unknown"
    tee_validation: str  # always BLOCKED unless full real-provider gate met
    mutated: bool
    provisioned: bool
    methods_allowed: tuple[str, ...] = ("GET",)
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str | bool | dict[str, Any] | list[str]]:
        return {
            "provider": self.provider.value,
            "provider_api_reachable": self.provider_api_reachable,
            "tee_validation": self.tee_validation,
            "mutated": self.mutated,
            "provisioned": self.provisioned,
            "methods_allowed": list(self.methods_allowed),
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }


def _presence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict, Mapping)):
        return len(value) > 0
    return bool(value)


def _base_local_pins_present(config: TeeVerifierConfig) -> dict[str, bool]:
    """Shared pins that real providers would also need once contracts exist."""

    measurements_ok = bool(config.allowed_measurements) and all(
        _presence(config.allowed_measurements.get(key))
        for key in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")
    )
    return {
        "tdx_trust_roots_pem": bool(config.tdx_trust_roots_pem),
        "gpu_trusted_keys_pem": bool(config.gpu_trusted_keys_pem),
        "expected_image_digest": bool(
            config.expected_image_digest and _DIGEST_RE.fullmatch(config.expected_image_digest)
        ),
        "allowed_measurements": measurements_ok,
        "expected_issuer": bool(config.expected_issuer),
        "expected_audience": bool(config.expected_audience),
    }


def build_hard_gate_checklist(
    config: TeeVerifierConfig,
    *,
    provider: TeeProviderKind,
) -> tuple[GateItem, ...]:
    """Enumerate every authoritative dependency and its current state.

    Until a real provider publishes a consumer-verifiable contract, every item that
    depends on external authority is reported missing even when local pins are set
    for fixture testing. Local pins never convert into REAL-PROVIDER readiness.
    """

    pins = _base_local_pins_present(config)
    # Local material may be present for fixture work; real-provider authority is independent.
    raw_contract = config.provider_contract
    contract: Mapping[str, Any]
    if isinstance(raw_contract, Mapping):
        contract = raw_contract
    else:
        contract = {}

    endpoints = _presence(contract.get("evidence_endpoint")) and str(
        contract.get("evidence_endpoint", "")
    ).startswith("https://")
    evidence_format = _presence(contract.get("evidence_format")) and bool(
        contract.get("format_authoritative")
    )
    issuer_audience = (
        _presence(contract.get("issuer"))
        and _presence(contract.get("audience"))
        and bool(contract.get("issuer_audience_authoritative"))
    )
    # Trust roots are authoritative only when the contract says so — local PEM pins do not count.
    trust_roots = bool(contract.get("trust_roots_authoritative")) and pins["tdx_trust_roots_pem"]
    freshness = bool(contract.get("freshness_policy_documented"))
    nonce = bool(contract.get("nonce_semantics_documented"))
    public_image = (
        _presence(contract.get("public_image_reference"))
        and bool(contract.get("public_image_digest_resolvable"))
        and pins["expected_image_digest"]
    )
    measurements = (
        bool(contract.get("measurement_policy_documented")) and pins["allowed_measurements"]
    )
    gpu_policy = bool(contract.get("gpu_claim_policy_documented")) and pins["gpu_trusted_keys_pem"]
    cross_binding = bool(contract.get("cross_binding_documented"))
    # Real workload artifact is never auto-available from configuration alone.
    real_artifact = bool(contract.get("real_workload_evidence_available"))

    # Credential / operator flags are intentionally NOT checklist successes.
    status = {
        "authoritative_evidence_endpoint": endpoints,
        "authoritative_evidence_format": evidence_format,
        "authoritative_issuer_audience": issuer_audience,
        "authoritative_trust_roots": trust_roots,
        "freshness_and_clock_policy": freshness,
        "nonce_semantics": nonce,
        "digest_pinned_public_worker_image": public_image,
        "measurement_policy": measurements,
        "gpu_claim_policy": gpu_policy,
        "cross_binding_semantics": cross_binding,
        "real_workload_evidence_artifact": real_artifact,
    }

    sources = {
        "authoritative_evidence_endpoint": "provider.contract.evidence_endpoint",
        "authoritative_evidence_format": "provider.contract.evidence_format",
        "authoritative_issuer_audience": "provider.contract.issuer+audience",
        "authoritative_trust_roots": "provider.contract.trust_roots_authoritative",
        "freshness_and_clock_policy": "provider.contract.freshness_policy_documented",
        "nonce_semantics": "provider.contract.nonce_semantics_documented",
        "digest_pinned_public_worker_image": "provider.contract.public_image_*",
        "measurement_policy": "provider.contract.measurement_policy_documented",
        "gpu_claim_policy": "provider.contract.gpu_claim_policy_documented",
        "cross_binding_semantics": "provider.contract.cross_binding_documented",
        "real_workload_evidence_artifact": "provider.contract.real_workload_evidence_available",
    }

    missing_hints = {
        "authoritative_evidence_endpoint": "no documented https evidence endpoint",
        "authoritative_evidence_format": "no authoritative evidence format/version",
        "authoritative_issuer_audience": "no documented issuer/audience contract",
        "authoritative_trust_roots": "trust roots not marked provider-authoritative",
        "freshness_and_clock_policy": "freshness/max-age policy undocumented",
        "nonce_semantics": "nonce and anti-replay semantics undocumented",
        "digest_pinned_public_worker_image": "no public digest-pinned worker image",
        "measurement_policy": "measurement allowlist not provider-documented",
        "gpu_claim_policy": "GPU claim policy not provider-documented",
        "cross_binding_semantics": "TDX/GPU cross-binding semantics undocumented",
        "real_workload_evidence_artifact": "no real workload evidence artifact available",
    }

    items: list[GateItem] = []
    for item_id in HARD_GATE_ITEMS:
        available = bool(status[item_id])
        items.append(
            GateItem(
                item_id=item_id,
                available=available,
                source=sources[item_id],
                detail="" if available else missing_hints[item_id],
            )
        )
    # Attach non-gate signals so readiness output documents credentials ≠ readiness.
    items.append(
        GateItem(
            item_id="operator_ready_flag",
            available=bool(
                config.lium_ready if provider is TeeProviderKind.LIUM else config.targon_ready
            ),
            source="config.lium_ready"
            if provider is TeeProviderKind.LIUM
            else "config.targon_ready",
            detail="operator flag is not an authoritative hard-gate input",
        )
    )
    items.append(
        GateItem(
            item_id="credentials_present",
            available=False,
            source="environment",
            detail="credentials never establish readiness (always non-authoritative)",
        )
    )
    return tuple(items)


def evaluate_provider_readiness(
    provider: TeeProviderKind | str,
    config: TeeVerifierConfig,
) -> ProviderReadinessReport:
    """Return the readiness report for Lium/Targon/local. Never grants REAL-PROVIDER PASS."""

    name = provider.value if isinstance(provider, TeeProviderKind) else str(provider)
    if name == TeeProviderKind.LOCAL_FIXTURE.value:
        gaps = list(config.readiness_gaps())
        ready = config.enabled and not gaps
        return ProviderReadinessReport(
            provider=TeeProviderKind.LOCAL_FIXTURE,
            ready=ready,
            classification=(
                TeeClassification.LOCAL_FIXTURE_PASS if ready else TeeClassification.BLOCKED
            ),
            reason=(
                TeeReasonCode.ACCEPTED_LOCAL_FIXTURE
                if ready
                else TeeReasonCode.VERIFIER_MISCONFIGURED
            ),
            detail="local fixture ready"
            if ready
            else f"missing: {','.join(gaps) or 'enabled=false'}",
            checklist=(),
            missing_items=tuple(gaps),
            max_effective_tier=2 if ready else 0,
            would_grant_real_provider_pass=False,
            metadata={"validation_source": "local_fixture"},
        )

    if name == TeeProviderKind.LIUM.value:
        return _evaluate_lium(config)
    if name == TeeProviderKind.TARGON.value:
        return _evaluate_targon(config)
    return ProviderReadinessReport(
        provider=TeeProviderKind.UNKNOWN,
        ready=False,
        classification=TeeClassification.FAIL,
        reason=TeeReasonCode.EVIDENCE_UNKNOWN_PROVIDER,
        detail=f"unknown provider {name}",
        checklist=(),
        missing_items=("provider",),
        max_effective_tier=0,
        would_grant_real_provider_pass=False,
    )


def _missing(checklist: Sequence[GateItem]) -> tuple[str, ...]:
    return tuple(
        item.item_id for item in checklist if item.item_id in HARD_GATE_ITEMS and not item.available
    )


def _evaluate_lium(config: TeeVerifierConfig) -> ProviderReadinessReport:
    checklist = build_hard_gate_checklist(config, provider=TeeProviderKind.LIUM)
    missing = _missing(checklist)
    # Even an operator-flipped ready flag with every boolean true cannot mint REAL PASS
    # inside this mission: the contract surface is not published, so real_workload and
    # authority markers remain the gating truth. If somehow every hard gate were true,
    # still refuse REAL-PROVIDER PASS here (mission defers real validation).
    all_gates = not missing
    if not config.lium_ready or missing:
        return ProviderReadinessReport(
            provider=TeeProviderKind.LIUM,
            ready=False,
            classification=TeeClassification.BLOCKED,
            reason=(
                TeeReasonCode.ADAPTER_NOT_READY
                if config.lium_ready and missing
                else TeeReasonCode.PROVIDER_BLOCKED
            ),
            detail=(
                f"lium incomplete: {','.join(missing)}"
                if missing
                else "lium real-provider validation blocked pending authoritative contract"
            ),
            checklist=checklist,
            missing_items=missing,
            max_effective_tier=0,
            would_grant_real_provider_pass=False,
            metadata={
                "operator_lium_ready": config.lium_ready,
                "all_hard_gates_present": all_gates,
                "real_provider_pass_allowed": False,
            },
        )
    # lium_ready and no missing — still blocked: no dual authority in this code path.
    return ProviderReadinessReport(
        provider=TeeProviderKind.LIUM,
        ready=False,
        classification=TeeClassification.BLOCKED,
        reason=TeeReasonCode.PROVIDER_BLOCKED,
        detail=(
            "lium contract not authoritative for real attestation "
            "(all local pins present but REAL-PROVIDER PASS is mission-blocked)"
        ),
        checklist=checklist,
        missing_items=() if all_gates else missing,
        max_effective_tier=0,
        would_grant_real_provider_pass=False,
        metadata={
            "operator_lium_ready": True,
            "all_hard_gates_present": all_gates,
            "real_provider_pass_allowed": False,
        },
    )


def _evaluate_targon(config: TeeVerifierConfig) -> ProviderReadinessReport:
    checklist = build_hard_gate_checklist(config, provider=TeeProviderKind.TARGON)
    missing = _missing(checklist)
    # Targon remains future/blocked by default and under partial enablement.
    if not config.targon_ready:
        return ProviderReadinessReport(
            provider=TeeProviderKind.TARGON,
            ready=False,
            classification=TeeClassification.BLOCKED,
            reason=TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
            detail="targon future/blocked by default",
            checklist=checklist,
            missing_items=missing or HARD_GATE_ITEMS,
            max_effective_tier=0,
            would_grant_real_provider_pass=False,
            metadata={
                "future_or_blocked": True,
                "operator_targon_ready": False,
                "speculative_endpoints_attempted": 0,
            },
        )
    if missing:
        return ProviderReadinessReport(
            provider=TeeProviderKind.TARGON,
            ready=False,
            classification=TeeClassification.BLOCKED,
            reason=TeeReasonCode.ADAPTER_NOT_READY,
            detail=f"targon incomplete: {','.join(missing)}",
            checklist=checklist,
            missing_items=missing,
            max_effective_tier=0,
            would_grant_real_provider_pass=False,
            metadata={
                "future_or_blocked": True,
                "operator_targon_ready": True,
                "speculative_endpoints_attempted": 0,
            },
        )
    return ProviderReadinessReport(
        provider=TeeProviderKind.TARGON,
        ready=False,
        classification=TeeClassification.BLOCKED,
        reason=TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
        detail="targon future enablement still lacks authoritative contract",
        checklist=checklist,
        missing_items=(),
        max_effective_tier=0,
        would_grant_real_provider_pass=False,
        metadata={
            "future_or_blocked": True,
            "operator_targon_ready": True,
            "speculative_endpoints_attempted": 0,
            "real_provider_pass_allowed": False,
        },
    )


def evaluate_watchtower_digest(
    payload: Mapping[str, Any] | None,
    *,
    expected_image_digest: str | None,
    now: datetime | None = None,
    max_age_seconds: int = 3_600,
) -> WatchtowerEvaluation:
    """Assess Lium watchtower-shaped metadata.

    A perfectly signed + fresh + matching digest is at most a tier-1 provenance
    *input* after signature validation — never tier 2 and never TEE PASS
    (VAL-TEE-044, VAL-TEE-045).
    """

    now = now or datetime.now(UTC)
    if not payload:
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
            detail="watchtower payload absent",
            image_digest=None,
        )

    digest = payload.get("digest") or payload.get("image_digest")
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
            detail="watchtower digest malformed or bare non-digest string",
            image_digest=digest if isinstance(digest, str) else None,
        )

    signature_valid = bool(payload.get("signature_valid"))
    signing_key_known = bool(payload.get("signing_key_known"))
    if not signature_valid or not signing_key_known:
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
            detail="watchtower signature invalid or key unknown",
            image_digest=digest,
            metadata={
                "signature_valid": signature_valid,
                "signing_key_known": signing_key_known,
            },
        )

    # Staleness
    ts_raw = payload.get("timestamp") or payload.get("issued_at")
    fresh = True
    if ts_raw is None:
        fresh = False
    else:
        if isinstance(ts_raw, datetime):
            ts = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=UTC)
        elif isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(float(ts_raw), tz=UTC)
        elif isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                return WatchtowerEvaluation(
                    accepted_as_tier1_input=False,
                    effective_tier=0,
                    reason=TeeReasonCode.FRESHNESS_INVALID,
                    detail="watchtower timestamp unparsable",
                    image_digest=digest,
                )
        else:
            return WatchtowerEvaluation(
                accepted_as_tier1_input=False,
                effective_tier=0,
                reason=TeeReasonCode.FRESHNESS_INVALID,
                detail="watchtower timestamp wrong type",
                image_digest=digest,
            )
        age = (now - ts).total_seconds()
        if age < -30 or age > max_age_seconds:
            fresh = False

    if not fresh:
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.FRESHNESS_INVALID,
            detail="watchtower timestamp stale or missing",
            image_digest=digest,
        )

    matches = bool(expected_image_digest) and digest == expected_image_digest
    if not matches:
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.IMAGE_DIGEST_MISMATCH,
            detail="watchtower digest does not match expected image pin",
            image_digest=digest,
        )

    # Pod/executor mismatch: if the payload carries them and they fail explicit expected ids.
    expected_pod = payload.get("expected_pod_id")
    expected_executor = payload.get("expected_executor_id")
    pod = payload.get("pod_id")
    executor = payload.get("executor_id")
    if expected_pod is not None and pod != expected_pod:
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
            detail="watchtower pod_id does not match expected workload",
            image_digest=digest,
        )
    if expected_executor is not None and executor != expected_executor:
        return WatchtowerEvaluation(
            accepted_as_tier1_input=False,
            effective_tier=0,
            reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
            detail="watchtower executor_id does not match expected workload",
            image_digest=digest,
        )

    # Success path for provenance input only — still not TEE tier 2.
    return WatchtowerEvaluation(
        accepted_as_tier1_input=True,
        effective_tier=1,
        reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
        detail=(
            "watchtower digest accepted as tier-1 provenance input only; "
            "cannot establish TEE attestation"
        ),
        unbound_properties=WATCHTOWER_UNBOUND,
        image_digest=digest,
        metadata={
            "path": "/watchtower/digest",
            "method": "GET",
            "grants_tier_2": False,
            "tee_attestation": False,
        },
    )


def classify_safe_probe(
    provider: TeeProviderKind | str,
    *,
    api_reachable: bool | None = None,
    http_status: int | None = None,
    path: str | None = None,
    method: str = "GET",
    readiness: ProviderReadinessReport | None = None,
) -> SafeProbeReport:
    """Classify a safe read-only probe. Never becomes REAL-PROVIDER PASS.

    Inventory, account, offers, apps, workloads, state, events, logs, and
    watchtower digest GETs may only establish API reachability (VAL-TEE-046/050).
    """

    name = provider.value if isinstance(provider, TeeProviderKind) else str(provider)
    kind = (
        TeeProviderKind.LIUM
        if name == "lium"
        else TeeProviderKind.TARGON
        if name == "targon"
        else TeeProviderKind.UNKNOWN
    )
    method_u = (method or "GET").upper()
    mutated = method_u not in {"GET", "HEAD", "OPTIONS"}
    if api_reachable is None and http_status is not None:
        api_reachable = 200 <= int(http_status) < 400
    reachable = (
        "true" if api_reachable is True else "false" if api_reachable is False else "unknown"
    )

    # Full real-provider gate is never met in this mission surface.
    tee_validation = TeeClassification.BLOCKED.value
    if readiness is not None and readiness.would_grant_real_provider_pass:
        # Defensive: even if a future report claimed readiness, this helper
        # refuses to emit PASS without an explicit separate path.
        tee_validation = TeeClassification.BLOCKED.value

    return SafeProbeReport(
        provider=kind,
        provider_api_reachable=reachable,
        tee_validation=tee_validation,
        mutated=mutated,
        provisioned=False,
        methods_allowed=("GET", "HEAD", "OPTIONS") if not mutated else (),
        detail=(
            f"safe probe classified for path={path or '<unspecified>'}; "
            "API reachability is independent of TEE validation"
        ),
        metadata={
            "path": path,
            "method": method_u,
            "http_status": http_status,
            "provider_resources_created": 0,
        },
    )


def real_provider_pass_is_possible(
    config: TeeVerifierConfig, provider: TeeProviderKind | str
) -> bool:
    """Mission-scoped gate: always False until an authoritative contract lands."""

    report = evaluate_provider_readiness(provider, config)
    return bool(report.would_grant_real_provider_pass and report.ready and not report.missing_items)
