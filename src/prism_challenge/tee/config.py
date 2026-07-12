"""Pinned TEE verifier policy (fail-closed when incomplete)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prism_challenge.config import PrismSettings

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_AUDIENCE = "prism.tee.verify"
DEFAULT_PURPOSE = "execution_attestation"
DEFAULT_MEASUREMENT_KEYS = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")

MAX_QUOTE_B64_CHARS = 65_536
MAX_JWT_CHARS = 32_768
MAX_EVIDENCE_JSON_BYTES = 131_072
MAX_MEASUREMENT_VALUE_HEX = 128


@dataclass(frozen=True)
class TeeVerifierConfig:
    """Explicit verifier policy. Empty/incomplete config fails closed for tier 2."""

    enabled: bool = True
    mode: str = "local_fixture"  # local_fixture | production
    expected_provider: str = "local_fixture"
    expected_issuer: str = "prism-local-fixture"
    expected_audience: str = DEFAULT_AUDIENCE
    expected_purpose: str = DEFAULT_PURPOSE
    # PEM-encoded pinned trust roots for TDX local fixtures (one or two for rotation).
    tdx_trust_roots_pem: tuple[str, ...] = ()
    # Overlap window for dual-root rotation (inclusive), UTC-aware datetimes.
    trust_rotation_not_before: datetime | None = None
    trust_rotation_not_after: datetime | None = None
    # PEM public keys (or certs) allowed to sign GPU EATs, keyed by kid.
    gpu_trusted_keys_pem: Mapping[str, str] = field(default_factory=dict)
    expected_image_digest: str | None = None
    allowed_measurements: Mapping[str, str] = field(default_factory=dict)
    allowed_gpu_models: tuple[str, ...] = ("H100", "H200")
    allowed_gpu_vendors: tuple[str, ...] = ("nvidia",)
    require_gpu_security_mode: str = "cc"
    max_age_seconds: int = 3_600
    clock_skew_seconds: int = 30
    challenge_slug: str = "prism"
    workload_id: str | None = None
    workload_version: str | None = None
    # When true, missing nonce store / trust roots / measurements block elevation.
    require_nonce_store: bool = True
    # Real provider adapters stay blocked unless every hard gate is present.
    lium_ready: bool = False
    targon_ready: bool = False
    allow_network: bool = False
    # Optional provider contract snapshot. Authority markers inside this mapping
    # are required for real-provider readiness (never inferred from credentials).
    provider_contract: Mapping[str, Any] = field(default_factory=dict)
    max_quote_b64_chars: int = MAX_QUOTE_B64_CHARS
    max_jwt_chars: int = MAX_JWT_CHARS
    max_evidence_json_bytes: int = MAX_EVIDENCE_JSON_BYTES

    def readiness_gaps(self) -> tuple[str, ...]:
        """Public configuration gaps that prevent elevated TEE acceptance."""

        gaps: list[str] = []
        if not self.enabled:
            gaps.append("enabled=false")
        if not self.expected_issuer:
            gaps.append("expected_issuer")
        if not self.expected_audience:
            gaps.append("expected_audience")
        if not self.tdx_trust_roots_pem:
            gaps.append("tdx_trust_roots_pem")
        if not self.gpu_trusted_keys_pem:
            gaps.append("gpu_trusted_keys_pem")
        if not self.expected_image_digest or not _DIGEST_RE.fullmatch(self.expected_image_digest):
            gaps.append("expected_image_digest")
        if not self.allowed_measurements:
            gaps.append("allowed_measurements")
        else:
            for key in DEFAULT_MEASUREMENT_KEYS:
                value = self.allowed_measurements.get(key)
                if not value or not _HEX64_RE.fullmatch(value.lower()):
                    gaps.append(f"allowed_measurements.{key}")
        return tuple(gaps)

    def is_ready_for_local_fixture(self) -> bool:
        return self.enabled and not self.readiness_gaps()

    def fingerprints(self) -> dict[str, Any]:
        """Public fingerprints of pinned trust material (never raw secrets)."""

        root_fps = tuple(
            hashlib.sha256(pem.encode("utf-8")).hexdigest()[:32] for pem in self.tdx_trust_roots_pem
        )
        key_fps = {
            kid: hashlib.sha256(pem.encode("utf-8")).hexdigest()[:32]
            for kid, pem in sorted(self.gpu_trusted_keys_pem.items())
        }
        return {
            "tdx_trust_root_fingerprints": root_fps,
            "gpu_key_fingerprints": key_fps,
            "expected_image_digest": self.expected_image_digest,
            "expected_issuer": self.expected_issuer,
            "expected_audience": self.expected_audience,
        }


def tee_config_from_settings(settings: PrismSettings) -> TeeVerifierConfig:
    """Build verifier config from Prism settings (nested ``tee`` block when present)."""

    worker_plane = settings.worker_plane
    enabled = bool(getattr(settings, "tee_verification_enabled", True))
    raw: Any = getattr(settings, "tee", None)
    if raw is None:
        return TeeVerifierConfig(
            enabled=enabled,
            expected_image_digest=worker_plane.pinned_image_digest,
            challenge_slug=getattr(settings, "slug", "prism") or "prism",
        )
    if isinstance(raw, TeeVerifierConfig):
        return raw

    # Pydantic TeeConfig or mapping-like.
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        return TeeVerifierConfig(
            enabled=enabled,
            expected_image_digest=worker_plane.pinned_image_digest,
            challenge_slug=getattr(settings, "slug", "prism") or "prism",
        )

    return TeeVerifierConfig(
        enabled=bool(data.get("enabled", enabled)) and enabled,
        mode=str(data.get("mode", "local_fixture")),
        expected_provider=str(data.get("expected_provider", "local_fixture")),
        expected_issuer=str(data.get("expected_issuer", "prism-local-fixture")),
        expected_audience=str(data.get("expected_audience", DEFAULT_AUDIENCE)),
        expected_purpose=str(data.get("expected_purpose", DEFAULT_PURPOSE)),
        tdx_trust_roots_pem=tuple(data.get("tdx_trust_roots_pem") or ()),
        trust_rotation_not_before=data.get("trust_rotation_not_before"),
        trust_rotation_not_after=data.get("trust_rotation_not_after"),
        gpu_trusted_keys_pem=dict(data.get("gpu_trusted_keys_pem") or {}),
        expected_image_digest=data.get("expected_image_digest") or worker_plane.pinned_image_digest,
        allowed_measurements=dict(data.get("allowed_measurements") or {}),
        allowed_gpu_models=tuple(data.get("allowed_gpu_models") or ("H100", "H200")),
        allowed_gpu_vendors=tuple(data.get("allowed_gpu_vendors") or ("nvidia",)),
        require_gpu_security_mode=str(data.get("require_gpu_security_mode", "cc")),
        max_age_seconds=int(data.get("max_age_seconds", 3_600)),
        clock_skew_seconds=int(data.get("clock_skew_seconds", 30)),
        challenge_slug=str(
            data.get("challenge_slug") or getattr(settings, "slug", "prism") or "prism"
        ),
        workload_id=data.get("workload_id"),
        workload_version=data.get("workload_version") or "1",
        require_nonce_store=bool(data.get("require_nonce_store", True)),
        lium_ready=bool(data.get("lium_ready", False)),
        targon_ready=bool(data.get("targon_ready", False)),
        allow_network=bool(data.get("allow_network", False)),
        provider_contract=dict(data.get("provider_contract") or {}),
    )
