"""Closed, bounded TEE evidence schema parsing (fail closed)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import (
    DEFAULT_MEASUREMENT_KEYS,
    MAX_EVIDENCE_JSON_BYTES,
    MAX_JWT_CHARS,
    MAX_QUOTE_B64_CHARS,
)
from .types import TeeProviderKind, TeeReasonCode, fail_decision

ALLOWED_EVIDENCE_TOP_LEVEL = frozenset(
    {
        "version",
        "provider",
        "evidence_type",
        "tdx_quote_b64",
        "gpu_eat_jwt",
        # Forbidden locators (detected and rejected, never followed).
        "jku",
        "x5u",
        "jwks_uri",
        "trust_bundle_url",
        "trust_bundle_path",
    }
)

ALLOWED_PROVIDERS = frozenset({"local_fixture", "lium", "targon"})
ALLOWED_EVIDENCE_TYPES = frozenset({"prism.tee.v1"})
ALLOWED_VERSIONS = frozenset({1, "1", "1.0"})
_B64_URLSAFE_RE = re.compile(r"^[A-Za-z0-9_\-]+={0,2}$")
_B64_STD_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


class EvidenceParseError(Exception):
    def __init__(self, reason: TeeReasonCode, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(detail or reason.value)


@dataclass(frozen=True)
class ParsedEvidence:
    version: str
    provider: TeeProviderKind
    evidence_type: str
    tdx_quote_b64: str
    gpu_eat_jwt: str
    raw_digest: str

    def evidence_digest(self) -> str:
        return self.raw_digest


def _reject(reason: TeeReasonCode, detail: str = "") -> None:
    raise EvidenceParseError(reason, detail)


def evidence_bytes_digest(raw: Mapping[str, Any]) -> str:
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_attestation_mapping(
    attestation: Any,
    *,
    max_quote_b64_chars: int = MAX_QUOTE_B64_CHARS,
    max_jwt_chars: int = MAX_JWT_CHARS,
    max_evidence_json_bytes: int = MAX_EVIDENCE_JSON_BYTES,
) -> ParsedEvidence:
    """Strict closed-schema parse of ``proof.attestation``.

    Presence of ``tdx_quote_b64`` / ``gpu_eat_jwt`` alone is never treated as attestation;
    both components must exist as bounded, well-typed strings after version/provider checks.
    """

    if attestation is None:
        _reject(TeeReasonCode.EVIDENCE_MISSING, "attestation is null/absent")
    if not isinstance(attestation, Mapping):
        _reject(TeeReasonCode.EVIDENCE_WRONG_TYPE, f"attestation type {type(attestation).__name__}")
    if not attestation:
        _reject(TeeReasonCode.EVIDENCE_MISSING, "attestation is empty")

    # Bound overall size via canonical JSON.
    try:
        raw_bytes = json.dumps(
            dict(attestation), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _reject(TeeReasonCode.EVIDENCE_MALFORMED, f"json encode failed: {exc}")
    if len(raw_bytes) > max_evidence_json_bytes:
        _reject(TeeReasonCode.EVIDENCE_OVERSIZE, "evidence json exceeds limit")

    unknown = set(attestation) - ALLOWED_EVIDENCE_TOP_LEVEL
    # Unknown critical fields fail closed.
    if unknown:
        _reject(
            TeeReasonCode.EVIDENCE_UNKNOWN_FIELD,
            f"unknown evidence fields: {','.join(sorted(unknown))}",
        )
    for forbidden in ("jku", "x5u", "jwks_uri", "trust_bundle_url", "trust_bundle_path"):
        if forbidden in attestation and attestation.get(forbidden) not in (None, ""):
            _reject(TeeReasonCode.TRUST_LOCATOR_FORBIDDEN, f"forbidden locator {forbidden}")

    version = attestation.get("version")
    if version not in ALLOWED_VERSIONS:
        _reject(TeeReasonCode.EVIDENCE_UNKNOWN_VERSION, f"version={version!r}")
    provider_raw = attestation.get("provider")
    if not isinstance(provider_raw, str) or provider_raw not in ALLOWED_PROVIDERS:
        _reject(TeeReasonCode.EVIDENCE_UNKNOWN_PROVIDER, f"provider={provider_raw!r}")
    evidence_type = attestation.get("evidence_type")
    if not isinstance(evidence_type, str) or evidence_type not in ALLOWED_EVIDENCE_TYPES:
        _reject(TeeReasonCode.EVIDENCE_UNKNOWN_TYPE, f"evidence_type={evidence_type!r}")

    tdx = attestation.get("tdx_quote_b64")
    gpu = attestation.get("gpu_eat_jwt")
    if tdx is None or gpu is None:
        _reject(TeeReasonCode.COMPONENT_MISSING, "both tdx_quote_b64 and gpu_eat_jwt required")
    if not isinstance(tdx, str) or not isinstance(gpu, str):
        _reject(TeeReasonCode.EVIDENCE_WRONG_TYPE, "tdx/gpu must be strings")
    tdx = tdx.strip()
    gpu = gpu.strip()
    if not tdx or not gpu:
        _reject(TeeReasonCode.COMPONENT_MISSING, "empty tdx or gpu component")
    if len(tdx) > max_quote_b64_chars:
        _reject(TeeReasonCode.EVIDENCE_OVERSIZE, "tdx_quote_b64 oversize")
    if len(gpu) > max_jwt_chars:
        _reject(TeeReasonCode.EVIDENCE_OVERSIZE, "gpu_eat_jwt oversize")

    # Encoding strictness for base64 and JWT segment count (crypto validated later).
    _require_b64(tdx, field="tdx_quote_b64")
    _require_jwt_shape(gpu)

    return ParsedEvidence(
        version="1",
        provider=TeeProviderKind(provider_raw),
        evidence_type=evidence_type,
        tdx_quote_b64=tdx,
        gpu_eat_jwt=gpu,
        raw_digest=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _require_b64(value: str, *, field: str) -> None:
    if not (_B64_STD_RE.fullmatch(value) or _B64_URLSAFE_RE.fullmatch(value)):
        _reject(TeeReasonCode.ENCODING_INVALID, f"{field} is not base64")
    try:
        # Validate decodes without depending on exact padding style.
        pad = "=" * ((4 - len(value) % 4) % 4)
        if _B64_URLSAFE_RE.fullmatch(value) and not _B64_STD_RE.fullmatch(value):
            base64.urlsafe_b64decode(value + pad)
        else:
            base64.b64decode(value + pad, validate=True)
    except (binascii.Error, ValueError) as exc:
        _reject(TeeReasonCode.ENCODING_INVALID, f"{field} decode failed: {exc}")


def _require_jwt_shape(token: str) -> None:
    parts = token.split(".")
    if len(parts) != 3:
        _reject(TeeReasonCode.ENCODING_INVALID, "jwt must have exactly 3 segments")
    for idx, part in enumerate(parts[:2]):
        if not part or not _B64_URLSAFE_RE.fullmatch(part):
            _reject(TeeReasonCode.ENCODING_INVALID, f"jwt segment {idx} invalid base64url")
    # Signature segment may be empty only for alg=none attacks (still shape-ok for later check).
    if parts[2] and not _B64_URLSAFE_RE.fullmatch(parts[2]):
        _reject(TeeReasonCode.ENCODING_INVALID, "jwt signature segment invalid")


def decode_b64(value: str) -> bytes:
    pad = "=" * ((4 - len(value) % 4) % 4)
    if "-" in value or "_" in value:
        return base64.urlsafe_b64decode(value + pad)
    return base64.b64decode(value + pad, validate=False)


def parse_or_fail(attestation: Any, **limits: Any):
    try:
        return parse_attestation_mapping(attestation, **limits)
    except EvidenceParseError as exc:
        return fail_decision(reason=exc.reason, detail=exc.detail)


def expected_measurements_complete(measurements: Mapping[str, Any]) -> bool:
    if not isinstance(measurements, Mapping):
        return False
    for key in DEFAULT_MEASUREMENT_KEYS:
        value = measurements.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"^[0-9a-f]{64}$", value.lower()):
            return False
    return True
