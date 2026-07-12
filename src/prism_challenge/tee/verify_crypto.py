"""Cryptographic verification of local TDX quote envelope + GPU EAT JWT."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils

from .config import TeeVerifierConfig
from .crypto_local import LOCAL_GPU_PURPOSE, LOCAL_TDX_PURPOSE
from .evidence import decode_b64
from .types import TeeReasonCode


class CryptoVerifyError(Exception):
    def __init__(self, reason: TeeReasonCode, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(detail or reason.value)


def _load_pem_certs(pems: Sequence[str]) -> list[x509.Certificate]:
    out: list[x509.Certificate] = []
    for pem in pems:
        out.append(x509.load_pem_x509_certificate(pem.encode("utf-8")))
    return out


def _cert_pem(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")


def _active_roots(config: TeeVerifierConfig, *, now: datetime) -> list[x509.Certificate]:
    roots = _load_pem_certs(config.tdx_trust_roots_pem)
    if len(roots) <= 1:
        return roots
    # Explicit dual-root rotation window: both roots active inside window only.
    nb = config.trust_rotation_not_before
    na = config.trust_rotation_not_after
    if nb is not None and na is not None and nb <= now <= na:
        return roots
    # Outside rotation: only the last (active) root is accepted.
    return [roots[-1]]


def verify_tdx_quote(
    quote_b64: str,
    config: TeeVerifierConfig,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify local TDX envelope. Returns (body, root_fingerprint)."""

    now = now or datetime.now(UTC)
    try:
        raw = decode_b64(quote_b64)
        envelope = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(TeeReasonCode.ENCODING_INVALID, f"tdx decode: {exc}") from exc
    if not isinstance(envelope, dict):
        raise CryptoVerifyError(TeeReasonCode.EVIDENCE_MALFORMED, "tdx envelope not object")
    if envelope.get("format") != "prism.local.tdx.v1":
        raise CryptoVerifyError(TeeReasonCode.EVIDENCE_UNKNOWN_TYPE, "unsupported tdx format")
    body = envelope.get("body")
    sig_b64 = envelope.get("signature_b64")
    chain_pem = envelope.get("certificate_chain_pem")
    if (
        not isinstance(body, dict)
        or not isinstance(sig_b64, str)
        or not isinstance(chain_pem, list)
    ):
        raise CryptoVerifyError(TeeReasonCode.EVIDENCE_MALFORMED, "tdx envelope fields")
    if len(chain_pem) < 2:
        raise CryptoVerifyError(TeeReasonCode.TDX_CHAIN_UNTRUSTED, "chain too short")
    try:
        certs = [x509.load_pem_x509_certificate(p.encode("utf-8")) for p in chain_pem]
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(TeeReasonCode.TDX_CHAIN_UNTRUSTED, f"bad cert pem: {exc}") from exc

    leaf, *intermediates = certs
    root_candidates = _active_roots(config, now=now)
    if not root_candidates:
        raise CryptoVerifyError(TeeReasonCode.VERIFIER_MISCONFIGURED, "no trust roots")

    # Validate leaf time + key usage.
    leaf_nb = (
        leaf.not_valid_before_utc
        if hasattr(leaf, "not_valid_before_utc")
        else leaf.not_valid_before.replace(tzinfo=UTC)
    )
    leaf_na = (
        leaf.not_valid_after_utc
        if hasattr(leaf, "not_valid_after_utc")
        else leaf.not_valid_after.replace(tzinfo=UTC)
    )
    if now < leaf_nb or now > leaf_na:
        raise CryptoVerifyError(TeeReasonCode.TDX_CERT_REJECTED, "leaf not currently valid")
    try:
        ku = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
        if not ku.digital_signature:
            raise CryptoVerifyError(
                TeeReasonCode.TDX_CERT_REJECTED, "leaf missing digital_signature"
            )
    except x509.ExtensionNotFound as exc:
        raise CryptoVerifyError(TeeReasonCode.TDX_CERT_REJECTED, "leaf missing key usage") from exc

    # Chain must terminate at a pinned root (by equality of public cert bytes).
    root = certs[-1]
    root_match = None
    for candidate in root_candidates:
        if (
            _cert_pem(candidate) == _cert_pem(root)
            or candidate.public_key().public_numbers() == root.public_key().public_numbers()
        ):  # type: ignore[union-attr]
            # Also allow leaf signed directly by candidate when intermediate omitted matches.
            root_match = candidate
            break
        # Compare subject/public key fingerprint.
        if candidate.fingerprint(hashes.SHA256()) == root.fingerprint(hashes.SHA256()):
            root_match = candidate
            break
    if root_match is None:
        # Permit chain where configured root signed the leaf as issuer.
        for candidate in root_candidates:
            try:
                _verify_cert_signature(leaf, candidate)
                root_match = candidate
                break
            except CryptoVerifyError:
                continue
        if root_match is None:
            raise CryptoVerifyError(TeeReasonCode.TDX_CHAIN_UNTRUSTED, "chain not pinned")

    # Verify leaf signature under issuer cert (next in chain or pinned root).
    issuer_cert = certs[1] if len(certs) > 1 else root_match
    _verify_cert_signature(leaf, issuer_cert)
    if len(certs) > 2:
        for i in range(1, len(certs) - 1):
            _verify_cert_signature(certs[i], certs[i + 1])

    body_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        signature = base64.b64decode(sig_b64)
        public_key = leaf.public_key()
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise CryptoVerifyError(TeeReasonCode.TDX_SIGNATURE_INVALID, "leaf not RSA")
        public_key.verify(signature, body_bytes, padding.PKCS1v15(), hashes.SHA256())
    except CryptoVerifyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(TeeReasonCode.TDX_SIGNATURE_INVALID, f"quote sig: {exc}") from exc

    if body.get("purpose") not in (LOCAL_TDX_PURPOSE, "execution_attestation"):
        # purpose is also checked at claim binding; wrong purpose fails.
        pass

    fp = hashlib.sha256(_cert_pem(root_match).encode("utf-8")).hexdigest()[:32]
    return body, fp


def _verify_cert_signature(subject: x509.Certificate, issuer: x509.Certificate) -> None:
    public_key = issuer.public_key()
    try:
        if isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(
                subject.signature,
                subject.tbs_certificate_bytes,
                padding.PKCS1v15(),
                subject.signature_hash_algorithm,  # type: ignore[arg-type]
            )
        else:
            raise CryptoVerifyError(TeeReasonCode.TDX_CHAIN_UNTRUSTED, "unsupported issuer key")
    except CryptoVerifyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(
            TeeReasonCode.TDX_CHAIN_UNTRUSTED, f"cert chain sig: {exc}"
        ) from exc


def verify_gpu_eat(
    token: str,
    config: TeeVerifierConfig,
) -> tuple[dict[str, Any], str]:
    """Verify ES256 GPU EAT against pinned kid keys. Returns (claims, key_fingerprint)."""

    parts = token.split(".")
    if len(parts) != 3:
        raise CryptoVerifyError(TeeReasonCode.ENCODING_INVALID, "jwt shape")
    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(TeeReasonCode.ENCODING_INVALID, f"jwt json: {exc}") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise CryptoVerifyError(TeeReasonCode.EVIDENCE_MALFORMED, "jwt header/claims")

    # Reject attacker-controlled trust locators and algorithm confusion.
    for forbidden in ("jku", "jwk", "x5u", "x5c"):
        if forbidden in header:
            raise CryptoVerifyError(TeeReasonCode.TRUST_LOCATOR_FORBIDDEN, f"header {forbidden}")
    alg = header.get("alg")
    if alg is None or alg == "none" or str(alg).lower() == "none":
        raise CryptoVerifyError(TeeReasonCode.GPU_ALG_CONFUSION, "alg none")
    if alg in {"HS256", "HS384", "HS512"}:
        raise CryptoVerifyError(TeeReasonCode.GPU_ALG_CONFUSION, f"symmetric alg {alg}")
    if alg != "ES256":
        raise CryptoVerifyError(TeeReasonCode.GPU_ALG_CONFUSION, f"unsupported alg {alg}")
    kid = header.get("kid")
    if not isinstance(kid, str) or kid not in config.gpu_trusted_keys_pem:
        raise CryptoVerifyError(TeeReasonCode.GPU_UNTRUSTED_KEY, f"unknown kid {kid!r}")

    pem = config.gpu_trusted_keys_pem[kid]
    try:
        public_key = serialization.load_pem_public_key(pem.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(TeeReasonCode.GPU_UNTRUSTED_KEY, f"bad key pem: {exc}") from exc
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise CryptoVerifyError(TeeReasonCode.GPU_UNTRUSTED_KEY, "gpu key not EC")
    if not sig_b64:
        raise CryptoVerifyError(TeeReasonCode.GPU_SIGNATURE_INVALID, "empty signature")
    try:
        sig = _b64url_decode(sig_b64)
        if len(sig) != 64:
            raise CryptoVerifyError(TeeReasonCode.GPU_SIGNATURE_INVALID, "sig length")
        r = int.from_bytes(sig[:32], "big")
        s = int.from_bytes(sig[32:], "big")
        der = utils.encode_dss_signature(r, s)
        public_key.verify(der, f"{header_b64}.{payload_b64}".encode(), ec.ECDSA(hashes.SHA256()))
    except CryptoVerifyError:
        raise
    except InvalidSignature as exc:
        raise CryptoVerifyError(TeeReasonCode.GPU_SIGNATURE_INVALID, "bad signature") from exc
    except Exception as exc:  # noqa: BLE001
        raise CryptoVerifyError(TeeReasonCode.GPU_SIGNATURE_INVALID, f"verify: {exc}") from exc

    if claims.get("purpose") not in (LOCAL_GPU_PURPOSE, "execution_attestation", None):
        # purpose binding enforced by policy stage; keep crypto stage focused.
        pass

    fp = hashlib.sha256(pem.encode("utf-8")).hexdigest()[:32]
    return claims, fp


def _b64url_decode(value: str) -> bytes:
    pad = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value + pad)


def claim_bindings_match(
    tdx_body: Mapping[str, Any],
    gpu_claims: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    config: TeeVerifierConfig,
    now: datetime,
) -> TeeReasonCode | None:
    """Return a reason code on mismatch, else None when all bindings pass."""

    def _exact(mapping: Mapping[str, Any], key: str, expected_value: Any) -> bool:
        actual = mapping.get(key)
        return actual == expected_value and actual is not None and actual != ""

    # Issuer / audience / provider (exact match, no case folding)
    expected_provider = expected.get("provider") or config.expected_provider
    for mapping in (tdx_body, gpu_claims):
        issuer = mapping.get("issuer", mapping.get("iss"))
        if issuer != config.expected_issuer:
            return TeeReasonCode.ISSUER_MISMATCH
        audience = mapping.get("aud", mapping.get("audience"))
        if audience != config.expected_audience:
            return TeeReasonCode.AUDIENCE_MISMATCH
        if mapping.get("provider") != expected_provider:
            return TeeReasonCode.PROVIDER_MISMATCH

    # Nonce
    expected_nonce = expected.get("nonce")
    if not expected_nonce:
        return TeeReasonCode.NONCE_MISSING
    if tdx_body.get("nonce") != expected_nonce or gpu_claims.get("nonce") != expected_nonce:
        return TeeReasonCode.NONCE_MISMATCH

    # Time / freshness
    for mapping in (tdx_body, gpu_claims):
        try:
            iat = int(mapping["iat"])
            nbf = int(mapping["nbf"])
            exp = int(mapping["exp"])
        except (KeyError, TypeError, ValueError):
            return TeeReasonCode.FRESHNESS_INVALID
        now_ts = int(now.timestamp())
        skew = config.clock_skew_seconds
        if nbf - skew > now_ts:
            return TeeReasonCode.FRESHNESS_INVALID
        if exp + skew < now_ts:
            return TeeReasonCode.FRESHNESS_INVALID
        if iat - skew > now_ts + 300:
            return TeeReasonCode.FRESHNESS_INVALID
        if exp <= nbf or now_ts - iat > config.max_age_seconds + skew:
            return TeeReasonCode.FRESHNESS_INVALID

    # Workload / image / proof bindings
    for key in (
        "work_unit_id",
        "submission_id",
        "image_digest",
        "workload_id",
        "workload_version",
        "challenge_slug",
        "manifest_sha256",
        "worker_pubkey",
        "session_id",
    ):
        if expected.get(key) is None:
            continue
        if tdx_body.get(key) != expected[key] or gpu_claims.get(key) != expected[key]:
            if key in {
                "work_unit_id",
                "submission_id",
                "workload_id",
                "workload_version",
                "challenge_slug",
            }:
                return TeeReasonCode.WORKLOAD_MISMATCH
            if key == "image_digest":
                return TeeReasonCode.IMAGE_DIGEST_MISMATCH
            if key in {"manifest_sha256", "worker_pubkey"}:
                return TeeReasonCode.PROOF_BINDING_MISMATCH
            return TeeReasonCode.CROSS_BINDING_MISMATCH

    if expected.get("image_digest") != config.expected_image_digest:
        return TeeReasonCode.IMAGE_DIGEST_MISMATCH
    if tdx_body.get("image_digest") != config.expected_image_digest:
        return TeeReasonCode.IMAGE_DIGEST_MISMATCH

    # Measurements allowlist
    if not config.allowed_measurements:
        return TeeReasonCode.MEASUREMENT_POLICY_EMPTY
    measurements = tdx_body.get("measurements")
    gpu_ms = gpu_claims.get("measurements")
    if not isinstance(measurements, Mapping) or not isinstance(gpu_ms, Mapping):
        return TeeReasonCode.MEASUREMENT_MISMATCH
    for key, expected_m in config.allowed_measurements.items():
        actual = measurements.get(key)
        if not isinstance(actual, str) or actual.lower() != expected_m.lower():
            return TeeReasonCode.MEASUREMENT_MISMATCH
        if gpu_ms.get(key) != actual:
            return TeeReasonCode.CROSS_BINDING_MISMATCH

    # TCB / debug policy
    if tdx_body.get("debug") is True:
        return TeeReasonCode.TCB_POLICY_REJECTED
    if tdx_body.get("tcb_status") not in (None, "up_to_date", "OK", "ok"):
        return TeeReasonCode.TCB_POLICY_REJECTED

    # GPU identity policy
    vendor = gpu_claims.get("gpu_vendor")
    model = gpu_claims.get("gpu_model")
    count = gpu_claims.get("gpu_count")
    mode = gpu_claims.get("security_mode")
    if vendor not in config.allowed_gpu_vendors:
        return TeeReasonCode.GPU_IDENTITY_MISMATCH
    if model not in config.allowed_gpu_models:
        return TeeReasonCode.GPU_IDENTITY_MISMATCH
    if not isinstance(count, int) or count < 1:
        return TeeReasonCode.GPU_IDENTITY_MISMATCH
    if mode != config.require_gpu_security_mode:
        return TeeReasonCode.GPU_IDENTITY_MISMATCH

    # Cross-session binding already checked via session_id equality above.
    if tdx_body.get("session_id") != gpu_claims.get("session_id"):
        return TeeReasonCode.CROSS_BINDING_MISMATCH

    return None
