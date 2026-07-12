"""Local cryptographic fixtures: TDX-quote envelope + GPU EAT JWT.

These prove verifier logic only. Outcomes must be classification
``LOCAL-FIXTURE PASS`` — never a real provider attestation.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

LOCAL_TDX_PURPOSE = "prism.local.tdx.quote"
LOCAL_GPU_PURPOSE = "prism.local.gpu.eat"
DEFAULT_MEASUREMENTS = {
    "mrtd": "11" * 32,
    "rtmr0": "22" * 32,
    "rtmr1": "33" * 32,
    "rtmr2": "44" * 32,
    "rtmr3": "55" * 32,
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64std(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def pem_public_key(public_key: rsa.RSAPublicKey | ec.EllipticCurvePublicKey) -> str:
    return public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("utf-8")


def fingerprint_pem(pem: str) -> str:
    return hashlib.sha256(pem.encode("utf-8")).hexdigest()[:32]


@dataclass
class LocalFixtureAuthority:
    """Ephemeral CA + GPU signing key for offline fixture generation."""

    ca_key: rsa.RSAPrivateKey
    ca_cert: x509.Certificate
    leaf_key: rsa.RSAPrivateKey
    leaf_cert: x509.Certificate
    gpu_key: ec.EllipticCurvePrivateKey
    gpu_kid: str = "local-gpu-1"
    issuer: str = "prism-local-fixture"

    @classmethod
    def generate(
        cls, *, issuer: str = "prism-local-fixture", now: datetime | None = None
    ) -> LocalFixtureAuthority:
        now = now or datetime.now(UTC)
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Prism Local TDX Root")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Prism Local TDX Leaf")])
        leaf_cert = (
            x509.CertificateBuilder()
            .subject_name(leaf_name)
            .issuer_name(ca_name)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(hours=1))
            .not_valid_after(now + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        gpu_key = ec.generate_private_key(ec.SECP256R1())
        return cls(
            ca_key=ca_key,
            ca_cert=ca_cert,
            leaf_key=leaf_key,
            leaf_cert=leaf_cert,
            gpu_key=gpu_key,
            issuer=issuer,
        )

    def ca_pem(self) -> str:
        return self.ca_cert.public_bytes(Encoding.PEM).decode("utf-8")

    def gpu_public_pem(self) -> str:
        return pem_public_key(self.gpu_key.public_key())

    def build_quote_body(
        self,
        *,
        nonce: str,
        work_unit_id: str,
        submission_id: str,
        image_digest: str,
        workload_id: str,
        workload_version: str,
        challenge_slug: str,
        manifest_sha256: str,
        worker_pubkey: str,
        provider: str = "local_fixture",
        measurements: Mapping[str, str] | None = None,
        debug: bool = False,
        tcb_status: str = "up_to_date",
        session_id: str,
        iat: datetime,
        nbf: datetime,
        exp: datetime,
        purpose: str = LOCAL_TDX_PURPOSE,
        issuer: str | None = None,
        audience: str = "prism.tee.verify",
    ) -> dict[str, Any]:
        return {
            "format": "prism.local.tdx.v1",
            "provider": provider,
            "issuer": issuer or self.issuer,
            "audience": audience,
            "purpose": purpose,
            "nonce": nonce,
            "work_unit_id": work_unit_id,
            "submission_id": submission_id,
            "image_digest": image_digest,
            "workload_id": workload_id,
            "workload_version": workload_version,
            "challenge_slug": challenge_slug,
            "manifest_sha256": manifest_sha256,
            "worker_pubkey": worker_pubkey,
            "measurements": dict(measurements or DEFAULT_MEASUREMENTS),
            "debug": debug,
            "tcb_status": tcb_status,
            "session_id": session_id,
            "iat": int(iat.timestamp()),
            "nbf": int(nbf.timestamp()),
            "exp": int(exp.timestamp()),
        }

    def sign_tdx_quote(self, body: Mapping[str, Any]) -> str:
        body_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = self.leaf_key.sign(body_bytes, padding.PKCS1v15(), hashes.SHA256())
        envelope = {
            "format": "prism.local.tdx.v1",
            "body": dict(body),
            "signature_b64": _b64std(signature),
            "certificate_chain_pem": [
                self.leaf_cert.public_bytes(Encoding.PEM).decode("utf-8"),
                self.ca_cert.public_bytes(Encoding.PEM).decode("utf-8"),
            ],
        }
        return _b64std(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def sign_gpu_eat(
        self,
        *,
        claims: Mapping[str, Any],
        alg: str = "ES256",
        kid: str | None = None,
        extra_headers: Mapping[str, Any] | None = None,
    ) -> str:
        header: dict[str, Any] = {"alg": alg, "typ": "JWT", "kid": kid or self.gpu_kid}
        if extra_headers:
            header.update(dict(extra_headers))
        header_b64 = _b64url(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        payload_b64 = _b64url(
            json.dumps(dict(claims), sort_keys=True, separators=(",", ":")).encode()
        )
        signing_input = f"{header_b64}.{payload_b64}".encode()
        if alg == "none":
            return f"{header_b64}.{payload_b64}."
        if alg == "ES256":
            signature = self.gpu_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
            # Convert DER -> raw R||S for JWT ES256.
            from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

            r, s = decode_dss_signature(signature)
            sig_bytes = r.to_bytes(32, "big") + s.to_bytes(32, "big")
            return f"{header_b64}.{payload_b64}.{_b64url(sig_bytes)}"
        if alg == "HS256":
            import hmac

            secret = b"attacker-symmetric-confusion"
            sig = hmac.new(secret, signing_input, hashlib.sha256).digest()
            return f"{header_b64}.{payload_b64}.{_b64url(sig)}"
        raise ValueError(f"unsupported fixture alg {alg}")

    def build_attestation(
        self,
        *,
        nonce: str,
        work_unit_id: str,
        submission_id: str,
        image_digest: str,
        workload_id: str,
        workload_version: str,
        challenge_slug: str,
        manifest_sha256: str,
        worker_pubkey: str,
        session_id: str | None = None,
        now: datetime | None = None,
        provider: str = "local_fixture",
        measurements: Mapping[str, str] | None = None,
        gpu_model: str = "H100",
        gpu_vendor: str = "nvidia",
        gpu_device_id: str = "gpu-0",
        gpu_count: int = 1,
        security_mode: str = "cc",
        audience: str = "prism.tee.verify",
        purpose: str = "execution_attestation",
        max_age_window: int = 600,
        debug: bool = False,
        tcb_status: str = "up_to_date",
    ) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        session = session_id or hashlib.sha256(f"{work_unit_id}:{nonce}".encode()).hexdigest()
        iat = now
        nbf = now - timedelta(seconds=5)
        exp = now + timedelta(seconds=max_age_window)
        quote_body = self.build_quote_body(
            nonce=nonce,
            work_unit_id=work_unit_id,
            submission_id=submission_id,
            image_digest=image_digest,
            workload_id=workload_id,
            workload_version=workload_version,
            challenge_slug=challenge_slug,
            manifest_sha256=manifest_sha256,
            worker_pubkey=worker_pubkey,
            provider=provider,
            measurements=measurements,
            debug=debug,
            tcb_status=tcb_status,
            session_id=session,
            iat=iat,
            nbf=nbf,
            exp=exp,
            audience=audience,
        )
        tdx_b64 = self.sign_tdx_quote(quote_body)
        gpu_claims = {
            "iss": self.issuer,
            "aud": audience,
            "purpose": purpose,
            "provider": provider,
            "nonce": nonce,
            "work_unit_id": work_unit_id,
            "submission_id": submission_id,
            "image_digest": image_digest,
            "workload_id": workload_id,
            "workload_version": workload_version,
            "challenge_slug": challenge_slug,
            "manifest_sha256": manifest_sha256,
            "worker_pubkey": worker_pubkey,
            "session_id": session,
            "gpu_vendor": gpu_vendor,
            "gpu_model": gpu_model,
            "gpu_device_id": gpu_device_id,
            "gpu_count": gpu_count,
            "security_mode": security_mode,
            "driver_policy": "allowed",
            "iat": int(iat.timestamp()),
            "nbf": int(nbf.timestamp()),
            "exp": int(exp.timestamp()),
            "measurements": dict(measurements or DEFAULT_MEASUREMENTS),
        }
        jwt = self.sign_gpu_eat(claims=gpu_claims)
        return {
            "version": 1,
            "provider": provider,
            "evidence_type": "prism.tee.v1",
            "tdx_quote_b64": tdx_b64,
            "gpu_eat_jwt": jwt,
        }
