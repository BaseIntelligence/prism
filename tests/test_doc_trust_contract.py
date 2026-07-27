"""Doc trust contract (todo 25): behavioural ceiling on effective_tier.

Docs claim no TEE and max effective tier 1. This test exercises the real
``effective_tier`` function — not a docs grep — and fails if any route returns > 1.
"""

from __future__ import annotations

import pytest

from prism_challenge.audit import effective_tier
from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature

PINNED = "sha256:" + ("ab" * 32)


def _proof(*, tier: int, image_digest: str | None = None) -> ExecutionProof:
    if tier >= 1 and image_digest is None:
        image_digest = PINNED
    attestation = None
    if tier >= 2:
        attestation = {"tdx_quote_b64": "opaque", "gpu_eat_jwt": "opaque"}
    return ExecutionProof(
        version=1,
        tier=tier,  # type: ignore[arg-type]
        manifest_sha256="c" * 64,
        image_digest=image_digest,
        provider=ProviderInfo(name="lium", pod_id="pod-doc-contract"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
        attestation=attestation,
    )


@pytest.mark.parametrize(
    "claimed,constation,pin",
    [
        (0, None, None),
        (0, True, PINNED),
        (1, None, PINNED),
        (1, False, PINNED),
        (1, True, PINNED),
        (1, True, "sha256:" + ("cd" * 32)),
        (2, True, PINNED),
        (2, False, PINNED),
        (3, True, PINNED),
        (99, True, PINNED),
    ],
)
def test_effective_tier_never_exceeds_one(
    claimed: int,
    constation: bool | None,
    pin: str | None,
) -> None:
    """Behavioural contract: docs may claim tier ceiling 1 only if code enforces it."""
    proof = _proof(tier=min(claimed, 2) if claimed >= 2 else claimed)
    # For claimed > 2, still construct a tier-2-shaped proof then override attribute if allowed.
    object.__setattr__(proof, "tier", claimed) if hasattr(proof, "__dict__") else None
    try:
        # Pydantic models may freeze tier; rebuild when needed.
        if int(proof.tier) != claimed:
            proof = ExecutionProof(
                version=1,
                tier=min(claimed, 2),  # type: ignore[arg-type]
                manifest_sha256="c" * 64,
                image_digest=PINNED,
                provider=ProviderInfo(name="lium", pod_id="pod-doc-contract"),
                worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
                attestation={"tdx_quote_b64": "x", "gpu_eat_jwt": "y"} if claimed >= 2 else None,
            )
            # effective_tier reads getattr tier — pass via model_copy if available
            if hasattr(proof, "model_copy"):
                proof = proof.model_copy(update={"tier": claimed})
    except Exception:
        proof = _proof(tier=2 if claimed >= 2 else claimed)

    got = effective_tier(
        proof,
        pinned_image_digest=pin,
        constation_ok_result=constation,
    )
    assert isinstance(got, int)
    assert got <= 1, (
        f"effective_tier returned {got} > 1 "
        f"(claimed={claimed}, constation={constation})"
    )
    assert got >= 0


def test_constation_true_claimed_1_is_at_most_one() -> None:
    proof = _proof(tier=1)
    assert effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=True) == 1


def test_claimed_tier_2_with_constation_still_zero_not_two() -> None:
    proof = _proof(tier=2)
    assert effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=True) == 0


def test_pin_match_alone_does_not_elevate() -> None:
    proof = _proof(tier=1)
    assert effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=None) == 0
    assert effective_tier(proof, pinned_image_digest=PINNED, constation_ok_result=False) == 0
