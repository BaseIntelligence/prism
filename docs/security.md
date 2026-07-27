# Security Model

PRISM evaluates untrusted miner code. It assumes submissions may be malicious and layers identity
verification, a static AST sandbox, **deterministic admission** (project shape, source similarity,
anti-cheat), a terminal duplicate policy, and a forced-init re-execution that makes the common cheats
inert rather than merely detected. The former LLM hard gate and master LLM gateway are **removed**.

## Identity and Authentication

BASE handles miner-facing uploads, verifying hotkey identity, signatures, timestamps, nonces, request
freshness, and challenge routing before forwarding the payload to `POST /internal/v1/bridge/submissions`;
PRISM trusts the verified hotkey header only on authenticated internal requests. Internal endpoints
require the shared BASE challenge token (`Authorization: Bearer <shared-token>`), read from
`PRISM_SHARED_TOKEN`, `CHALLENGE_SHARED_TOKEN`, or a secret file (prefer secret files in production).

## Static Sandbox

Before any GPU work, PRISM runs the static gates over both scripts, in order:

1. **AST hard-blocks** over `architecture.py` and `training.py`: no `os`, `sys`, `subprocess`,
   `socket`, network clients, `pickle`/`torch.load` of untrusted paths, `ctypes`, dynamic `importlib`,
   `eval`/`exec`/`compile`, attribute escapes (`__globals__`, `__reduce__`, `__class__` walking), or
   filesystem writes outside `artifacts_dir`.
2. **Forced-seed parameter cap**: `build_model(ctx)` is instantiated under the forced seed in a bounded
   child process and rejected if it exceeds the stage param cap (124M explore / 350M promote; realized first-forward shapes).
3. **Multi-GPU static contract**: the training script must use the distributed primitives and a rank-0
   write guard; a `gpu_count > 8` or multi-node request is rejected.

A rejection at any static gate is terminal before similarity admission and before any GPU work.

## Deterministic Admission

After static gates pass, PRISM applies challenge-owned deterministic checks only:

- **Project shape** — two-script contract resolution and fingerprints (no single-module re-export).
- **Source similarity** — exact-source-hash duplicates are rejected; borderline (quarantine-band)
  similarity is also a **terminal reject**. There is no held-for-review or LLM quarantine path.
- **Anti-cheat / scoring gates** — forced-init invariants and score anomaly multipliers (see below).

Legacy env keys or settings related to LLM review, gateway URL/token, architecture auto-report, or
component-agent hold policies fail closed at configuration load. Unknown residual knobs are rejected
rather than silently ignored when they map to removed surfaces.

## Forced-Init Re-Execution (Anti-Cheat Core)

The challenge re-executes the miner's `training.py` under a **forced random init** with a fixed,
challenge-controlled seed and deterministic flags set **before** any miner code runs, feeds it fresh
single-pass batches from the locked train split, and records the online loss itself. This neutralizes
the three cheat classes:

- **No pretrained weights** — forced random init makes smuggled weights inert; an impossibly low step-0
  loss is flagged and zeroes the score; `network=none` and the sandbox block IO/network/deserialization
  escapes.
- **No metric manipulation** — the metric comes from the captured loss stream, so any miner-reported
  number and any miner-written manifest are ignored. The fixed seed and data order make runs
  reproducible within tolerance.
- **No memorization** — the `val`/`test` splits are secret and **never exposed** to the miner script; an
  excessive train-vs-held-out gap penalizes the score.

## External Result Envelope (Worker Plane)

When the worker plane is enabled, Prism ingests reconciled external evaluation results **only** as the
Base SDK `ExternalResultEnvelope` (api_version, assignment/challenge bindings, execution proof). Dual
or legacy reduced bodies fail closed with a 422 before scoring or persistence. Proof verification and
plausibility gates run before finalization; duplicate deliveries are idempotent and conflicts refuse
mutation of sealed scores.

## Worker-plane integrity: constation (tamper-evidence only)

### Threat model (binding)

- The miner **rents and controls** the Lium pod, including **root**.
- Integrity checks are **tamper-evidence**, not tamper-prevention. A determined root miner can
  defeat all six mechanisms below.
- Prism does **not** ship a TEE verifier and does **not** require TEE evidence to finalize scores.
- `effective_tier` **ceiling is 1**. Claimed tier ≥ 2 always collapses to 0. Opaque
  `tdx_quote_b64` / `gpu_eat_jwt` fields never imply tier 2 by presence alone.
- Lium TDX (or similar host quotes), if present in the wild, attest **executor infrastructure**,
  not the renter workload inside the pod.
- The Lium API image/template digest is **declared configuration**, not a runtime measurement of
  the container rootfs.

### Sole elevation and score gate

| Predicate | Role |
| --- | --- |
| `constation_ok(bundle)` | **Only** path that can grant `effective_tier == 1` |
| Self-reported `PRISM_IMAGE_DIGEST` / pin string match | Telemetry only; **never** elevates alone |
| Missing or failed constation bundle | **No score row written** (P1 fail-closed) |
| Operator break-glass | May admit **infra_fault** runs only; never `miner_fault`; score at tier 0 |

Historical label **IMAGE_PIN** meant "digest pin matched." Under P1 that label is **not** a soft
orthogonal badge: without a valid constation bundle there is no score to rank. See
[scoring.md](scoring.md).

`attestation_mode` on proofs is `miner_rent_image_pin_evidence_v1` (evidence under miner-rent pods).
Forbidden modes include any TEE-implying string (`tee`, `tdx`, `sev`, …).

### Six mechanisms (each with honest limitation)

1. **Nonce-bound attestation** — BASE per-run nonce; sidecar signs bound fields; defeats replay.
   *Limit:* live root with the build secret can still answer fresh challenges.
2. **Same-account corroboration** — Lium API declared digest must agree with sidecar report.
   *Limit:* **never independent**; miner supplies the API key and owns the pod. Agreement alone
   never elevates if another check fails.
3. **Continuous constation** — start / end / randomized polls; gap or drift fails the bundle.
   *Limit:* shrinks swap windows; does not seal the kernel against a twin process.
4. **BASE-built digest allowlist** — only BASE-produced digests for a registered commit score.
   *Limit:* proves build provenance, not that the pod still runs that digest after deploy.
5. **In-image self-measurement** — sidecar hashes sealed harness / rules / data-window at runtime.
   *Limit:* software on a miner-controlled host; root can patch the measurer.
6. **Build-time per-build secret** — unique BASE-injected secret required in signed responses.
   *Limit:* root on the pod can extract `/run/prism/attestation_hmac_key`.

Full recipe-side narrative: [prism-recipe security](../../prism-recipe/docs/security.md).

### Provider trust (operational, not crypto)

Operators still choose which GPU providers to enable (e.g. Lium). That is operational trust in
billing and reachability, **not** a cryptographic claim that the renter workload is honest.
Deploy smoke checks are reachability/infra only; always terminate pods.

**REAL-PROVIDER TEE** as a Prism product goal is **retired**. Historical lab tables may still show
`real_provider_tee=BLOCKED` as honesty history; do not implement a TEE production scoring gate.

## Locked Data, No Network

The train split is mounted **read-only** at `ctx.data_dir` and is the only data the miner script sees;
the `val`/`test` splits are secret. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is **no network** during training and the miner cannot download data,
tokenizers, or weights at runtime.

## Duplicate Review

An exact-source-hash duplicate is rejected, and a borderline-similarity quarantine is folded into a
terminal rejection at ingress (there is no operator hold-resolution surface; the v1-NAS component-review,
LLM review, and ownership machinery were decommissioned).

## Execution Isolation

PRISM never executes submitted code inside the API process without isolation. The scored run happens in
a broker-backed container that is non-root, has a read-only rootfs except `artifacts_dir`, uses
`network=none` and `no-new-privileges`, and is bounded by CPU, memory, PID, and wall-clock caps.
Host-side static instantiation and held-out scoring run in bounded child processes with
`weights_only=True` for any deserialization. Application code does **not** create ephemeral evaluator
containers; evaluation is the long-lived challenge runtime (or external worker-plane GPUs on trusted
providers when enabled).

On the **miner-rent Lium** path, isolation is weaker by design: the miner is root on their pod.
Constation is the evidence layer for that path, not a substitute for container isolation on BASE-owned
evaluators.

## ZIP Hardening

ZIP extraction rejects symlinks, path traversal, unsafe paths, unsupported file types, and excessive
file counts or bytes before code review begins.

## Weights And Chain Boundary

PRISM exposes `get_weights` for inventory/compatibility and pushes authenticated **raw** hotkey weights
to the BASE master. The master aggregates the final vector; **validators** call `set_weights` with
their own wallets. The challenge and master never write weights on-chain.

## Reference Studies

- **Supply-chain attacks** — Gu, Dolan-Gavitt, and Garg, 2017/2019 (*BadNets*): treat submitted code
  and artifacts as adversarial even when metrics look normal.
- **Untrusted deserialization** — pickle/`torch.load` RCE guidance: load host-side artifacts with
  `weights_only=True` from the challenge-recorded path only.
- **Dataset provenance** — Penedo et al., 2024 (*The FineWeb Datasets*): pin the revision and shard
  hashes; keep held-out splits secret.

## Operational Guidance

- Use real secret files in production, not inline tokens.
- Keep public submissions disabled when PRISM is deployed only behind BASE.
- Keep the eval container on `network=none` and the rootfs read-only except `artifacts_dir`.
- Do **not** configure LLM gateway URL/token fields; those surfaces are gone and residual knobs fail closed.
- Require a full constation bundle for Lium worker-plane scores; do not treat pin string match as tier 1.
- Do not enable any removed TEE production scoring path.
- Monitor rejected, failed, and completed submissions separately (legacy held is not a live path).
- Attribute rejects with `miner_fault:*` vs `infra_fault:*`; break-glass only for infra with operator audit.
