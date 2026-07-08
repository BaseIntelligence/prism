# Security Model

PRISM evaluates untrusted miner code. It assumes submissions may be malicious and layers identity
verification, a static AST sandbox, an LLM hard gate, a duplicate check, and a forced-init re-execution
that makes the common cheats inert rather than merely detected.

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
   child process and rejected if it exceeds the 150M cap (realized first-forward shapes).
3. **Multi-GPU static contract**: the training script must use the distributed primitives and a rank-0
   write guard; a `gpu_count > 8` or multi-node request is rejected.

A rejection at any static gate is terminal and skips the LLM review entirely.

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

## LLM Hard Gate

After the static gates pass, a strong LLM reviews both scripts as a **hard gate**, routing **only**
through the BASE master LLM gateway at `{root}/llm/v1` with a scoped token (the `X-Gateway-Token` header)
from the Docker secret at `/run/secrets/base_gateway_token`. PRISM holds no raw provider key and pins no
model: the gateway selects the provider and model server-side (a `master.yaml` choice) and injects them
per request. The gate checks architecture-to-training coherence, cheating and obfuscation (smuggled
weights, hidden network, dead/no-op loops, metric gaming), and dangerous operations the static sandbox
might miss.

The verdict is structured JSON. A `reject` is terminal: the pipeline stops **before any GPU work** and
the submission ends `rejected`. A transient error or ambiguous result fails closed to a held quarantine.
The gate is on by default; only a configuration-disabled gate is skipped.

## Locked Data, No Network

The train split is mounted **read-only** at `ctx.data_dir` and is the only data the miner script sees;
the `val`/`test` splits are secret. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is **no network** during training and the miner cannot download data,
tokenizers, or weights at runtime.

## Duplicate Review

An exact-source-hash duplicate is rejected, and a borderline-similarity quarantine is folded into a
terminal rejection at ingress (there is no operator hold-resolution surface; the v1-NAS component-review
and ownership machinery was decommissioned).

## Execution Isolation

PRISM never executes submitted code inside the API or worker process. The scored run happens in a
broker-backed container that is non-root, has a read-only rootfs except `artifacts_dir`, uses
`network=none` and `no-new-privileges`, and is bounded by CPU, memory, PID, and wall-clock caps.
Host-side static instantiation and held-out scoring run in bounded child processes with
`weights_only=True` for any deserialization.

## ZIP Hardening

ZIP extraction rejects symlinks, path traversal, unsafe paths, unsupported file types, and excessive
file counts or bytes before code review begins.

## Dry-Run Weights

Weights are normalized per hotkey from the bits-per-byte `final_score` and exposed only via
`get_weights`. They are always **dry-run** and are never written on-chain.

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
- Keep the LLM hard gate enabled for production.
- Monitor rejected, held, failed, and completed submissions separately.
