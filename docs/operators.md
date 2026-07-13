# Operator Guide

Local validation and production configuration for running PRISM as a BASE challenge under
**Docker Compose** (long-lived combined challenge service). Swarm is not a supported install path.

## Installation

```bash
git clone https://github.com/BaseIntelligence/prism.git
cd prism
uv sync --frozen --extra dev
```

PRISM depends on the immutable Base public wheel **v3.1.2**:

```text
https://github.com/BaseIntelligence/base/releases/download/v3.1.2/base-3.1.2-py3-none-any.whl
#sha256=3a61c2d3a343ed6de55e80215486e3de0c9639276443d08f2ed316bc807f2ff0
```

(see `pyproject.toml`). There is no LLM gateway dependency in this pin.

## Local Validation

```bash
.venv/bin/ruff check src
.venv/bin/mypy src/prism_challenge
.venv/bin/python -m pytest tests -q
```

GPU re-execution, HuggingFace publication, and external provider calls are mocked in tests. The LLM
gateway is not part of the test or deploy path.

## Required Runtime Configuration

```bash
PRISM_DATABASE_URL=sqlite+aiosqlite:////data/prism.sqlite3
PRISM_SHARED_TOKEN_FILE=/run/secrets/base/challenge_token
PRISM_EXECUTION_BACKEND=base_gpu
```

The shared token must match the token configured in the BASE master for this challenge. Prefer secret
files over inline tokens.

## Combined Mode (Compose)

Preferred production layout is the BASE master Compose project with one long-lived PRISM service in
**combined mode**: the API process drains the eval queue in-process (same worker/DB as the HTTP app).
Do not deploy Swarm stacks or launch ephemeral evaluator jobs from application code.

Typical challenge container settings for combined mode:

```bash
# Enable API + in-process queue drain in one long-lived service
# (exact env name follows PrismSettings; set in the Compose challenge service)
PRISM_COMBINED_MODE=true
```

Digest-pin the PRISM image via the master Compose install / watcher path (`PRISM_IMAGE_REPOSITORY` +
`PRISM_IMAGE_DIGEST`). Mutable `latest` Swarm service mutation is unsupported.

## Docker Broker Configuration

When evaluation uses the BASE Docker broker with the CI-published evaluator image:

```bash
PRISM_DOCKER_ENABLED=true
PRISM_DOCKER_BACKEND=broker
PRISM_DOCKER_BROKER_URL=http://base-docker-broker:8082
PRISM_DOCKER_BROKER_TOKEN_FILE=/run/secrets/base/challenge_token
PRISM_BASE_EVAL_IMAGE=ghcr.io/baseintelligence/prism-evaluator@sha256:<pinned>
PRISM_BASE_EVAL_GPU_COUNT=1
PRISM_DOCKER_NETWORK=none
```

The scored run is single-node (`torchrun --standalone --nnodes=1 --nproc-per-node=1`). The evaluator
image must ship `sentencepiece` and an offline tiktoken gpt2 cache so reference tokenizers load with no
network. Prefer digest-pinned references over floating tags.

## Locked FineWeb-Edu Data Plane

The broker bind-mounts the locked FineWeb-Edu data read-only into the eval container (`network=none`):

```bash
PRISM_BASE_EVAL_DATA_DIR=/data/fineweb-edu/train       # miner-visible, read-only
PRISM_BASE_EVAL_VAL_DATA_DIR=/data/fineweb-edu/val     # secret; scorer-only, never mounted into eval
PRISM_BASE_EVAL_REFERENCE_TOKENIZER_DIR=/opt/reference-tokenizers
```

`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set inside the eval container. The `val`/`test`
splits are secret and must never be exposed to a miner script.

## Compute Budget

Compute-normalized scoring; wall-clock is only a layered safety cap:

```bash
PRISM_BASE_EVAL_BUDGET_SECONDS=1200            # graceful stop; score the partial stream
PRISM_BASE_EVAL_WATCHDOG_GRACE_SECONDS=120     # hard watchdog above the graceful budget
PRISM_BASE_EVAL_TIMEOUT_SECONDS=1800           # outer docker/broker backstop
PRISM_BASE_EVAL_ARTIFACTS_QUOTA_BYTES=2147483648
```

## Deterministic Admission (No LLM Gateway)

Admission is challenge-owned and deterministic. **Do not** set gateway URL/token or LLM review
settings; residual keys fail closed at load:

```text
# Removed — do not configure
# PRISM_LLM_REVIEW_ENABLED
# PRISM_LLM_GATEWAY_URL
# PRISM_GATEWAY_TOKEN_FILE
# BASE_LLM_GATEWAY_URL
# BASE_GATEWAY_TOKEN
```

Pipeline order before GPU:

1. AST sandbox hard-blocks over both scripts.
2. Forced-seed parameter cap (150M).
3. Multi-GPU static contract and single-node bound.
4. Source similarity + duplicate policy (quarantine band → terminal **reject**, never held).

## Multi-GPU Static Contract

```bash
PRISM_DISTRIBUTED_CONTRACT_POLICY=reject     # reject | flag | off
PRISM_BASE_EVAL_MAX_GPU_COUNT=8
```

`reject` (default) hard-rejects a non-distributed `training.py`; `flag` advances but logs; `off` skips
the check.

## Duplicate / Similarity Policy

```bash
# thresholds also live in SQL runtime config under duplicate_thresholds
PRISM_PLAGIARISM_ENABLED=true
```

An exact-source-hash duplicate is rejected; a borderline-similarity quarantine is folded into a terminal
rejection at ingress. There is no operator hold-resolution surface.

## Raw-Weight Push

When master coordination is configured, PRISM periodically pushes authenticated raw hotkey weights to
the master private ingress (versioned epoch/revision, digest, idempotent). Configure master base URL
and challenge-scoped credentials via settings/secret files; never log tokens. Validators submit the
master-aggregated vector on-chain; PRISM does not call `set_weights`.

## External Results And TEE

Worker-plane result ingest accepts **only** `ExternalResultEnvelope` from the Base SDK. Legacy bodies
without the full binding/proof schema fail closed.

TEE:

- Local cryptographic fixtures may yield **`LOCAL-FIXTURE PASS`** under the fail-closed verifier.
- Real Lium/Targon production PASS is **blocked** until contracts and digests exist.
- Safe provider probes (inventory/health) never become REAL-PROVIDER PASS.

## Running Locally

```bash
PRISM_SHARED_TOKEN=dev-secret \
PRISM_DATABASE_URL=sqlite+aiosqlite:///./prism.sqlite3 \
.venv/bin/uvicorn prism_challenge.app:app --host 0.0.0.0 --port 8000
```

## BASE Deployment

PRISM registers as a challenge image reached by the master over the internal challenge network on
Compose. Public miner traffic goes through the BASE proxy. Use the master Compose install path
(`deploy/compose/install-master.sh` and related manifests), not Swarm.

> **Atomic rebrand deploy:** internal auth headers use `X-Base-*`
> (`X-Base-Challenge-Slug`, `X-Base-Verified-Hotkey`). PRISM accepts only `X-Base-*`, so cut over all
> services together; any sender left on legacy `X-Platform-*` fails internal auth until the cutover
> completes.

## Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/version
curl -H "Authorization: Bearer dev-secret" -H "X-Base-Challenge-Slug: prism" \
  http://localhost:8000/internal/v1/get_weights
```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `invalid internal token` | Shared token mismatch between BASE and PRISM |
| submission rejected before container | Static sandbox, two-script contract, param cap, distributed contract, or similarity/anti-cheat |
| residual LLM/gateway env rejected | Removed gateway fields still set; remove them |
| evaluation failed | Broker, image, GPU, timeout, missing locked data, or container error |
| empty weights / no push | No completed scored submissions yet; raw-weight push disabled or master URL missing |
| `result_envelope_invalid` | Body is not a full `ExternalResultEnvelope` |
| TEE elevated tier never grants | Expected until real-provider contracts land; only local fixtures can PASS |
| `missing_locked_data` | The read-only FineWeb-Edu train mount is absent or empty on the GPU node |
