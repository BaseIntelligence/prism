# Operator Guide

Local validation and production configuration for running PRISM as a BASE challenge.

## Installation

```bash
git clone https://github.com/BaseIntelligence/prism.git
cd prism
uv sync --frozen --extra dev
```

## Local Validation

```bash
.venv/bin/ruff check src
.venv/bin/mypy src/prism_challenge/evaluator
.venv/bin/python -m pytest tests -q
```

## Required Runtime Configuration

```bash
PRISM_DATABASE_URL=sqlite+aiosqlite:////data/prism.sqlite3
PRISM_SHARED_TOKEN_FILE=/run/secrets/base/challenge_token
PRISM_EXECUTION_BACKEND=base_gpu
```

The shared token must match the token configured in the BASE master for this challenge.

## Docker Broker Configuration

Production evaluation uses the BASE Docker broker with the CI-published evaluator image:

```bash
PRISM_DOCKER_ENABLED=true
PRISM_DOCKER_BACKEND=broker
PRISM_DOCKER_BROKER_URL=http://base-docker-broker:8082
PRISM_DOCKER_BROKER_TOKEN_FILE=/run/secrets/base/challenge_token
PRISM_BASE_EVAL_IMAGE=ghcr.io/baseintelligence/prism-evaluator:latest
PRISM_BASE_EVAL_GPU_COUNT=1
PRISM_DOCKER_NETWORK=none
```

The scored run is single-node (`torchrun --standalone --nnodes=1 --nproc-per-node=1`). The evaluator
image must ship `sentencepiece` and an offline tiktoken gpt2 cache so reference tokenizers load with no
network.

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

## LLM Hard Gate Configuration

The LLM hard gate is on by default and reviews both scripts before any GPU work. It routes **only**
through the BASE master LLM gateway: PRISM holds no raw provider key and pins no model; the gateway
selects the provider and model server-side (a `master.yaml` choice, keyed by the scoped token) and
injects them per request, so the challenge stays provider-agnostic.

```bash
PRISM_LLM_REVIEW_ENABLED=true
PRISM_LLM_GATEWAY_URL=http://base-master-proxy:19080/llm/v1
PRISM_GATEWAY_TOKEN_FILE=/run/secrets/base_gateway_token
```

The gate authenticates with the scoped token (the `X-Gateway-Token` header) and posts to
`{gateway}/chat/completions`. In a BASE deployment the master injects the equivalent
`BASE_LLM_GATEWAY_URL` (=`{gateway_root}/llm/v1`) and `BASE_GATEWAY_TOKEN`, so operators normally do not
set these by hand. A `reject` is terminal, and the eval container carries no gateway token or provider
key (the gate runs host-side before launch).

## Multi-GPU Static Contract

```bash
PRISM_DISTRIBUTED_CONTRACT_POLICY=reject     # reject | flag | off
PRISM_BASE_EVAL_MAX_GPU_COUNT=8
```

`reject` (default) hard-rejects a non-distributed `training.py`; `flag` advances but logs; `off` skips
the check.

## Duplicate Review

```bash
PRISM_PLAGIARISM_ENABLED=true
```

An exact-source-hash duplicate is rejected; a borderline-similarity quarantine is folded into a terminal
rejection at ingress. There is no operator hold-resolution surface.

## Running Locally

```bash
PRISM_SHARED_TOKEN=dev-secret \
PRISM_DATABASE_URL=sqlite+aiosqlite:///./prism.sqlite3 \
.venv/bin/uvicorn prism_challenge.app:app --host 0.0.0.0 --port 8000
```

## BASE Deployment

PRISM registers as a challenge image reached by the master over the internal challenge network. Public
miner traffic goes through the BASE proxy. Weights are exposed only via `get_weights` and are always
dry-run.

> **Atomic rebrand deploy:** internal auth headers were renamed `X-Platform-*` → `X-Base-*`
> (`X-Base-Challenge-Slug`, `X-Base-Verified-Hotkey`). PRISM accepts only `X-Base-*`, so cut over all
> services together (base → agent-challenge → frontend, then PRISM); any sender left on `X-Platform-*`
> fails internal auth until the cutover completes.

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
| submission rejected before container | Static sandbox, two-script contract, param cap, distributed contract, or LLM hard-gate reject |
| submission held | LLM review quarantine (transient error or ambiguous verdict) |
| evaluation failed | Broker, image, GPU, timeout, missing locked data, or container error |
| empty weights | No completed, scored submissions yet |
| `missing_locked_data` | The read-only FineWeb-Edu train mount is absent or empty on the GPU node |
