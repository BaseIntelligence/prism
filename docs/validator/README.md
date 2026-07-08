# Validator Guide

## Purpose

PRISM lets validators operate an "ability to learn" challenge for BASE: accept signed miner
submissions, run the static sandbox and the LLM hard gate, re-execute the miner's training loop under a
forced random init on locked data, compute the prequential bits-per-byte score themselves, and expose
normalized dry-run weights to BASE.

## Responsibilities

- accept only signed submissions and enforce replay protection and size limits;
- keep evaluation isolated from the host (broker-backed containers, `network=none`);
- keep the locked `val`/`test` splits secret and never expose them to a miner script;
- force the seed and deterministic flags so runs reproduce;
- compute the score from the challenge-owned capture, never trusting miner-reported numbers;
- protect shared BASE, broker, and LLM gateway tokens;
- monitor scoring, rejections, failures, quarantine, and exported weights.

## Evaluation Lifecycle

1. A miner submits a signed two-script bundle; PRISM validates the hotkey, timestamp, nonce, and size.
2. The bundle is resolved into the two-script contract and inspected by the AST sandbox.
3. The forced-seed `build_model` instantiation enforces the 150M parameter cap.
4. The multi-GPU static contract and single-node bound are checked.
5. The LLM hard gate reviews both scripts; a `reject` is terminal before any GPU work.
6. The challenge re-executes `training.py` under a forced random init on the locked train split and
   captures the single-pass online loss itself.
7. PRISM computes the prequential bits-per-byte score, the held-out delta tie-breaker, and the
   anti-memorization gap, and writes `prism_run_manifest.v2.json`.
8. Scores persist; the leaderboard ranks by `final_score`; BASE reads normalized dry-run weights.

## Runtime Configuration

Settings use the `PRISM_` prefix (and compatible `CHALLENGE_` values where declared) to bootstrap the
process and provide fallback defaults. Runtime policy is SQL-first, and official runtime config fails
closed when an active SQL value is invalid:

```text
SQL active value → env/Pydantic default → schema default
```

### SQL Runtime Config Keys

| Key | Policy area |
| --- | --- |
| `reward_pools` | Weight-split row (compat); live weights normalize per hotkey from `final_score`. |
| `score_weights` | Score-weight row (compat); live primary score is prequential bits-per-byte. |
| `benchmark_weights` | Benchmark-mix row (compat); not part of the live bits-per-byte score. |
| `duplicate_thresholds` | Source/graph/quarantine/static-reject thresholds. |
| `llm_review_policy` | Hard-gate enable/required, confidence, timeout, evidence; provider/model gateway-injected (a `master.yaml` choice). |
| `gpu_policy` | Max GPU count, fixed GPU count, GPU type, fixed-profile flag. |
| `dataset_configs` | Locked FineWeb-Edu sample count, frozen revision, split names. |
| `execution_mode_targets` | The `gpu_proxy_eval` and `full_scale_eval` token/GPU targets. |
| `artifact_limits` | Code/artifact size limits plus the required `prism_run_manifest.v2.json` name. |
| `sandbox_limits` | Docker, CPU, memory, PID, timeout, network, read-only limits. |
| `diagnostics_thresholds` | Activation, gradient, attention, representation health thresholds. |
| `loss_comparability_policy` | Comparable-loss requirements and byte-normalized fallback. |

Rows are audited (`config_key`, `value_json`, `schema_version`, `updated_by`, `updated_at`,
`effective_from`, `enabled`); the active row per key is the newest enabled row whose `effective_from`
has arrived.

### Environment Settings

| Setting | Purpose |
| --- | --- |
| `PRISM_DATABASE_URL` | Persistent SQLite storage location. |
| `PRISM_SHARED_TOKEN` / `PRISM_SHARED_TOKEN_FILE` | BASE internal-call token (prefer file delivery). |
| `PRISM_PUBLIC_SUBMISSIONS_ENABLED` | Enables the direct public submission route. |
| `PRISM_SIGNATURE_TTL_SECONDS` | Replay-protection timestamp window. |
| `PRISM_EPOCH_SECONDS` | Scoring epoch length. |
| `PRISM_MAX_CODE_BYTES` | Maximum submission size. |
| `PRISM_MAX_PARAMETERS` | Hard parameter cap (default 150M). |
| `PRISM_BASE_EVAL_IMAGE` | CI-published `prism-evaluator` image (sentencepiece + offline tiktoken). |
| `PRISM_BASE_EVAL_DATA_DIR` | Read-only locked FineWeb-Edu **train** mount. |
| `PRISM_BASE_EVAL_VAL_DATA_DIR` | Secret held-out **val** split (scorer-only; never mounted into eval). |
| `PRISM_BASE_EVAL_MAX_GPU_COUNT` | Max GPU count (default and hard max 8). |
| `PRISM_BASE_EVAL_GPU_COUNT` | Scored GPU count (default 1; the `nproc=1` path). |
| `PRISM_DISTRIBUTED_CONTRACT_POLICY` | `reject` / `flag` / `off` for the multi-GPU contract. |
| `PRISM_LLM_REVIEW_ENABLED` | Enables the LLM hard gate (default on). |
| `PRISM_LLM_GATEWAY_URL` | Gateway base URL (`{root}/llm/v1`); alias `BASE_LLM_GATEWAY_URL`. |
| `PRISM_GATEWAY_TOKEN_FILE` | Scoped gateway token file (default `/run/secrets/base_gateway_token`); alias `BASE_GATEWAY_TOKEN_FILE`. |

The gate routes **only** through `PRISM_LLM_GATEWAY_URL` (`{root}/llm/v1`) with the scoped
`X-Gateway-Token`; the gateway injects the provider and model server-side (no raw provider key, no
pinned model). Use secret files for the shared, broker, and gateway tokens.

## FineWeb-Edu And Execution Modes

| Mode | Operator use | Dataset target |
| --- | --- | --- |
| `gpu_proxy_eval` | Default official scored re-execution. | FineWeb-Edu `sample-10BT` locked shards. |
| `full_scale_eval` | Larger official scored re-execution. | FineWeb-Edu `sample-10BT` then `sample-100BT` phases. |

Both modes are score-eligible and run on the locked, read-only FineWeb-Edu data with `network=none`. The
retired local-CPU smoke mode is gone. Official scoring uses a fixed `gpu_policy` profile: the max is 8
GPUs and the scored run uses 1 GPU (`torchrun --standalone --nnodes=1 --nproc-per-node=1`); PRISM is
single-node only.

## Routes

Public miner surface and BASE contract:

```http
POST /v1/submissions
GET  /v1/submissions/{submission_id}
GET  /v1/leaderboard
GET  /v1/architectures
GET  /v1/architectures/{architecture_id}/variants
GET  /v1/epochs/current
GET  /health
GET  /version
GET  /internal/v1/get_weights      # Authorization: Bearer <shared-token>, X-Base-Challenge-Slug: prism
```

`get_weights` returns one normalized weight per hotkey (best submission per hotkey). Weights are always
dry-run and are never written on-chain.

## Review And Quarantine

PRISM uses the static AST sandbox, the forced-seed parameter cap, the multi-GPU static contract, the LLM
hard gate, and a deterministic duplicate check. A `reject` from any static gate or the LLM gate is
terminal before any GPU work. A borderline duplicate is folded into a terminal rejection at ingress:
there is no operator hold-resolution surface (the v1-NAS component-review and ownership machinery was
decommissioned).

## Checklists

**Setup:** persistent SQLite storage; shared-token delivery via files or a secret manager; the broker,
evaluator image, and read-only locked-data mounts; submission size and parameter limits; the scoped LLM
gateway token; then run `pytest tests/test_config.py -q` plus the scoring/harness suites and submit a
known-safe bundle to confirm the leaderboard and `get_weights`.

**Operation and security:** require hotkey signatures and short replay windows; keep the `val`/`test`
splits secret and out of any miner-visible path, fixture, or log; keep the eval container on
`network=none` with the rootfs read-only except `artifacts_dir`; keep broker, BASE, and LLM gateway
tokens out of logs; monitor rejected, failed, quarantined, and completed submissions separately; and
confirm weights stay dry-run with no on-chain weight-setter.
