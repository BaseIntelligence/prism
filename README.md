<div align="center">

# PRISM

**An "ability to learn" ML challenge — two-script submissions, locked data, challenge-owned scoring.**

<a href="docs/overview.md">Overview</a> ·
<a href="docs/miner/README.md">Miners</a> ·
<a href="docs/validator/README.md">Validators</a> ·
<a href="docs/architecture.md">Architecture</a> ·
<a href="docs/scoring.md">Scoring</a> ·
<a href="docs/security.md">Security</a>

[![License](https://img.shields.io/github/license/BaseIntelligence/prism)](LICENSE)
[![Bittensor](https://img.shields.io/badge/Bittensor-subnet-black.svg)](https://bittensor.com/)
[![BASE](https://img.shields.io/badge/BASE-v3.1.2-6f42c1.svg)](https://github.com/BaseIntelligence/base/releases/tag/v3.1.2)

![PRISM Banner](assets/banner.png)

</div>

---

## Overview

PRISM is a [BASE](https://joinbase.ai) subnet challenge that measures a model's **ability to learn**
from scratch. Miners submit a **two-script** bundle — `architecture.py` (`build_model(ctx)`) and
`training.py` (`train(ctx)`) — and the challenge owns everything else: a locked **FineWeb-Edu**
dataset (read-only, no network) and the scoring. **The miner owns** the model and the training loop;
the challenge owns the data and the score.

Every scored run is re-executed under a **forced random init**, so the score is a **prequential**
(online) compression metric in **bits-per-byte** — the area under the from-scratch loss curve,
normalized by bytes consumed. Admission and scoring are **deterministic** (no LLM gateway). Raw
weights are pushed to BASE for master aggregation; validators fetch the final vector and call
`set_weights` under their own hotkeys.

### Base SDK pin

PRISM depends on the immutable Base public wheel:

```text
https://github.com/BaseIntelligence/base/releases/download/v3.1.2/base-3.1.2-py3-none-any.whl
#sha256=3a61c2d3a343ed6de55e80215486e3de0c9639276443d08f2ed316bc807f2ff0
```

(see `pyproject.toml`). There is no LLM gateway dependency in this pin.

## How It Works

```mermaid
flowchart LR
    M[Miner two-script bundle] --> G{Static sandbox + param cap}
    G -- reject --> X[[rejected]]
    G --> A[Deterministic admission]
    A --> V[Validator re-executes<br/>forced random init]
    V --> S[Prequential bpb + held-out delta]
    S --> W[Raw-weight push → BASE master]
```

1. **Submit** — a signed `architecture.py` + `training.py` bundle (a single combined module is rejected).
2. **Static gates** — AST sandbox, 150M parameter cap, single-node multi-GPU contract; any failure is terminal before GPU.
3. **Deterministic admission** — challenge-owned checks only; the former LLM gateway hard gate is removed.
4. **Forced-init re-execution** — one validator re-runs the loop on the locked FineWeb-Edu train split and captures the online loss itself (miner-reported numbers are ignored).
5. **Scoring** — the challenge computes prequential bits-per-byte plus a secret held-out delta tie-breaker.
6. **Weights** — emission splits two-tier (best architecture `0.60` / best training variant `0.40`); raw weights push to BASE master aggregation, then validators submit on-chain (or a fake chain in tests).

## Anti-Cheat By Construction

Common cheats are **inert**, not merely detected:

- **No pretrained weights** — forced random init makes smuggled weights inert; an anomalous step-0 loss zeroes the score; the container runs `network=none`.
- **No metric gaming** — the challenge recomputes the metric from the loss it captured; miner-reported numbers and manifests are ignored.
- **No memorization** — the secret `val`/`test` splits never leave the master; an excessive train-vs-held-out gap is penalized.
- **Deterministic** — fixed seeds and a challenge-controlled data order reproduce the same score within tolerance.

## TEE Verifier

PRISM includes a **Prism-only, fail-closed local TEE fixture verifier** for unit and contract tests.
Real Lium/Targon remote attestation that would produce a production PASS is **blocked** until those
provider readiness gates are satisfied. Local fixture verification does not imply live TEE production
readiness on Lium or Targon.

## Worker Plane (optional)

PRISM can move GPU re-execution onto **miner-funded workers** (deployed on Lium/Targon via the BASE
`base worker` CLI). Validators then run verify-only plausibility checks plus probabilistic audits,
and each result carries an `ExecutionProof` (manifest hash + worker sr25519 signature, with optional
image-digest and attestation tiers). Gated behind `worker_plane` (default off). See the
<a href="https://github.com/BaseIntelligence/base/blob/main/docs/miner/worker-plane.md">worker deployment guide</a>.

## Documentation

| Guide | Contents |
|-------|----------|
| <a href="docs/overview.md">Overview</a> | The challenge in one page |
| <a href="docs/miner/README.md">Miner guide</a> | Build and submit a two-script bundle |
| <a href="docs/validator/README.md">Validator guide</a> | Run evaluation on your own broker |
| <a href="docs/architecture.md">Architecture</a> | Service design and forced-init re-execution |
| <a href="docs/submissions.md">Submission format</a> | The two-script contract and `PrismContext` |
| <a href="docs/scoring.md">Scoring & rewards</a> | Leaderboard prequential bits-per-byte and tie-breakers |
| <a href="docs/official-comparison.md">Official Comparison v1</a> | Held-out primary / bpb secondary pair protocol (lab) + multimetric.v1.1 scorecard annex |
| <a href="docs/scaling.md">Scaling</a> | Single-node multi-GPU contract |
| <a href="docs/security.md">Security model</a> | Sandbox, deterministic admission, anti-cheat |
| <a href="docs/api.md">API</a> | Internal and public routes |
| <a href="docs/operators.md">Operators</a> | Deploy and run under BASE Compose |

## Development

```bash
uv run ruff check .
uv run mypy
uv run pytest --cov=prism_challenge --cov-fail-under=80
```

GPU re-execution, HuggingFace publication, and external provider calls are mocked in tests; real GPU
and provider keys are wired only at deploy. The LLM gateway is not part of the test or deploy path.

## License

Apache-2.0
