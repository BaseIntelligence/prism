# Miner Guide

## Purpose

PRISM rewards miners whose models learn fast from scratch. You submit two scripts, a model
`architecture.py` and a custom `training.py` loop; the challenge re-executes your loop under a forced
random init on locked FineWeb-Edu data and scores it with a prequential bits-per-byte metric. You bring
the model and the loop, not the data or the metric.

For offline architecture-agnostic **Official Comparison Protocol v1** (held-out generalization
primary, prequential bpb secondary, honest hooks, wall-clock never ranks), see
[Official Comparison](../official-comparison.md). The live leaderboard path remains described in
[Scoring](../scoring.md).

## Miner Flow

1. Build a two-script bundle that follows the PRISM contract.
2. Sign and submit it with your miner hotkey.
3. PRISM runs the static sandbox and **deterministic admission** (similarity / anti-cheat). There is no
   LLM hard gate.
4. The challenge re-executes your `training.py` under a forced random init on the locked train split.
5. The challenge computes your prequential bits-per-byte score and the held-out delta tie-breaker.
6. Better learners earn more weight after master aggregation of PRISM raw weights (validators submit
   on-chain; the challenge does not).

## Lab seed families

Tracked lab seeds under `examples/` package with the same outer two-script zip contract via
`scripts/pack_seed_family.py` / `prism_challenge.seed_packaging`:

| Family id | Path | Notes |
| --- | --- | --- |
| `transformer-tiny-1m` | `examples/tiny-1m` | Weight-tied ~1M decoder transformer; param count is forced-seed realized params; multi-GPU single-node ≤8 |
| `mamba-tiny-1m` | `examples/mamba-tiny` | Pure-PyTorch selective SSM (Mamba-style); **no** `mamba_ssm` C++/CUDA dep for the static lab path; same 150M cap and multi-GPU contract |

Family knobs that matter for lab interpretation (not product baseline tables):

- **Param counting** — architecture-agnostic, counts tensors from `build_model(ctx)`; both seeds weight-tie emb/lm_head. Mamba counts include `A_log`/`D`/conv/dt projections rather than MHA/MLP tensors.
- **Step throughput** — `LOCAL_BATCH`, optimizer LR, and token budget dominate step flu; score is compute-normalized. Pure-torch Mamba sequential scan is slower/token than fused CUDA kernels (use modest LR; default seed uses `0.003` vs transformer `0.005`).
- **Stability** — multi-GPU static contract requires distributed primitives + rank-0 writes; works at `world_size=1` for both families. Mamba pure-torch caveat: do not introduce blocked `mamba_ssm` / `cpp_extension` imports if you still need AST sandbox static pass.

## The Two-Script Contract

A bundle is a `.zip` (or directory) with two distinct scripts. An optional `prism.yaml` declares the
entrypoints and tokenizer:

```yaml
architecture:
  entrypoint: architecture.py
training:
  entrypoint: training.py
tokenizer: gpt2
```

`architecture.py` exposes the model factory; `training.py` exposes the loop you own:

```python
# architecture.py
def build_model(ctx):
    return MyModel(ctx.vocab_size)
```

```python
# training.py
from architecture import build_model

def train(ctx):
    model = build_model(ctx)
    # optimizer/schedule, read the locked train split from ctx.data_dir, tokenize,
    # run the loop, handle multi-GPU, write only under ctx.artifacts_dir.
    ...
```

`build_model(ctx)` returns any `torch.nn.Module` under the AST sandbox, the 150M parameter cap, and the
resource limits; it must not read data, open files, touch the network, or reference the dataset.
`train(ctx)` owns the optimizer, schedule, dataloading, tokenization, multi-GPU strategy, and loop. The
single-module re-export idiom no longer satisfies the contract: the two roles must be distinct files.

## Context And Limits

`ctx` is a `PrismContext` supplying the metadata and limits you need:

- `vocab_size`, `max_seq_len` — token-id geometry;
- `max_params` — the 150M cap;
- `seed` — the forced seed you cannot change;
- `data_dir` — read-only path to the locked FineWeb-Edu **train** split;
- `artifacts_dir` — the only writable path;
- `world_size`, `rank`, `local_rank`, `device` — the distributed launch;
- `token_budget` / `step_budget` — the compute budget;
- `ctx.build_model()` and `ctx.reference_tokenizer("gpt2" | "llama")` — offline, no network.

Read raw text from `ctx.data_dir` and tokenize with your own tokenizer or a pre-staged reference; fail
closed if the locked data is missing.

## Locked Data, No Network

The train split is read-only at `ctx.data_dir`; the `val`/`test` splits are secret and never exposed to
your script. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is no network during training: do not download data, tokenizers, or
weights at runtime.

## Multi-GPU

Your `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>`; PRISM is single-node (1-8 GPUs) and the
official scored run uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path). A
correct loop calls `init_process_group`, wraps the model with DDP or FSDP, shards data per-rank, does
rank-0-only writes, all-reduces metrics, tears down the process group, and also works at `world_size=1`.
It is checked with a static contract and a gloo multi-rank test. See [Scaling](../scaling.md).

## The Challenge Computes The Score

PRISM re-executes your loop under a forced random init, captures the single-pass online loss itself, and
writes a challenge-authored `prism_run_manifest.v2.json`; any value or manifest you write is ignored.
The score is the prequential bits-per-byte (area under the from-scratch loss curve, normalized by raw
UTF-8 bytes) with a held-out delta-over-random-init tie-breaker. A smuggled pretrained model shows an
anomalous step-0 loss and is zeroed; an excessive train-vs-held-out gap is penalized as memorization.

## Submitting Work

Submit through the public route when enabled, or through the BASE proxy in production:

```http
POST /v1/submissions
Content-Type: application/json
```

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

The hotkey must match the signature (timestamps and nonces block replay), and stay within the size
limit. Unsafe imports, network access, arbitrary filesystem access, deserialization escapes, and the
single-module idiom are rejected at static review before any GPU work. Close source copies of prior
work can be rejected by deterministic similarity (including borderline bands) with no operator review
queue.

## What Improves Your Score

- Drive the from-scratch loss down fast (lower bits-per-byte is better).
- Use the compute budget efficiently (scoring is compute-normalized, never wall-clock).
- Grow the held-out delta-over-random-init on the secret val split (the near-tie tie-breaker).
- Keep the train-vs-held-out gap small (a large gap is penalized as memorization).
- Ship correct, DDP-safe, rank-aware distributed behavior.

## Miner Checklist

- Ship two distinct scripts: `architecture.py` with `build_model(ctx)` and `training.py` with `train(ctx)`.
- Keep `build_model` pure (no data, files, or network); read only `ctx.data_dir`, write only `ctx.artifacts_dir`.
- Stay under the 150M parameter cap and inside the AST sandbox.
- Make the loop deterministic under the forced seed and correct at `world_size=1`.
- Remove secrets, private endpoints, generated caches, and unrelated files.
