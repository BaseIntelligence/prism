# Getting started

## The contract (recipe v1.2.0)

You ship **two scripts only**. The operator harness (`prism_harness.py`) imports them,
downloads the pinned dataset, verifies its SHA-256, times the run, and reports
`METRICS_JSON` (bpb, tokens, steps, wall clock, gpu, params).

```python
# architecture.py
def build_model(ctx):
    """Return a model given the recipe context (devices, dims, seed)."""

# training.py
def train(model, ctx):
    """Train the model; must respect ctx.budget():
    budget.max_steps <= 20000 and budget.max_seconds <= 21600 (6h train)."""
```

No third source file, no offline weights, no network at pod runtime beyond the pinned
dataset pull.

## Telemetry hooks (required since recipe 1.1.0)

The harness registers a `prism_telemetry` module before your code loads (also at
`ctx["telemetry"]`). Your `training.py` **MUST**:

```python
import prism_telemetry

prism_telemetry.report(loss=..., step=..., grad_norm=..., layer_stats=...)  # every N steps
prism_telemetry.finish_evaluation()  # optional early stop: score the model as-is
```

- `report(...)` feeds the loss/gradient/layer series persisted master-side and shown on
  the site (`/v1/site/arenas/prism/submissions/{id}/telemetry`).
- `finish_evaluation()` raises a `BaseException` through `train()` so your own
  `except Exception` blocks cannot swallow it; without it the eval ends when `train()`
  returns or the wall-clock cap fires.
- **Missing hooks are a hard contract violation**: the review fails the submission
  (`missing_telemetry_hooks`, zero score, terminal — no retry).

The [baseline example](../examples/baseline/) shows the exact pattern, including an
offline fallback stub for local testing.

## Training-only submissions (architecture competition, recipe ≥ 1.2.0)

Instead of shipping both scripts you can submit `training.py` + `arch_id` referencing a
**published** architecture. The master pulls `architecture.py` from the registry; the
same harness contract applies unchanged. Published archs: `GET /v1/architectures`.
See [Submit](submit.md#training-only-entries).

## Pinned dataset

| Field | Value |
|-------|-------|
| Ref | `HuggingFaceFW/fineweb-edu@sample/10BT` |
| URL | `…/resolve/main/sample/10BT/010_00000.parquet` |
| Bytes | 2 152 798 864 |
| SHA-256 | `e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6` |

The harness re-verifies the hash on the file it actually fetched; a mismatch ends the
eval as `ChallengeInternal` — never a miner score.

## Budget & caps

| Cap | Value |
|-----|-------|
| Train wall clock | 6.0 h per submission |
| Pod lifetime | 7.0 h (train + bootstrap margin) |
| Hard step cap | 20 000 |
| Source size | 128 KiB per script |
| Model parameters | ≤ **350 000 000** after `build_model` |
| `train_rows` (from `GET /v1/recipe`) | **2048** — baseline / default cut in `ctx` |
| `val_rows` | **256** — frozen val scored by the harness |

`train_rows` is what the **sealed baseline** trains on (~2M GPT-2 tokens for
that slice). It is **not** a hard “you only get 2048 rows” ceiling for
competitive recipes: the harness gives you the full pinned parquet at
`ctx["dataset_path"]`, and you may stream it until the 6h / 20k-step guard
fires. Token count then depends on your loop and the GPU — a long Lium run can
reach ~O(10⁹) tokens. Marketing charts that once said “2.6B tokens · single
pass” were showing a leader’s **observed** telemetry, not a fixed recipe
quota. Always trust live `GET /v1/recipe` (`pin_hex`, `train_rows`, caps).

The sealed baseline is deliberately mediocre (short cut, few steps). Matching
a board BPB near ~4–5 requires a competitive trainer, not an unmodified
baseline on a 4090 for a few minutes.

## Recipe pin

`GET /v1/recipe` returns the versioned descriptor (dataset URL/hash, caps,
`train_rows` / `val_rows`, recipe version, `pin_hex`). Production today is
recipe **1.2.0** — open docs PRs that advertise 1.3+/1.4.0/v3 scoring describe
**unreleased** control-plane work (`prism-better`), not what
`https://chain.joinbase.ai` executes. `GET /v1/recipe/baseline` returns the
official baseline scripts — the best starting point for your own architecture.

## Next

→ [Submit](submit.md)  
→ [Scoring & competition](scoring.md)
