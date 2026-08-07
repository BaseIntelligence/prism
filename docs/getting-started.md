# Getting started

## The contract (recipe v1.3.0)

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

No offline weights, no network at pod runtime beyond the pinned dataset pull.
Since recipe **1.3.0** you may alternatively ship a full **source tree** (the two seam
files plus `prism.toml`, `count_params.py`, `kernels/`, `vendor.lock`) — see
[Submit](submit.md#source-tree-zip-recipe--130).

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

## Recipe pin

`GET /v1/recipe` returns the versioned descriptor (dataset URL/hash, caps, harness
digest, recipe version). `GET /v1/recipe/baseline` returns the official baseline
scripts — the best starting point for your own architecture.

## Next

→ [Submit](submit.md)  
→ [Scoring & competition](scoring.md)
