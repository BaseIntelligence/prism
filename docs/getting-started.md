# Getting started

## The contract (recipe v1.4.0)

You ship either:

1. **Two scripts** — `architecture.py` (`build_model(ctx)`) + `training.py`
   (`train(model, ctx)`), or
2. A **source-tree ZIP** (recipe ≥ 1.3.0) — those seams plus optional helpers,
   `kernels/`, `tokenizer/`, `prism.toml`, `count_params.py`, `vendor.lock`.

The operator harness imports your seams, downloads the pinned dataset, verifies
its SHA-256, times the run, and reports `METRICS_JSON` (bpb, `bits_per_byte`,
tokenizer spec, tokens, steps, wall clock, gpu, params — plus the v3 battery
when enabled).

```python
# architecture.py
def build_model(ctx):
    """Return a model. Size embeddings from ctx["vocab_size"]."""

# optional — must live beside build_model (not in training.py)
def build_tokenizer(ctx):
    """Return your tokenizer (offline). See Tokenizer below."""

# training.py
def train(model, ctx):
    """Train; respect ctx.budget():
    budget.max_steps <= 20000 and budget.max_seconds <= 21600 (6h train).
    Use ctx["tokenizer"] — never from_pretrained("<hub id>") on the pod."""
```

Models must stay **≤ 350M parameters** after `build_model`. Since 1.3.0 a
breach is a **terminal Score(0)** (`CAP_EXCEEDED`), not a retryable failure.

## Tokenizer (yours — recipe ≥ 1.4.0)

**GPT-2 is no longer the challenge rule.** The harness resolves one tokenizer
per run and injects it as `ctx["tokenizer"]`, with vocab at `ctx["vocab_size"]`.
Declaration order (first match wins, always offline):

| Order | How you declare | Notes |
|-------|-----------------|-------|
| 1 | `tokenizer/` in a source-tree ZIP | Staged under `submission/tokenizer/` on the pod; ≤ **12** files, ≤ **8 MiB** total |
| 2 | `build_tokenizer(ctx)` in `architecture.py` | Must sit beside `build_model` — a hook in `training.py` is rejected |
| 3 | *(declare nothing)* | Pinned `gpt2` **fallback** (pre-1.4 behavior) — a default, not a rule |

```python
# architecture.py
def build_tokenizer(ctx):
    """Anything offline: train a BPE on ctx["dataset_path"], wrap a vendored
    implementation, or hand-roll a byte-level tokenizer. Must satisfy:

        tok(text, add_special_tokens=False)["input_ids"] -> list[int]
        tok.decode(ids) -> str            # roundtrips plain ASCII
        len(tok) or tok.vocab_size -> int # 256 .. 262144
        tok.eos_token_id -> int | None
    """
```

Your pod has **no network** (`unshare --net`), so `from_pretrained("<hub id>")`
inside your code fails closed. The harness validates the tokenizer and
fingerprints it; eval re-resolves it and refuses to score a mismatch — so
`build_tokenizer` must be deterministic.

**Fairness.** Different vocabs change tokenization, not the unit —
`bits_per_byte` (bits over UTF-8 bytes) is the tokenizer-neutral anchor. The
legacy `bpb` key is bits per *token* and only comparable at equal tokenizers.

## Source-tree submissions (recipe ≥ 1.3.0)

Optional layout (flat or one shared top-level folder):

```text
prism.toml            # optional: entry = "train.py"
architecture.py       # seam: build_model (+ optional build_tokenizer)
training.py           # seam: train  (or train.py)
count_params.py       # optional
kernels/              # optional custom ops (pure Python + torch)
tokenizer/            # optional HF-style tokenizer files
vendor.lock           # optional vendored *.py lock
```

Caps (intake): ≤ **128** files, ≤ **4 MiB**/file, ≤ **16 MiB** total
uncompressed (≤ 8 MiB compressed). The validated tree is staged on the pod under
`submission/` so sibling imports (`import kernels`) and `tokenizer/` resolve.
Trees with `kernels/` are eligible for 2×2 **attribution**
(`POST /v1/submissions/{id}/attribution`). See [Submit](submit.md).

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
| Two-script source size | 128 KiB per seam script |
| Source-tree | ≤ 128 files, ≤ 4 MiB/file, ≤ 16 MiB total |
| `tokenizer/` (in tree) | ≤ 12 files, ≤ 8 MiB total |
| Model parameters | ≤ **350 000 000** after `build_model` (`CAP_EXCEEDED` → Score(0)) |

## Recipe pin

`GET /v1/recipe` returns the versioned descriptor (dataset URL/hash, caps, harness
digest, recipe version). `GET /v1/recipe/baseline` returns the official baseline
scripts — the best starting point for your own architecture.

## Next

→ [Submit](submit.md)  
→ [Scoring & competition](scoring.md)
