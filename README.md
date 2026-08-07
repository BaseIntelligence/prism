<div align="center">

# PRISM

**Miner guide for the BASE prism challenge — HTTP recipe submit.**

[![BASE](https://img.shields.io/badge/BASE-subnet-black.svg)](https://github.com/BaseIntelligence/base)
[![Bittensor](https://img.shields.io/badge/Bittensor-subnet-black.svg)](https://bittensor.com/)
[![License](https://img.shields.io/github/license/BaseIntelligence/prism)](LICENSE)

![PRISM banner](assets/banner.jpg)

[Overview](docs/README.md) ·
[Getting started](docs/getting-started.md) ·
[Submit](docs/submit.md) ·
[Scoring & competition](docs/scoring.md) ·
[API](docs/api.md) ·
[Examples](examples/baseline/)

</div>

## What it is

PRISM is a research challenge: you try **new architectures** (and optionally
training recipes / tokenizers) and the challenge re-executes them fairly. You
submit a ZIP — either the classic two scripts (`architecture.py` +
`training.py`) or a **source-tree** with helpers, `kernels/`, and optional
`tokenizer/` — and the operator runs them on a GPU pod against a pinned
FineWeb-Edu shard. Live leaf score is pure **bits-per-byte** (bpb, lower is
better); v3 also measures a G1–G8 battery in shadow mode. There is **no** miner
Docker image, no CVM, no on-chain write from miners — HTTP submit only.

| | |
|---|---|
| Challenge id | `prism` |
| Production gateway | `https://chain.joinbase.ai` |
| Staging gateway | `http://staging.api.joinbase.ai` |
| Submit path | `/challenge/prism/v1/submissions` |
| Recipe | **v1.4.0** — miner-chosen tokenizer; G5 = RULER + BABILong + natural docs (**pretrain-only**) |

This repository holds **miner documentation and examples only**. Control-plane
source lives in [BaseIntelligence/base](https://github.com/BaseIntelligence/base).

## Start here

1. Read [Getting started](docs/getting-started.md) — tokenizer + source-tree
   contracts matter from recipe **1.3.0 / 1.4.0**.
2. Copy [`examples/baseline/`](examples/baseline/) — required telemetry hooks
   (`prism_telemetry.report` + `finish_evaluation`) and `ctx["tokenizer"]`.
3. Zip and submit — see [Submit](docs/submit.md).
4. Poll events until `terminated`, then check your bpb — see [API](docs/api.md).

```bash
export GATEWAY=https://chain.joinbase.ai
export HOTKEY=<64 lowercase hex>   # public hotkey only — never a secret key

cd examples/baseline
zip -j submission.zip architecture.py training.py

curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

## The three things miners get wrong

1. **Missing telemetry hooks** — `training.py` must import `prism_telemetry` and call
   `report(...)` during training (and may call `finish_evaluation()` to stop early).
   Missing hooks = hard contract violation, zero score, terminal.
2. **Copying someone's `architecture.py`** — the pre-GPU copy gate rejects byte/AST
   copies of *earlier* architectures with zero score, no appeal. Starting from the
   published baseline is fine.
3. **Hub downloads / hardcoded GPT-2** — the pod has **no network**. Use
   `ctx["tokenizer"]` (and size embeddings from `ctx["vocab_size"]`). GPT-2 is
   only the harness **fallback** when you declare nothing — not a challenge rule.
   A second architecture submit while gated returns `409 submission_gated`;
   training-only entries on published archs are separate slots.
