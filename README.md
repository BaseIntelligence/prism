<div align="center">

# PRISM

**Miner guide for the BASE prism challenge — HTTP AutoModel patch submit.**

Recipe **2.1.0** is a **new competition** (`prism-v2.1`). Old 2.0 scores do not pay; weights burn until the first eligible 2.1 run.

[![BASE](https://img.shields.io/badge/BASE-subnet-black.svg)](https://github.com/BaseIntelligence/base)
[![Bittensor](https://img.shields.io/badge/Bittensor-subnet-black.svg)](https://bittensor.com/)
[![License](https://img.shields.io/github/license/BaseIntelligence/prism)](LICENSE)

![PRISM banner](assets/banner.jpg)

[Overview](docs/README.md) ·
[Getting started](docs/getting-started.md) ·
[Submit](docs/submit.md) ·
[Scoring & competition](docs/scoring.md) ·
[API](docs/api.md) ·
[Full guide](docs/prism.md)

</div>

## What it is

PRISM is a research challenge on a pinned
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) base: you fork the
operator pin, edit under that tree, and submit a **unified git diff**. The
operator applies your patch fail-closed, then re-executes training on a
miner-funded **4-GPU** Lium pod against a pinned FineWeb-Edu shard. Live leaf
score is the **equal-weight G2 public-suite mean** (`scoring_version` 4).
There is **no** miner Docker image, no CVM, no on-chain write from miners —
HTTP submit only.

| | |
|---|---|
| Challenge id | `prism` |
| Production gateway | `https://chain.joinbase.ai` |
| Staging gateway | `http://staging.api.joinbase.ai` |
| Submit path | `/challenge/prism/v1/submissions` |
| Recipe | **2.1.0** — AutoModel pin + patch (`automodel@v0.5.0`), 4-GPU CUDA 13/TE |
| Live GPU | Miner-funded Lium — pass `X-Lium-Api-Key` |

This repository holds **miner documentation and examples only**. Control-plane
source lives in [BaseIntelligence/base](https://github.com/BaseIntelligence/base).

## Start here

1. Read [Getting started](docs/getting-started.md) (or the [full guide](docs/prism.md)).
2. `GET /v1/recipe` — copy `automodel_pin_id`, `automodel_git_commit`, and caps.
3. Checkout that AutoModel commit → edit → `git diff <commit> > automodel.patch`.
4. Pack `automodel.base` + `automodel.patch` (+ optional `prism.toml` /
 `requirements.txt`) and submit with your hotkey + **`X-Lium-Api-Key`** —
 see [Submit](docs/submit.md). Worked example: [LoopMoE](examples/loopmoe/).
5. Poll events until `terminated`, then check G2 benches — see [API](docs/api.md).

```bash
export GATEWAY=https://chain.joinbase.ai
export HOTKEY=<64 lowercase hex>   # public hotkey only — never a secret key
export LIUM_API_KEY=<your Lium API key>

# After forking the pin and producing automodel.base + automodel.patch:
zip -j submission.zip automodel.base automodel.patch   # + prism.toml if used

curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  --data-binary @submission.zip
```

## The three things miners get wrong

1. **Legacy 1.x ZIPs** — `architecture.py` + `training.py` (or training-only
 `arch_id`) return `400 unsupported_layout` / `recipe_version` on live 2.1.
 Ship `automodel.base` + `automodel.patch` only.
2. **Wrong pin / stale diff** — `automodel.base` must equal live
   `automodel_pin_id` (`automodel@v0.5.0`); regenerate the patch against the
   exact `automodel_git_commit` from `/v1/recipe`.
3. **Missing `X-Lium-Api-Key`** — live eval runs on **your** Lium account.
   Missing key → `400 missing_lium_api_key`.
