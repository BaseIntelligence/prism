# PRISM miner docs

Live recipe is **2.1.0** / competition **`prism-v2.1`** (`scoring_generation`
**21**): submit an AutoModel pin id + unified git diff (`automodel.base` +
`automodel.patch`), not a free-form two-script ZIP. Pods default to **1×
NVIDIA B200** (CUDA 13 / Transformer Engine). Caps: **850M–1B params**,
**3.0e18 attested FLOPs** + wall from `/v1/recipe`. Leaf scoring is **G2
benchmarks** (`scoring_version` 4). Prior recipe 2.0 harvests cannot win.

| Page | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | Fork pin → edit → `git diff` → pack ZIP |
| [Submit](submit.md) | ZIP/JSON, BYOK Lium key, gating, precheck, retries |
| [Scoring & competition](scoring.md) | bpb lattice, patch anti-copy, causal ban, top-model |
| [API](api.md) | Routes, statuses, diff + telemetry |
| [Troubleshooting](troubleshooting.md) | `unsupported_layout`, pin/patch failures, Lium |
| [Full guide](prism.md) | Complete miner guide (mirrors BASE `docs/external-miner/prism.md`) |

Normative sources (BASE monorepo):
[`docs/PRISM.md`](https://github.com/BaseIntelligence/base/blob/main/docs/PRISM.md),
[`docs/PRISM_RECIPE.md`](https://github.com/BaseIntelligence/base/blob/main/docs/PRISM_RECIPE.md),
[`docs/external-miner/prism.md`](https://github.com/BaseIntelligence/base/blob/main/docs/external-miner/prism.md).
