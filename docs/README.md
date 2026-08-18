# PRISM miner docs

Live recipe is **2.1.0**: submit an AutoModel pin id + unified git diff
(`automodel.base` + `automodel.patch`), not a free-form two-script ZIP.

| Page | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | Fork pin → edit → `git diff` → pack ZIP |
| [Submit](submit.md) | ZIP/JSON, BYOK Lium key, gating, precheck, retries |
| [Scoring & competition](scoring.md) | G2 lattice, G1–G8 battery, emission, anti-copy |
| [API](api.md) | Routes, statuses, diff + telemetry |
| [Troubleshooting](troubleshooting.md) | `unsupported_layout`, pin/patch failures, Lium |
| [Full guide](prism.md) | Complete miner guide (mirrors BASE `docs/external-miner/prism.md`) |

Normative sources (BASE monorepo):
[`docs/PRISM.md`](https://github.com/BaseIntelligence/base/blob/main/docs/PRISM.md),
[`docs/PRISM_RECIPE.md`](https://github.com/BaseIntelligence/base/blob/main/docs/PRISM_RECIPE.md),
[`docs/external-miner/prism.md`](https://github.com/BaseIntelligence/base/blob/main/docs/external-miner/prism.md).
