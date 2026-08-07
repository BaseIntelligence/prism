# PRISM miner docs

| Page | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | Contract, tokenizer, source-tree, hooks, dataset, budgets |
| [Submit](submit.md) | ZIP/JSON/source-tree submit, gating, retries, training-only |
| [Scoring & competition](scoring.md) | bpb lattice, v3 G1–G8 (G5 pretrain-only), anti-copy, top-model |
| [API](api.md) | Routes, statuses, telemetry, Zone B / attribution |
| [Troubleshooting](troubleshooting.md) | Common failures |
| [`examples/baseline/`](../examples/baseline/) | Reference recipe with hooks + `ctx["tokenizer"]` |

**Recipe:** `1.4.0` (miner-chosen tokenizer; G5 = RULER + BABILong + natural
docs, pretrain-only). Normative freeze lives in the BASE monorepo:
`docs/PRISM.md`, `docs/PRISM_RECIPE.md`, `docs/external-miner/prism.md`.
