# Historical recipe 1.x baseline (not accepted on live)

> **Live recipe is 2.1.0.** Submitting `architecture.py` + `training.py` returns
> `400 unsupported_layout` / `recipe_version` on production. Use the AutoModel
> patch workflow instead — see [Getting started](../../docs/getting-started.md)
> and the [full guide](../../docs/prism.md).

This directory keeps the old two-script baseline for reference only (tiny
GPT-style causal transformer + AdamW loop with telemetry hooks). It is **not**
a valid live submission under recipe 2.1.

## Live path (recipe 2.1)

1. `GET /v1/recipe` → copy `automodel_pin_id` (`automodel@v0.5.0`) and
   `automodel_git_commit`.
2. Checkout that AutoModel commit, edit, `git diff <commit> > automodel.patch`.
3. Pack `automodel.base` + `automodel.patch` (+ optional `prism.toml`).
4. `POST /v1/submissions` with `X-Miner-Hotkey` + **`X-Lium-Api-Key`**.

## Files (legacy 1.x)

| File | Role |
|------|------|
| `architecture.py` | `build_model(ctx)` → TinyGPT (historical) |
| `training.py` | `train(model, ctx)` → AdamW + `prism_telemetry` hooks (historical) |

Telemetry hooks remain required under the AutoModel train entry on live 2.1 —
patches that remove them fail with `missing_telemetry_hooks`.
