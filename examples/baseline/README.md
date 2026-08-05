# PRISM baseline recipe (example)

The official baseline submission — a tiny GPT-style causal transformer (~12M params)
plus an AdamW training loop that demonstrates the **required telemetry hooks**.

Starting from this baseline is always allowed (the anti-copy gate exempts it). To be
competitive, ship your **own** architecture — see
[scoring & competition](../../docs/scoring.md).

## Files

| File | Role |
|------|------|
| `architecture.py` | `build_model(ctx)` → TinyGPT (tied embeddings, causal mask, block=512) |
| `training.py` | `train(model, ctx)` → AdamW loop with `prism_telemetry.report(...)` + `finish_evaluation()` |

## Hook pattern (required since recipe 1.1.0)

`training.py` imports `prism_telemetry` with a local fallback stub so you can also run
it outside the operator harness:

```python
try:
    import prism_telemetry
except ImportError:
    class _TelemetryFallback:
        @staticmethod
        def report(**_kwargs): ...
        @staticmethod
        def finish_evaluation(): ...
    prism_telemetry = _TelemetryFallback()
```

Inside the operator harness the real module captures your series into
`METRICS_JSON.telemetry.loss_series` (persisted master-side; served at
`/v1/site/arenas/prism/submissions/{id}/telemetry`).

- `prism_telemetry.report(loss=..., step=..., grad_norm=..., layer_stats=...)` — call
  every N steps.
- `prism_telemetry.finish_evaluation()` — optional early stop; the harness scores the
  in-memory model as-is. It raises a `BaseException` through `train()`, so it cannot be
  swallowed by your own `except Exception` blocks.

## Submit

```bash
zip -j submission.zip architecture.py training.py
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

The same sources are always available live at `GET /v1/recipe/baseline`.
