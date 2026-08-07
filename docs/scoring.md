# Scoring & competition

## Pure bpb

`final_score = score_from_bpb(measured_bpb)` on the integer lattice `[0, SCORE_MAX]` —
lower bpb, higher score. The LLM reviews are **gates, not graders**: they verify the
submission is coherent and not cheating; their quality notes never move the score.

## v3 composite (shadow-by-default)

Since recipe 1.3.0 every run is *also* measured on the organizer-run **G1–G8 battery**
(Zone A — computed by the harness, never by your code):

| Group | Axis | Weight |
|-------|------|--------|
| G1 | intrinsic fit (frozen-val + multi-domain/fresh-crawl bpb) | 0.25 |
| G2 | commonsense/reading 0-shot core | 0.15 |
| G3 | retrieval/associative recall (gated ≥ 0.25) | 0.10 |
| G4 | reasoning at small scale | 0.15 |
| G5 | long-context | 0.15 |
| G6 | sample efficiency (train probe curve) | 0.075 |
| G7 | inference efficiency | 0.075 |
| G8 | training stability + µP (gated ≥ 0.5) | 0.05 |

The battery runs in a **two-phase pod flow**: training completes first, then the
operator stages private eval assets (held-out + fresh-crawl data) and a fresh eval
process measures the model. While `PRISM_SCORING_MODE=shadow` (default) your leaf score
stays **pure bpb, bit-identical to v2**. After the reference baselines
(**Transformer++**, **hybrid delta**) are measured and the anchor set is
pre-registered, governance may flip to `composite`: anchor-normalized group scores,
gates, a weighted geometric mean, and bootstrap lower-confidence-bound ranking
(`lattice = round(SCORE_MAX × max(0, C − 1.645·SE))`, `scoring_version 3`).

**What the harness reports (METRICS_JSON v2).** Every v1 key (`bpb`, `tokens_seen`,
`wall_clock_seconds`, `gpu_type`, `n_params`, `telemetry`, …) plus the v3 blocks:
`flow`, `eval_tier` (`"private"` | `"public_dev"`), `gate`, `probe_curve` (G6),
`train_metrics` (your Zone B dict, sanitized, never scored), and `battery`. The
battery's canonical surface is `battery.metrics` — a **flat** map of
`org.<group>.<name>` keys to a bare float or `{value, clusters}` (`clusters` are
per-template means, the units of randomization for the clustered bootstrap). A metric
that was never measured is **absent, never fabricated**. `battery.mirrors` carries the
contamination-gap pairs for G2/G4: the same metric scored on the public dev-seed asset
family vs a private mirror family — in the `public_dev` tier no private assets exist,
so each pair is degenerate (gap 0, honestly labelled).

Your `train()` return dict lands in **Zone B** (`miner.*` keys): displayed but labelled
participant-reported, validated at ingest, **never scored**. Never emit `org.*` keys —
that quarantines the report as anti-cheat evidence. You can also post additional
self-reports out-of-band:

```
POST /v1/submissions/{id}/zone-b
```

```json
{
  "schema_version": "<recipe version>",
  "prev_hash": "<previous report_hash — optional>",
  "metrics": {
    "miner.<group>.<name>": {"kind": "scalar | series | histogram"}
  }
}
```

Reports chain per submission (`prev_hash` → the previous `report_hash`; omit it for
master-chained ingest) and are capped at 64 scalars / 16 series / 10 000 points / 1 MB.
Each report is validated against organizer ground truth (token/step/wall-clock
counters, MFU ceiling, terminal-loss band) and the cross-miner cohort, and lands a
stored verdict — `ok` / `flagged` / `quarantined`. Verdicts are evidence, never an
auto-zero. Malformed or over-cap envelopes reject `422` and store nothing.

Per-run rows: `GET /v1/submissions/{id}/metrics?zone=a|b`; anchor registry:
`GET /v1/anchors` and `GET /v1/preregistration`.

Kernel-carrying source trees can be decomposed with the 2×2 **attribution** planner
(`POST /v1/submissions/{id}/attribution`): your architecture on reference kernels vs
the reference architecture on your kernels, isolating arch vs kernel contributions.

Note: a model over the **350M parameter cap** is now a terminal `Score(0)`
(`CAP_EXCEEDED`), not a retryable failure.

## Anti-copy (architecture-only)

- A **pre-LLM copy gate** compares your `architecture.py` against earlier submissions
  (byte hash + AST fingerprints, `created_at` ordered). A byte/AST copy of a
  strictly-earlier architecture is terminal `rejected` with zero score — **no GPU time,
  no LLM review**, no appeal. The published baseline is exempt.
- Similarity is judged on `architecture.py` **only**: `training.py` is exempt on both
  sides — the same training loop on two different architectures is legitimate, and
  training-only entries on a published arch are never "copies" by construction.
- After the gate, an LLM similarity review + agentic anti-cheat still run:
  `Copied` / `Suspicious` / `cheat` verdicts → hard zero.

## Architecture competition (emission math)

Per epoch, your emission is the **max** of:

1. **Challenger credit** — your own best training result this epoch (any arch), and
2. **Owner credit** — for each architecture you own, that arch's **best result by any
   trainer** this epoch.

Max, never summed — architecture owners are rewarded when *anyone* trains well on
their architecture. `Score(0)` rows (cheat / copy-gate) never set an arch's best.

Published architectures and their best bpb so far: `GET /v1/architectures`.

## Top-model publish

Whenever a new **global-best bpb** lands, the master publishes the winning
`architecture.py` + `training.py` + `METRICS.json` to
[`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism) under
[`top-model/`](https://github.com/BaseIntelligence/prism/tree/main/top-model) and
journals the publication. The `top-model/` directory always mirrors the current
champion; history lives in git.

## Telemetry

Your `prism_telemetry.report(...)` series (loss, gradient norms, per-layer stats) is
persisted master-side and served on the site:

```
GET /v1/site/arenas/prism/submissions/{id}/telemetry
```

`finish_evaluation()` (early stop) is recorded as the eval's finish reason — scoring
uses the model as-is at that point, before any cap fires.

## Weights

Leaves per epoch feed the BASE gateway seal (`/v1/weights/latest`); prism's emission
share is owner-controlled via the trust root. Miners never write on-chain weights.

## Next

→ [API](api.md)
