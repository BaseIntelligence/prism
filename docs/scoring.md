# Scoring & competition

## Pure bpb

`final_score = score_from_bpb(measured_bpb)` on the integer lattice `[0, SCORE_MAX]` —
lower bpb, higher score. The LLM reviews are **gates, not graders**: they verify the
submission is coherent and not cheating; their quality notes never move the score.

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
- Probe the cheap gate first with `POST /v1/submissions/precheck` (3/coldkey/UTC
  day) — see [Submit](submit.md#precheck-similarity-before-you-submit).


## Causal LM contract (banned: non-causal label leak)

Prism scores **next-token** cross-entropy → BPB. Architectures must not let
position `t` read tokens `t+1…` (including the label). Dense sequence mixers —
MLP-Mixer-style `TokenMix` / `t_mix` / `nn.Linear` over the full time axis after
`transpose(1, 2)` — **without** a causal mask (`triu` / `tril` / `is_causal` /
attention mask) are a hard ban (`non_causal_label_leak`, `Score(0)`, terminal,
often caught **before** GPU rent). Channel mixing and causal attention / causal
conv are fine; bidirectional full-sequence mixes used as a next-token LM are not.

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
