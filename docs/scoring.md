# Scoring & competition

## Pure bpb (live leaf)

`final_score = score_from_bpb(measured_bpb)` on the integer lattice `[0, SCORE_MAX]` —
lower bpb, higher score. The LLM reviews are **gates, not graders**: they verify the
submission is coherent and not cheating; their quality notes never move the score.

**Fairness across tokenizers.** `bits_per_byte` (bits over UTF-8 bytes of the scored
region) is the tokenizer-neutral anchor reported in `METRICS_JSON`. The legacy `bpb`
key is bits per *token* and is only comparable across submissions that share a
tokenizer.

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
their architecture. `Score(0)` rows (cheat / copy-gate / `CAP_EXCEEDED`) never set an
arch's best.

Published architectures and their best bpb so far: `GET /v1/architectures`.

## v3 scoring (shadow-by-default)

Recipe ≥ 1.3.0 harnesses run a **two-phase** pod flow: your code trains
(`phase=train`), checkpoints, then a fresh eval subprocess runs frozen-val bpb plus
the **G1–G8 battery** (intrinsic fit, commonsense/reading, retrieval/recall,
reasoning, long-context, sample efficiency, inference efficiency, training
stability/µP). Battery metrics are organizer-measured (**Zone A**, `org.*`) — your
code never emits them.

While scoring mode is `shadow` (default), the **leaf score stays pure bpb**,
bit-identical to v2. After reference baselines are measured and anchors
pre-registered, governance may flip to `composite`. Inspect anchors at
`GET /v1/anchors` and `GET /v1/preregistration`; per-run rows at
`GET /v1/submissions/{id}/metrics?zone=a|b`.

### G5 long-context (recipe ≥ 1.4.0 — pretrain-only)

G5 scores a **base LM**, not an instruction-tuned chat model: completion-style /
few-shot base prompts, short exact-match or multiple-choice logprob — **no** chat
templates, free-form summarization, or LLM-as-judge on the ranked path. Length
targets are counted in tokens of **your** tokenizer (`ctx["tokenizer"]`).

Scored keys (group weight 0.15 total):

| Key | Weight |
|-----|--------|
| `org.g5.ruler_acc` | 0.35 |
| `org.g5.babilong_acc` | 0.25 |
| `org.g5.natural_mcq_acc` | 0.15 |
| `org.g5.helmet_rag_acc` | 0.15 |
| `org.g5.lstar` | 0.10 |

**L\*** is the highest length where pooled RULER+BABILong accuracy stays ≥ 90% of the
shortest-grid accuracy and ≥ 0.25 (else 0). Natural MCQ / HELMET RAG packs are
mirrored like G2/G4.

### Zone B (self-report, never scored)

Your `train()` return dict (`train_metrics` in `METRICS_JSON` v2) is **Zone B**:
participant-reported, displayed-but-labelled, validated at ingest, and **never
scored**. Do not emit `org.*` keys. Optional out-of-band reports:
`POST /v1/submissions/{id}/zone-b`.

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
Scores land in the leaf set emitted at the first chain-epoch boundary **after** your
run finalizes.

## Next

→ [API](api.md)
