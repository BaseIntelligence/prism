# Scoring & competition

## Live leaf: G2 benchmarks (`scoring_version` 4)

Leaf score = **equal-weight mean of available G2 public accuracies**
(HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, BoolQ, LAMBADA,
OpenBookQA when present) mapped to `round(SCORE_MAX × mean)` on the integer
lattice `[0, SCORE_MAX]`.

Bits/token bpb is still measured (display / G1) but **does not** farm emission
under the default `PRISM_SCORING_MODE=benchmarks`. Tokenizer length cannot
game the rank. LLM reviews remain **gates, not graders**.

Legacy: `PRISM_SCORING_MODE=shadow` restores pure bits/token bpb (v2);
`composite` uses the full G1–G8 lattice when anchors are ready.

## Anti-copy (patch / delta)

Copying another miner's **patch** (or an equivalent touched-file rewrite of
an earlier champion delta) is terminal `rejected` with zero score — judged
before or without burning GPU when the gate can decide from the diff alone.
Review focuses on your unified diff and touched files (`arch` / `trainer` /
`data` / `other`), not the whole AutoModel tree. Starting from the operator
pin and submitting only your delta is the intended path.

After the gate, an LLM similarity review + agentic anti-cheat still run:
`Copied` / high-confidence `Suspicious` (≥ 0.9 with non-generic evidence) /
`cheat` → hard zero. Standard components (RMSNorm, RoPE, SwiGLU, LayerNorm,
…) are **not** plagiarism signals.

Probe the cheap gate first with `POST /v1/submissions/precheck` (3/coldkey/UTC
day) — see [Submit](submit.md#precheck-similarity-before-you-submit).

## Causal LM contract (banned: non-causal label leak)

Prism scores **next-token** cross-entropy → BPB. Architectures must not let
position `t` read tokens `t+1…` (including the label). Dense sequence mixers —
MLP-Mixer-style `TokenMix` / `t_mix` / `nn.Linear` over the full time axis after
`transpose(1, 2)` — **without** a causal mask (`triu` / `tril` / `is_causal` /
attention mask) are a hard ban (`non_causal_label_leak`, `Score(0)`, terminal,
often caught **before** GPU rent). Channel mixing and causal attention / causal
conv are fine; bidirectional full-sequence mixes used as a next-token LM are not.

## Competition (emission)

**Competition (live default):** emission is **winner-take-all** on
weight-eligible AutoModel rows (`PRISM_EMISSION_MODE=wta`). Architecture-owner
credit and `top3` / significance-gated modes are **implemented but default-off**
— do not assume they are live. Only the single highest own score that epoch
receives Prism's share (50% of the subnet); ties break by lexicographically
smallest hotkey.

Scores first land in the leaf set emitted at the first chain-epoch boundary
**after** your run finalizes. Positive scores then keep participating in later
epochs' competition sets until a better valid score supersedes them (WTA still
collapses to one leaf winner).

## Top-model publish

Whenever a new **global-best scored run** lands, the master publishes the winning
sources + `ARTIFACT.json` / checkpoint release to
[`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism) under
[`top-model/`](https://github.com/BaseIntelligence/prism/tree/main/top-model)
and journals the publication. The `top-model/` directory always mirrors the
current champion; history lives in git.

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
