# Scoring & competition

## Pure bpb

`final_score = score_from_bpb(measured_bpb)` on the integer lattice `[0, SCORE_MAX]` —
lower bpb, higher score. The LLM reviews are **gates, not graders**: they verify the
submission is coherent and not cheating; their quality notes never move the score.

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

**Competition (temporary):** emission uses **your own best training score
only** — architecture-owner credit (rewarding arch owners when others train
well on their code) is **disabled** for now so the best-BPB trainer keeps
Prism's weights. Emission remains **winner-take-all**: only the single highest
own score that epoch receives Prism's share (50% of the subnet); ties break by
lexicographically smallest hotkey.

Scores first land in the leaf set emitted at the first chain-epoch boundary
**after** your run finalizes. Positive scores then keep participating in later
epochs' competition sets until a better valid score supersedes them (WTA still
collapses to one leaf winner).

## Top-model publish

Whenever a new **global-best bpb** lands, the master publishes the winning
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
