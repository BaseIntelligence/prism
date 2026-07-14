# Prism Official Comparison Protocol v1

**Version label:** `prism_official_compare.v1`  
**Document status:** Operator- and miner-facing protocol (docs + tests first)  
**Audience:** Operators running offline A-vs-B architecture/training comparisons; miners building unknown architectures under fair matched budgets  

This document defines **Prism Official Comparison Protocol v1**: an architecture-agnostic protocol for ranking any two qualifying training submissions (or seed packages) under matched data, tokens/bytes, seeds, and fairness constraints. It is the scientific lab/compare surface. It is **not** a rewrite of the live emission leaderboard and it never claims live **REAL-PROVIDER TEE PASS**.

For production subnet scoring (leaderboard `final_score`, raw-weight push), see [Scoring](scoring.md). That path still uses prequential bits-per-byte as the **leaderboard primary** with held-out as a **bounded near-tie**. **Official Comparison mode inverts the ranking axes** as specified below.

---

## 1. Purpose

Official Comparison answers:

> Under one pinned FineWeb-Edu revision, forced random init, challenge-owned single-pass batches, matched token/byte budgets, and multi-seed residual, which of two unknown architectures (or training recipes) generalizes better, with prequential compression as the secondary signal?

Goals:

1. Fair compare of **unknown architectures** expressible under the AST sandbox and 150M param cap (Transformer, Mamba/SSM pure-torch, and any other `nn.Module` that meets the forward contract).
2. Prefer **held-out generalization** as the ranking primary so train-only compression or memorization does not win headlines.
3. Keep **prequential bpb** as a always-Prism-recomputed secondary signal.
4. Keep wall-clock, miner self-reports, and REAL-PROVIDER TEE labels out of the ranking key.
5. Document that real multi-step GPU pair trains are **deferred** on hosts without NVIDIA (this protocol still allows CPU/fixture unit compares).

Non-goals in v1:

- Replacing subnet emission two-tier ownership (architecture 0.60 / training 0.40).
- Using lm-eval / GSM8K / MMLU as primary fair learning signals for from-scratch compare.
- Inventing Lium/Targon trust roots or labeling Lium deploy smoke as REAL-PROVIDER PASS.

---

## 2. Catalogue of terms

| Term | Meaning |
| --- | --- |
| **Leaderboard scoring** | Production path: prequential bpb drives `final_score`; held-out is bounded near-tie; used for subnet ranking and raw weights. |
| **Official Comparison mode** | Offline/lab pair ranking under this protocol: **held-out/generalization primary**, prequential bpb secondary. |
| **Protocol pin** | Frozen JSON-ish knobs (dataset pin, tokenizer, budgets, seeds, device class). Both sides must run under the same pin hash. |
| **Unknown architecture** | Any `torch.nn.Module` built by `build_model(ctx)` under sandbox and param cap; no family-specific score path. |
| **Miner self-report** | Any number, log line, or manifest the miner writes; **never authoritative**. |
| **Challenge-owned metric** | Metrics Prism recomputes from its own capture / host held-out (`prism_run_manifest.v2.json`). |
| **REAL-PROVIDER TEE** | Cryptographic provider attestation PASS. Orthogonal to Official Comparison; still **BLOCKED** until external provider contracts exist. See [Security](security.md). |

---

## 3. Modes

| Mode | Frozen | Free variable | Headline use |
| --- | --- | --- | --- |
| **ArchCompare** (default official headline) | Same `training.py` (hash), protocol pin, tokenizer, seeds | `architecture.py` only | Is architecture X a better learner than Y under matched train recipe? |
| **TrainCompare** | Same `architecture.py`, pin, seeds | `training.py` only | Is recipe R better on fixed architecture? |
| **SystemCompare** | Only the protocol pin | Both scripts | Closest to open miner battle; weaker scientific isolation |

Live subnet submission remains SystemCompare-shaped; Official Comparison **ArchCompare** is the recommended scientific claim.

---

## 4. Fixed fairness constraints (normative)

Every Official Comparison pair member MUST satisfy:

1. **Matched token/byte budget** — both sides bound primarily by the same `token_budget` (and optional identical `step_budget`). Prefer `stopped_reason == "token_budget"`. If either side is bound by wall-clock alone, the pair run is **invalid** for official ranking.
2. **Matched byte measure** — primary denominators remain UTF-8 **bytes** on the locked stream (tokenizer-agnostic compression measure). Protocol may additionally force a referee tokenizer so packing is comparable.
3. **Forced random init** — challenge seed family applied before miner import; `ctx.seed` immutable; bootstrap / resume checkpoints forbidden for official inject.
4. **Challenge-owned batches** — training sinks via `ctx.iter_train_batches(...)` (single-pass instrument; predict-then-train capture). Private re-loaders that bypass the instrument disqualify.
5. **150M parameter cap** — `max_parameters = 150_000_000`; realized param count rechecked after first forward on the scored model.
6. **Wall-clock never primary** — wall-clock is a safety watchdog only. It never enters the ranking key and must not silently become the effective sample-size limit for a fair pair.
7. **Secret val/test** — never mounted into the train container; host-only held-out.
8. **Miner self-report never authoritative** — miner-written manifests, self-reported bpb, wall-time claims, and free-form diagnostics cannot certify official rank.
9. **Matched multi-seed residual** — official claims use the same seed list K≥3 for both sides (default seeds documented below). Rank residual uses multi-seed aggregate, not a cherry-picked seed.
10. **Sandbox purity** — pure Python/torch surface; no network, no custom blocked CUDA plugin imports.
11. **Scored path nproc** — default official pin uses `scored_nproc = 1` for both; multi-GPU only if both sides run under the same world size pin.

---

## 5. Ranking axes (Official Comparison mode)

### 5.1 PRIMARY: held-out / generalization

Primary ranking prefers **held-out generalization** quality.

Documented primary forms (use one consistently per pin; higher is better for delta form, lower is better for absolute val bpb form):

| Form | Direction | Notes |
| --- | --- | --- |
| **`heldout_delta`** (preferred) | **higher better** | `bpb(random-init twin on val) − bpb(trained on val)` on the secret val prefix |
| **`val_bpb_trained`** (alternate pin option) | **lower better** | Absolute held-out free energy when twin delta is unavailable |

“Honestly lower held-out free energy” (or larger honest improvement over the random-init twin) ranks higher. A train-only memorizer must not win on primary.

When multi-seed:

\[
\text{primary}_S = \frac{1}{K}\sum_{k=1}^{K} \text{heldout\_delta}(S; \mathrm{seed}_k)
\quad\text{(higher better)}
\]

(or the documented alternate mean of `val_bpb_trained`, lower better).

### 5.2 SECONDARY: prequential bits-per-byte

**Prequential bpb** remains the architecture-agnostic secondary signal and is **always recomputed by Prism** from challenge-owned run data:

```text
bpb = (sum_neg_log_likelihood_nats / ln(2)) / covered_bytes
```

Lower bpb is better. Multi-seed:

\[
\text{secondary}_S = \frac{1}{K}\sum_{k=1}^{K} \mathrm{bpb}(S; \mathrm{seed}_k)
\]

Miner-reported train metrics **cannot** certify secondary rank. Production `final_score = 1/(1+bpb)` (before penalty/tie fold) may be reported for continuity with the leaderboard transform, but Official Comparison scientific ranking keys secondary as raw recomputed bpb first.

### 5.3 Anti-overfit gates (always active)

| Gate | Threshold / rule | Effect under Official Comparison |
| --- | --- | --- |
| **Memorization gap** | `gap > 1.0` bpb (`MEMORIZATION_GAP_THRESHOLD_BPB`) using converged train vs val bpb when bases comparable | Flag + ×0.5 penalty on score surface; higher overfit rate loses tertiary rank |
| **Step-0 anomaly** | `step0_loss < 0.5 * ln(vocab)` (`STEP0_ANOMALY_FRACTION`) | Anti-cheat multiplier → 0; that seed run disqualified / final surface zeroed |
| **No-learning** | Late loss ~ baseline on worker plausibility path | Disqualify run |
| **Degenerate bpb** | non-positive / non-finite / empty coverage / insane band | Fail closed, not ranked |

These gates prevent pure train memorizers and smuggled-weight runs from winning on secondary compression looks.

### 5.4 Wall-clock

Wall-clock seconds may appear in observability `compute` blocks and lab diagnostics. **Wall-clock never ranks.** Two records with identical quality metrics and different `wall_clock_seconds` must compare equal on the official key.

### 5.5 Deterministic compare rule `compare_official(A, B)`

Given two official multi-seed aggregates (or single-seed records when residual multi-seed is not yet filled):

```text
ε_h  = 5e-3     # primary near-tie band (heldout_delta or same units as pin primary)
ε_b  = 5e-3     # secondary bpb near-tie band (reuses HELDOUT_DELTA_BPB_EPSILON scale)
K    = len(seeds)  # ≥ 1 for unit fixtures; ≥ 3 for official public claims

1) Validity: both sides pass stop_reason / finite / no step0 on included seeds.
2) PRIMARY held-out:
   if primary(A) better than primary(B) by > ε_h → A
   if primary(B) better than primary(A) by > ε_h → B
3) SECONDARY prequential bpb (lower better):
   if secondary(A) < secondary(B) − ε_b → A
   if secondary(B) < secondary(A) − ε_b → B
4) Anti-overfit residual:
   lower memorization/overfit fraction wins; if step0/disqualify asymmetric, the clean side wins
5) Multi-seed residual (when K>1 and still tied):
   lower variance / better median secondary among clean seeds; else TIE
6) Never consult wall_clock or miner self-report
```

“Better” for primary:

- if pin uses `heldout_delta`: larger mean is better  
- if pin uses `val_bpb_trained`: smaller mean is better  

Implementation helps land as a pure deterministic helper (fixture-tested) in the scoring track; this document is the normative order those helpers must implement.

### 5.6 Explicit invert vs leaderboard narrative

| Axis | Production leaderboard ([scoring.md](scoring.md)) | Official Comparison Protocol v1 (this document) |
| --- | --- | --- |
| Rank key 1 | Prequential bpb → `final_score` | **Held-out generalization** |
| Rank key 2 | Bounded held-out delta (near-tie only) | **Prequential bpb** (Prism-recomputed) |
| Anti-overfit / step-0 | Active | Active (unchanged semantics) |
| Wall-clock | Not scored | Not ranked |
| Emission weights | Derived from leaderboard path | **Not** driven by Official Comparison labeling |

Do not treat “bpb primary” language in production scoring docs as applying to Official Comparison headline ranking.

---

## 6. Protocol pin (`ProtocolPin`)

Official pairs share one pin object. Operators SHOULD record a content hash of the pin into the compare report provenance.

```yaml
protocol_id: prism_official_compare.v1
data:
  dataset: HuggingFaceFW/fineweb-edu
  pin_sha: "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"  # must match product dataset.py
  split: train
  partition: as locked MANIFEST.json buckets
tokenizer: gpt2                 # forced offline reference for ArchCompare packing hygiene
vocab_size: 50304
seq_len: 128                    # lab default; raise on GPU hosts when budgets allow
batch_size: 4
token_budget: 500000            # must bind both sides before wall_clock
step_budget: null
wall_clock_seconds: 1200        # safety only; fair-pair invalid if stop_reason is wall_clock alone
seeds: [1337, 2027, 4242]       # K=3 minimum for public official claims
param_cap: 150000000
scored_nproc: 1
device: cuda                    # or cpu for fixture / short CPU seeds
force_iter_train_batches: true
require_trained_state: true
heldout:
  val_byte_budget: 65536
  gap_threshold_bpb: 1.0
primary_form: heldout_delta     # or val_bpb_trained
```

**Hard validity for each pair member:**

1. Manifest is challenge-authored; `miner_reported_ignored: true`.
2. `stopped_reason` is `token_budget` (preferred) or both sides data-exhausted at equal covered_bytes under the same pin.
3. Finite positive bpb in-band; no step-0 anomaly on included seeds.
4. Realized `model_params ≤ param_cap`.
5. Seed list applied identically (matched residual).
6. Held-out present for official master/host path; worker `skip_heldout` outputs are not official compare witnesses alone.

Lab seeds that share zip packaging:

- Transformer: `examples/tiny-1m` (`transformer-tiny-1m`)
- Mamba pure-torch: `examples/mamba-tiny` (`mamba-tiny-1m`)

Package via `scripts/pack_seed_family.py` / `prism_challenge.seed_packaging`.

---

## 7. Training-script honest hooks contract

Official Comparison reuses the two-script product contract ([Submissions](submissions.md)). The honest recovery surface:

### 7.1 Required hooks

| Hook | Script | Contract |
| --- | --- | --- |
| `build_model(ctx) -> nn.Module` | `architecture.py` | Pure factory. No data reads, no network, no files. Param count ≤ 150M under forced seed. Logits contract: `forward(tokens) → logits[B,T,V]` (or `.logits` / first tuple). |
| `train(ctx) -> None` | `training.py` | Owns optimizer, schedule, multi-GPU, loop. Builds via `build_model` / `ctx.build_model()`. **Consumes challenge batches only** through `ctx.iter_train_batches(...)`. Return values are ignored for score. |

### 7.2 Challenge fields miners must honor

| Field / method | Role |
| --- | --- |
| `ctx.seed` | Forced, immutable honest init seed |
| `ctx.data_dir` | Read-only train split path |
| `ctx.artifacts_dir` | Sole writable root (rank 0) |
| `ctx.token_budget` / `ctx.step_budget` | Binding single-pass caps for fair pin |
| `ctx.iter_train_batches(model, batch_size=, seq_len=, tokenizer=)` | **Required** instrument for honest leak capture |
| `ctx.build_model()` | Force-seeded factory invocation |
| `ctx.reference_tokenizer("gpt2" \| "llama")` | Offline staged; no network |
| Optional `artifacts_dir/trained_state.pt` | Host held-out needs state when pin requires it |

### 7.3 Optional non-authoritative diagnostics

Miners MAY log progress strings, publish extra tensors under `artifacts_dir`, or attach free-form diagnostics for human debugging.

Those diagnostics are:

- allowed to be **recorded**,
- **never** used as the official rank key,
- superseded by Prism recomputation of bpb, held-out, memorization flags, and step-0.

### 7.4 Forbidden trust roots

| Forbidden | Why |
| --- | --- |
| Miner self-reported `final_score` / bpb as authority | Score is challenge-owned only |
| Miner-written `prism_run_manifest*.json` | Challenge authors `prism_run_manifest.v2.json` |
| Pretrained weight smuggle / resume capable of surviving forced init claims | Step-0 anomaly + forced seed |
| Private loaders that skip `iter_train_batches` under official pin | Breaks single-pass capture honesty |
| Ranking by wall-clock or GPU count | Explicitly out of ranking key |
| Claiming REAL-PROVIDER TEE PASS from compare docs | TEE decision is orthogonal; real provider still blocked |

Sketch (honest `train`):

```python
def train(ctx):
    model = ctx.build_model().to(ctx.device)
    opt = ...  # miner-owned
    tok = ctx.reference_tokenizer("gpt2")  # protocol pin tokenizer for ArchCompare
    for batch in ctx.iter_train_batches(
        model, batch_size=4, seq_len=128, tokenizer=tok
    ):
        logits = model(batch.tokens)
        loss = next_token_ce(logits, batch.tokens)
        loss.backward()
        opt.step()
    if ctx.rank == 0 and ctx.artifacts_dir:
        torch.save(model.state_dict(), f"{ctx.artifacts_dir}/trained_state.pt")
```

---

## 8. GPU verification deferred (host without NVIDIA)

Official Comparison **does not require** a GPU host for protocol docs, unit fixtures, or CPU/synthetic dual-family compare harnesses.

### 8.1 Environment fact (this mission host, 2026-07-14)

| Probe | Result |
| --- | --- |
| `nvidia-smi` | Not found |
| `/dev/nvidia*` | Absent |
| Docker runtimes | `runc` / `runsc` — **no nvidia runtime** |
| GPU device requesting containers | Cannot select device driver with GPU capabilities |

Therefore:

- **Real multi-step GPU official smoke is DEFERRED** on this host and any host with the same absence.
- Workers MUST NOT claim GPU verification PASS, “NVIDIA validated,” or REAL-PROVIDER GPU TEE PASS from this environment.
- CPU unit fixtures, synthetic score records, and short seed packaging tests remain valid Protocol surfaces (`device: cpu` pin or fixture manifests).

### 8.2 When GPU becomes available

On a host with working NVIDIA + Docker GPU runtime:

1. Run both sides under the same ProtocolPin with `device: cuda` and `token_budget` large enough that both stop on `token_budget`, not wall-clock.
2. Capture challenge-authored manifests per seed, pin hash, and `compare_official` outcome.
3. Still label TEE honestly: LOCAL-FIXTURE vs blocked REAL-PROVIDER paths independently of the scientific ranking.

---

## 9. TEE honesty is orthogonal

Official Comparison ranking compares **learning quality metrics** only.

| Classification | Relation to official rank |
| --- | --- |
| No TEE / legacy path | May still produce lab metric records if product mode allows; does not improve rank |
| `LOCAL-FIXTURE PASS` | Elevated local crypto class for verifier honesty only; **never** REAL-PROVIDER |
| Lium / Targon deploy smoke | Infra `DEPLOY SMOKE PASS|FAIL` only |
| `REAL-PROVIDER TEE PASS` | **BLOCKED** until authoritative evidence contracts, trust roots, digests exist; **not** unlocked by Official Comparison |

Protocol docs and compare reports MUST keep a clean separation paragraph: scoring comparison never claims REAL-PROVIDER PASS. Fail-closed TEE-required production scoring (when enabled) remains a product gate independent of this ranking table.

---

## 10. Report sketch (`prism_compare_report.v1`)

Offline harnesses SHOULD emit a stable report (JSON) shaped like:

```json
{
  "schema": "prism_compare_report.v1",
  "protocol_id": "prism_official_compare.v1",
  "protocol_hash": "...",
  "mode": "ArchCompare",
  "primary_form": "heldout_delta",
  "side_a": {"label": "transformer-tiny-1m", "bundle_hash": "..."},
  "side_b": {"label": "mamba-tiny-1m", "bundle_hash": "..."},
  "seeds": [1337, 2027, 4242],
  "aggregate": {
    "a": {
      "mean_heldout_delta": 0.0,
      "mean_bpb": 0.0,
      "std_bpb": 0.0,
      "overfit_rate": 0.0
    },
    "b": {}
  },
  "ranking": {
    "winner": "a|b|tie",
    "rule": "heldout_primary_then_bpb_secondary",
    "eps_heldout": 0.005,
    "eps_bpb": 0.005
  },
  "validity": {"ok": true, "reasons": []},
  "tee_note": "orthogonal; REAL-PROVIDER PASS not claimed"
}
```

Unit fixtures may fill synthetic metrics without multi-hour train. Live emission weights are out of this report.

---

## 11. Operator checklist

1. Choose mode (ArchCompare recommended).
2. Freeze ProtocolPin; write pin hash into evidence.
3. Package both sides under two-script zip contract and 150M/sandbox gates.
4. Run matched seeds; require challenge-authored manifests.
5. Verify validity (token_budget stop, no step0, held-out present for official claims).
6. Aggregate multi-seed primary held-out and secondary bpb.
7. Apply `compare_official` order; ignore wall-clock and miner diagnostics.
8. Document GPU deferred if no NVIDIA; document TEE class without overclaim.
9. Do not push compare winners into emission narrative as if they were automatic weight crowns unless production scoring independently agrees.

### 11.1 Offline dual-family CPU/fixture harness (no NVIDIA)

Product harness: `python -m prism_challenge.evaluator.official_compare_harness`.

It packages the registered unknown-style dual families (`transformer-tiny-1m` + `mamba-tiny-1m`), builds multi-seed official records from challenge-owned synthetic metrics (or operator-supplied fixture metrics), and emits `prism_compare_report.v1.json` with a clear `ranking.winner` A vs B under this protocol. See [Operator Guide — Offline Official Comparison](operators.md#offline-official-comparison-cpu--fixture) for copy-paste commands.

```bash
export UV_CACHE_DIR=/var/tmp/uv-cache
uv run python -m prism_challenge.evaluator.official_compare_harness \
  --output-dir dist/official-compare --device-class fixture
uv run pytest tests/test_official_compare_harness.py -q
```

GPU verification status in the report is **DEFERRED** when the host lacks NVIDIA; the harness never toggles `claim_gpu_pass` to true.

---

## 12. Relation to other docs

| Doc | Relationship |
| --- | --- |
| [Scoring](scoring.md) | Production leaderboard math (bpb primary); explicit Official Comparison invert callout |
| [Submissions](submissions.md) | Two-script contract and `PrismContext` honest hooks |
| [Operators](operators.md) | Offline dual-family Official Comparison harness commands (CPU/fixture) |
| [Miner guide](miner/README.md) | Lab seed families and submission shape |
| [Security](security.md) | Sandbox, deterministic admission, TEE fail-closed honesty |
| [Scaling](scaling.md) | Single-node multi-GPU contract (pin may force nproc=1) |

Research background (mission-only audit, not shipping product claim): assembly notes that predated this lock may still talk “bpb primary” for scientific MDL framing of the **capture metric**. Under **user-locked Official Comparison v1 ranking**, held-out/generalization is PRIMARY and bpb is SECONDARY.

---

## 13. Versioning

- **v1** = this document: held-out primary, bpb secondary, anti-overfit + step-0, matched budgets, wall-clock never primary, multi-seed residual, honest hooks, miner self-report non-authoritative, TEE orthogonal, GPU deferred without NVIDIA.
- Breaking changes to pin fields, ranking order, or honesty rules require a new protocol_id (`v2+`) and a docs bump.

---

*End of Prism Official Comparison Protocol v1.*
