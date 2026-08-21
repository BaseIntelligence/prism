# Troubleshooting

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| Rejected submit | Recipe contract | `GET /v1/recipe` (`automodel_pin_id` + caps); follow recipe **2.0** |
| `400 unsupported_layout` | Legacy 1.x ZIP or missing AutoModel members | Ship `automodel.base` + `automodel.patch` (+ optional `prism.toml`). Two-script / source-tree / `arch_id` layouts are rejected on live 2.0 |
| `400 recipe_version` | Payload implies recipe 1.x while live advertises ≥ 2.0 | Re-pack as AutoModel patch ZIP; do not send `architecture.py`/`training.py` |
| Patch apply failure / conflict | Diff not against the live pin, or stale rebase | Checkout exact `automodel_git_commit` from `/v1/recipe`; regenerate `git diff <commit>`; ensure `automodel.base` == `automodel_pin_id` |
| Wrong / unknown pin id | `automodel.base` ≠ recipe `automodel_pin_id` | Copy `automodel_pin_id` (live: `automodel@v0.5.0`) byte-identical from `/v1/recipe` |
| Binary / path-escape / oversized patch | Fail-closed apply rules | Text-only unified diff; no path escape outside allowlisted roots; keep diff within intake budgets |
| Tokenizer / hub errors on pod | No network; Hub download from miner code | Stay offline; use pin/harness tokenizer paths — do not `from_pretrained("<hub id>")` |
| `CAP_EXCEEDED` / Score 0 | Model > 350M params | Terminal — resize model config in your patch; not auto-retried |
| `missing_telemetry_hooks` | Patch removed / bypassed harness telemetry | Keep `prism_telemetry.report` (+ optional `finish_evaluation`) under the AutoModel train entry |
| Score 0 after review | `Copied` / high-confidence `Suspicious` (≥0.9, non-trope) | Similarity on **your delta**; rewrite unique hunks; tropes alone are not plagiarism |
| `similar: true` on precheck | Would hit intake copy gate | Change the patch vs prior champions; starting from the operator pin is fine |
| `429 precheck_quota_exceeded` | 3 prechecks/coldkey/UTC day used | Wait until next UTC day; rotating hotkeys does not reset |
| `control_plane_restart` / `harness_detached` | Challenge process restarted mid-pod | Stop the Lium pod if still billing; resubmit with `X-Lium-Api-Key` |
| `400 missing_lium_api_key` | Live path needs miner-funded Lium or Verda | Pass `X-Lium-Api-Key` or the Verda header triplet; see [Submit](submit.md) |
| `400 missing_verda_credentials` | Verda headers incomplete | Send client id, secret, and inference key |
| `400 ambiguous_compute_provider` | Both Lium and Verda complete | Add `X-Compute-Provider: lium` or `verda` |
| `400 miner_image_override` | JSON set image/cmd/template | Remove those fields — operator pins the container |
| **409 `not_failed`** on `/retry` | Row is not `failed` | `/retry` only recovers **failed** rows. Re-POSTing the identical ZIP is `already-queued` (no new GPU). After infra failure: `POST .../retry` with **`X-Lium-Api-Key`** (hotkey/Bearer alone is not enough) |
| **400 `missing_lium_api_key`** on `/retry` | Need another GPU rent | Pass `X-Lium-Api-Key` or Verda BYOK on live |
| Non-5090 / slow tok/s vs peers | Marketplace drew another SKU | Prism hard-pins **1× RTX 5090**; non-5090 is rejected at rent (no silent score normalize) |
| `403 hotkey_not_in_metagraph` | Hotkey not registered | Check the hex (64 lowercase, no `0x`) |
| `409 submission_gated` | 1-max slot already used | One accepted patch per hotkey; identical pin+patch is idempotent |
| `503 metagraph_unavailable` | Snapshot lag after a fresh registration | Retry in a couple of minutes |
| Stuck `Provisioning` | Lium market / underfunded key | Check your Lium balance; watch `GET /v1/jobs` / events |
| Idempotent replay | Same `submission_id` (pin id + patch bytes) | Expected — returns prior row |
| Wrong host | Not using gateway prefix | Use **gateway** `/challenge/prism/...` (prod `https://chain.joinbase.ai`) |

When something fails, poll `events` — status alone is rarely enough:

```bash
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/events"
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/diff"
```

Frozen contracts live in the BASE monorepo:
[`docs/PRISM.md`](https://github.com/BaseIntelligence/base/blob/main/docs/PRISM.md) +
[`docs/PRISM_RECIPE.md`](https://github.com/BaseIntelligence/base/blob/main/docs/PRISM_RECIPE.md).
