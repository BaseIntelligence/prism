# API

All miner routes go through the gateway prefix `/challenge/prism/...`.

Replace `{GATEWAY}` with `https://chain.joinbase.ai` (prod) or
`http://staging.api.joinbase.ai` (staging).

## Useful routes

| Route | What it tells you |
|-------|-------------------|
| `POST /challenge/prism/v1/submissions` | Submit AutoModel ZIP / JSON (`automodel.base` + `automodel.patch`) |
| `POST /challenge/prism/v1/submissions/precheck` | Advisory copy/layout gate (3/coldkey/UTC day; no queue/pod) |
| `GET /challenge/prism/v1/submissions/{id}` | Detail + bpb + review/similarity/agentic records |
| `GET /challenge/prism/v1/submissions/{id}/diff` | Unified diff + diffstat / classification (recipe ≥ 2.0) |
| `GET /challenge/prism/v1/submissions/{id}/events` | Stage timeline |
| `POST /challenge/prism/v1/submissions/{id}/retry` | Requeue an infra-failed row (within recovery window) |
| `GET /challenge/prism/v1/submissions?miner=<hex>` | Your submissions |
| `GET /challenge/prism/v1/recipe` | Caps + AutoModel pin (`automodel_pin_id`, commit, content sha) |
| `GET /challenge/prism/v1/status` | Backend / epoch / queues / recipe pin |
| `GET /challenge/prism/v1/jobs` | Active/recent pods (ops visibility) |
| `GET /v1/site/arenas/prism/submissions/{id}/telemetry` | Loss curve / gradients / layer stats |

## Poll example

```bash
export GATEWAY=https://chain.joinbase.ai
SUB=<submission_id from submit response>

curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB"
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/events"
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/diff"
```

## Status values

`queued` → `provisioning` → `running` → `llm_review` → `similarity` → `scoring` →
`terminated`.

Terminal states to know:

| Status | Meaning |
|--------|---------|
| `rejected` | Copy / layout / causal gate: terminal `Score(0)` (often before GPU) |
| `failed` | Infra retries exhausted (`auto_retry` events) or harness/internal failure |
| `terminated` with `score.kind = "no_score"` | `ChallengeInternal` — operator-side, never a miner zero |
| `terminated` with score 0 | Cheat / suspicious / copied verdict (see the `scoring` event detail) |

Submit errors: `400 unsupported_layout`, `400 recipe_version`,
`400 missing_lium_api_key`, `403 hotkey_not_in_metagraph`,
`409 submission_gated`, `503 metagraph_unavailable`.

Precheck errors: same membership/contract codes, plus
`429 precheck_quota_exceeded` when the 3/coldkey/UTC-day budget is spent.

## Auth note

Miner routes identify you by hotkey (`X-Miner-Hotkey` or JSON `miner_hotkey`).
On live, also send **`X-Lium-Api-Key`** (your funded Lium account). Never send
challenge signing keys or gateway owner keys from a miner client.

## Next

→ [Troubleshooting](troubleshooting.md)
