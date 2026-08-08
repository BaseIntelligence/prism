# API

All miner routes go through the gateway prefix `/challenge/prism/...`.

Replace `{GATEWAY}` with `https://chain.joinbase.ai` (prod) or
`http://staging.api.joinbase.ai` (staging).

## Useful routes

| Route | What it tells you |
|-------|-------------------|
| `POST /challenge/prism/v1/submissions` | Submit zip / JSON / training-only |
| `POST /challenge/prism/v1/submissions/precheck` | Advisory copy-gate (3/coldkey/UTC day; no queue/pod) |
| `GET /challenge/prism/v1/submissions/{id}` | Detail + bpb + review/similarity/agentic records |
| `GET /challenge/prism/v1/submissions/{id}/events` | Stage timeline |
| `POST /challenge/prism/v1/submissions/{id}/retry` | Requeue an infra-failed row |
| `GET /challenge/prism/v1/submissions?miner=<hex>` | Your submissions |
| `GET /challenge/prism/v1/architectures` | Published archs + per-arch best bpb |
| `GET /challenge/prism/v1/recipe` | Versioned recipe descriptor + pin |
| `GET /challenge/prism/v1/recipe/baseline` | Official baseline scripts |
| `GET /challenge/prism/v1/status` | Backend / epoch / queues / recipe pin |
| `GET /v1/site/arenas/prism/submissions/{id}/telemetry` | Loss curve / gradients / layer stats |

## Poll example

```bash
export GATEWAY=https://chain.joinbase.ai
SUB=<submission_id from submit response>

curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB"
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/events"
```

## Status values

`queued` → `provisioning` → `running` → `llm_review` → `similarity` → `scoring` →
`terminated`.

Terminal states to know:

| Status | Meaning |
|--------|---------|
| `rejected` | Pre-LLM copy gate: byte/AST copy of an *earlier* architecture (`Score(0)`, no GPU time, no LLM review) |
| `failed` | Infra retries exhausted (`auto_retry` events) or harness/internal failure |
| `terminated` with `score.kind = "no_score"` | `ChallengeInternal` — operator-side, never a miner zero |
| `terminated` with score 0 | Cheat / suspicious / copied verdict (see the `scoring` event detail) |

Submit errors: `403 hotkey_not_in_metagraph`, `404 unknown_arch`,
`409 submission_gated`, `503 metagraph_unavailable`.

Precheck errors: same membership/contract codes, plus
`429 precheck_quota_exceeded` when the 3/coldkey/UTC-day budget is spent.

## Auth note

Miner routes identify you by hotkey (`X-Miner-Hotkey` or JSON `miner_hotkey`). Never
send challenge signing keys or gateway owner keys from a miner client.

## Next

→ [Troubleshooting](troubleshooting.md)
