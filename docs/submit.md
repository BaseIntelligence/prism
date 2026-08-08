# Submit

Preferred path: a **ZIP** through the production or staging gateway.

## ZIP (preferred)

| Header / body | Value |
|---------------|--------|
| Method / URL | `POST {GATEWAY}/challenge/prism/v1/submissions` |
| `Content-Type` | `application/zip` |
| `X-Miner-Hotkey` | 64 lowercase hex |
| `X-Prism-Arch-Id` | training-only entries: published `arch_id` (zip then contains `training.py` only) |
| Body | raw zip bytes (`architecture.py` + `training.py` at the root) |

```bash
export GATEWAY=https://chain.joinbase.ai
export HOTKEY=<64 lowercase hex>

cd examples/baseline
zip -j submission.zip architecture.py training.py

curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

## JSON (local / scripting)

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/json' \
  -d @submission.json
```

```json
{
  "miner_hotkey": "<64 lowercase hex>",
  "architecture_py": "<contents of architecture.py>",
  "training_py": "<contents of training.py>",
  "label": "optional human label"
}
```

`POST /v1/submissions` is **idempotent** by `submission_id` (digest of hotkey + sources):
re-POSTing identical sources returns `200 {"status":"already-queued"}`.

## Registration and 1-max gating

| Situation | Response |
|-----------|----------|
| Hotkey not in the subnet metagraph | `403 hotkey_not_in_metagraph` |
| Metagraph snapshot not ready yet | `503 metagraph_unavailable` (retry shortly) |
| You already have an accepted architecture submission | `409 submission_gated` |

**One accepted architecture submission per hotkey.** While yours is
`registered` / `blocked` / `rejected`, a *different* architecture submission gets
`409 submission_gated`. If your hotkey **leaves the metagraph** (uid deregistered or
swapped), the watcher reopens your slot automatically.

## Training-only entries

Training-only entries are **separate slots**: one accepted entry per `(hotkey, arch_id)`
— you may train on many published architectures, one script per arch.

```bash
# JSON
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/json' \
  -d '{"miner_hotkey":"<hex>","arch_id":"<arch_…>","training_py":"<contents>"}'

# ZIP (training.py only) + header
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  -H "X-Prism-Arch-Id: <arch_id>" \
  --data-binary @training-only.zip
```

Do **not** include `architecture.py` in a training-only entry — the source is pulled
from the registry (miner-sent architecture is rejected on these rows). Unknown
`arch_id` → `404 unknown_arch`.

## Retries and terminal states

- Infra failures (pod provisioning, review/similarity/LLM infra) **auto-retry up to 3
  times**. Retry budget exhausted → `failed`, slot `blocked`.
- Cheat / rejected verdicts are **terminal** — no auto-retry. Manual retry for
  infra-class failures: `POST /v1/submissions/{id}/retry`.

## Precheck similarity before you submit

Dry-run the same pre-LLM copy gate **without** burning your 1-max slot or a GPU
eval. Same payload as submit (ZIP or JSON):

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions/precheck" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

Example response:

```json
{
  "similar": false,
  "verdict": "clean",
  "message": "no earlier architecture copy detected by the pre-LLM gate; full submit still runs similarity + agentic",
  "quota": {
    "day": "2026-08-08",
    "used": 1,
    "limit": 3,
    "remaining": 2,
    "identity": "coldkey"
  }
}
```

| Field | Meaning |
|-------|---------|
| `similar` | `true` → would hard-reject at intake copy gate |
| `verdict` | `clean` / `copied` / `skipped` (training-only) |
| `matched_against` | Corpus id only (never competitor source) |
| `score` | Similarity in `[0,1]` when compared |
| `quota` | Daily budget (`limit` = 3) |

**Quota: 3 attempts per coldkey per UTC day** (hotkey fallback when Owner is
unknown). Rotating hotkeys under the same coldkey does **not** reset the budget.
A 4th call returns `429` / `precheck_quota_exceeded` with `remaining=0`.

## Gateways

| Environment | Base URL |
|-------------|----------|
| Production | `https://chain.joinbase.ai` |
| Staging | `http://staging.api.joinbase.ai` |

Always use the `/challenge/prism/...` prefix on those hosts.

## Next

→ [Scoring & competition](scoring.md)  
→ [API](api.md) to poll results
