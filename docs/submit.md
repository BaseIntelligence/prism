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

## Source-tree ZIP (recipe ≥ 1.3.0)

Instead of only two scripts you may submit a full **source tree**: the two seam files
plus optional `prism.toml` (entry point), `count_params.py`, a `kernels/` directory of
custom ops (pure Python + torch, per `KERNEL_INTERFACE.md` — no prebuilt binaries, no
`ctypes`, no I/O or threads), and a `vendor.lock`. Trees are validated at intake
(size budgets, banned-pattern scan, canonical hash) and re-audited in-pod by the
harness cheatguard. Send tree ZIPs via the JSON `zip_base64` field (the raw-zip path
rejects them with a pointer to `zip_base64` so the full tree is validated and
retained):

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/json' \
  -d '{"miner_hotkey":"<hex>","zip_base64":"<base64 of tree.zip>"}'
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

## Gateways

| Environment | Base URL |
|-------------|----------|
| Production | `https://chain.joinbase.ai` |
| Staging | `http://staging.api.joinbase.ai` |

Always use the `/challenge/prism/...` prefix on those hosts.

## Next

→ [Scoring & competition](scoring.md)  
→ [API](api.md) to poll results
