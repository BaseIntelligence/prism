# Submit

Preferred path: a **ZIP** through the production or staging gateway with
`automodel.base` + `automodel.patch` (+ optional `prism.toml`).

## ZIP (preferred)

| Header / body | Value |
|---------------|--------|
| Method / URL | `POST {GATEWAY}/challenge/prism/v1/submissions` |
| `Content-Type` | `application/zip` |
| `X-Miner-Hotkey` | 64 lowercase hex |
| `X-Lium-Api-Key` | Lium API key (live, if you pay Lium) |
| `X-Verda-Client-Id` / `X-Verda-Client-Secret` / `X-Verda-Inference-Key` | Verda BYOK (live, if you pay Verda). `X-Verda-Api-Key` aliases the inference token. If both providers are complete, add `X-Compute-Provider: lium` or `verda`. You cannot set `image` / `cmd` / `template`. |
| Body | raw zip bytes (`automodel.base` + `automodel.patch` at the root) |

```bash
export GATEWAY=https://chain.joinbase.ai
export HOTKEY=<64 lowercase hex>
export LIUM_API_KEY=<your Lium API key>

# automodel.base = single line equal to recipe automodel_pin_id (automodel@v0.5.0)
# automodel.patch = git diff <automodel_git_commit>
zip -j submission.zip automodel.base automodel.patch   # + prism.toml if used

curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  --data-binary @submission.zip
```

## JSON (local / scripting)

Same members as the ZIP (or `zip_base64`). Prefer ZIP on live.

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/json' \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  -d @submission.json
```

`POST /v1/submissions` is **idempotent** by `submission_id` (hash of **pin id ‖
`0x00` ‖ patch bytes**): re-POSTing the identical pin+patch returns
`200 {"status":"already-queued"}`.

## Registration and 1-max gating

| Situation | Response |
|-----------|----------|
| Hotkey not in the subnet metagraph | `403 hotkey_not_in_metagraph` |
| Metagraph snapshot not ready yet | `503 metagraph_unavailable` (retry shortly) |
| You already have an accepted patch submission | `409 submission_gated` |
| Missing Lium key on live | `400 missing_lium_api_key` |

**One accepted patch submission per hotkey.** While yours is `registered` /
`rejected`, or `blocked` **outside** the infra recovery window, a *different*
patch submission gets `409 submission_gated`. Re-POSTing the **identical**
pin+patch is always safe (idempotent).

If your hotkey **leaves the metagraph**, the watcher reopens your slot(s)
automatically — resubmit under your new uid.

## Retries and terminal states

- Infra failures (Lium pod, review/similarity/LLM infra) **auto-retry up to 3
  times**. Cheat / rejected verdicts are terminal.
- After an infra failure (`ChallengeInternal`), you may **resubmit within 30
  minutes** (new POST or `POST /v1/submissions/{id}/retry`). After 30 minutes
  the slot stays blocked until your hotkey leaves the metagraph.

## Precheck similarity before you submit

Dry-run the copy / layout gate **without** burning your 1-max slot or a GPU
eval (send the same AutoModel ZIP you would submit):

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions/precheck" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

| Field | Meaning |
|-------|---------|
| `similar` | `true` → would hard-reject at intake copy gate |
| `verdict` | `clean` / `copied` / `skipped` |
| `matched_against` | Corpus id only (never competitor source) |
| `score` | Similarity in `[0,1]` when compared |
| `quota` | `{ day, used, limit: 3, remaining, identity }` |

**Quota: 3 attempts per coldkey per UTC day** (falls back to hotkey when the
metagraph Owner coldkey is unknown). Rotating hotkeys under the same coldkey
does **not** reset the budget. A 4th call returns `429` /
`precheck_quota_exceeded` with `remaining=0`. Precheck never creates a scored
submission and never rents a Lium pod.

## Inspect your applied diff

After intake:

```bash
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/diff"
```

Returns the full unified diff plus diffstat / classification
(`arch` / `trainer` / `data` / `other`).

## Gateways

| Environment | Base URL |
|-------------|----------|
| Production | `https://chain.joinbase.ai` |
| Staging | `http://staging.api.joinbase.ai` |

Always use the `/challenge/prism/...` prefix on those hosts.

## Next

→ [Scoring & competition](scoring.md)  
→ [API](api.md) to poll results
