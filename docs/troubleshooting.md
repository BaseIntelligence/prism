# Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `400` on submit | Contract shape: `def build_model(` in `architecture.py`, `def train(` in `training.py`, ≤ 128 KiB per script |
| `403 hotkey_not_in_metagraph` | Hotkey not registered on the subnet; check the hex (64 lowercase, no `0x`) |
| `404 unknown_arch` | Training-only `arch_id` not in the registry — `GET /v1/architectures` |
| `409 submission_gated` | You already have an accepted submission (1-max per hotkey, per `(hotkey, arch_id)` for training-only) |
| `503 metagraph_unavailable` | Snapshot lag after a fresh registration — retry in a couple of minutes |
| `missing_telemetry_hooks` | `training.py` must import `prism_telemetry`, call `report(...)`, and may call `finish_evaluation()` |
| Status `rejected` | Pre-LLM copy gate: your `architecture.py` matches an *earlier* one. Write your own; start from the baseline |
| `similar: true` on precheck | Would hit intake copy gate — revise before `POST /v1/submissions` |
| `429 precheck_quota_exceeded` | 3 prechecks/coldkey/UTC day used; rotating hotkeys does not reset |
| `auto_retry` events | Infra failure retrying (up to 3 retries) — wait for the final state |
| `failed` after retries | Slot `blocked`; leaving the metagraph reopens it via the watcher |
| No telemetry on the site | Your loop never called `report(...)` — hooks are mandatory |
| Wrong host | Use the **gateway** `/challenge/prism/...` prefix (prod `https://chain.joinbase.ai`) |

When something fails, poll `events` — status alone is rarely enough:

```bash
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/events"
```

Frozen contracts live in the BASE monorepo: `docs/PRISM.md` + `docs/PRISM_RECIPE.md`.
