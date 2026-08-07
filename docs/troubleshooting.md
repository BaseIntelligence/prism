# Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `400` on submit | Contract shape: `def build_model(` in `architecture.py`, `def train(` in `training.py`, size limits; source-tree caps (128 files / 4 MiB/file / 16 MiB); banned patterns (`ctypes`, prebuilt binaries, …) |
| Source-tree rejected on raw ZIP | Use JSON `zip_base64` so the full tree is validated and retained |
| Tokenizer / hub errors on pod | Pod has **no network** — use `ctx["tokenizer"]`; declare via `tokenizer/` or `build_tokenizer` in `architecture.py` (not `training.py`) |
| `build_tokenizer` ignored / rejected | Hook must live in `architecture.py` beside `build_model` |
| `CAP_EXCEEDED` / Score 0 | Model > 350M params after `build_model` — terminal, not retried |
| `403 hotkey_not_in_metagraph` | Hotkey not registered on the subnet; check the hex (64 lowercase, no `0x`) |
| `404 unknown_arch` | Training-only `arch_id` not in the registry — `GET /v1/architectures` |
| `409 submission_gated` | You already have an accepted submission (1-max per hotkey, per `(hotkey, arch_id)` for training-only) |
| `503 metagraph_unavailable` | Snapshot lag after a fresh registration — retry in a couple of minutes |
| `missing_telemetry_hooks` | `training.py` must import `prism_telemetry`, call `report(...)`, and may call `finish_evaluation()` |
| Status `rejected` | Pre-LLM copy gate: your `architecture.py` matches an *earlier* one. Write your own; start from the baseline |
| `auto_retry` events | Infra failure retrying (up to 3 retries) — wait for the final state |
| `failed` after retries | Slot `blocked`; leaving the metagraph reopens it via the watcher |
| No telemetry on the site | Your loop never called `report(...)` — hooks are mandatory |
| Wrong host | Use the **gateway** `/challenge/prism/...` prefix (prod `https://chain.joinbase.ai`) |

When something fails, poll `events` — status alone is rarely enough:

```bash
curl -sS "$GATEWAY/challenge/prism/v1/submissions/$SUB/events"
```

Frozen contracts live in the BASE monorepo: `docs/PRISM.md` + `docs/PRISM_RECIPE.md`.
