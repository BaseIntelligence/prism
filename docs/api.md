# API Reference

PRISM exposes public challenge routes and internal BASE routes.

## Public Routes

- `GET /health` — challenge health metadata.
- `GET /version` — version, API version, SDK version, and capabilities.
- `GET /v1/submissions/history` — daily submission counts over a window.
- `GET /v1/leaderboard` — submissions ranked by `final_score` for the current epoch
  (earliest-commit-wins on a tie, one entry per hotkey).
- `GET /v1/architectures/{architecture_id}` — one architecture's lab detail (name, owner, best
  score/submission, variant and submission counts, `first_seen_at`, `updated_at`); `404` if absent.
- `GET /v1/architectures/{architecture_id}/variants` — the architecture's training-script variants
  (best first); `404` if the architecture is absent, empty `variants` is valid.
- `GET /v1/architectures/{architecture_id}/report` — the cached LLM auto-report, generated lazily and
  non-blockingly through the master LLM gateway and grounded only in measured facts. `report.status`
  is `ready`, `pending`, or `unavailable`; `content` may be `null` when not `ready`.
- `GET /v1/submissions/{submission_id}/curve` — the persisted loss curve (`online_loss` +
  `covered_bytes_cumulative`, downsampled to at most 500 points, first and last preserved), the
  prequential bits-per-byte scalars, and the reconciled compute profile (`estimated_flops`,
  `gpu_hours`, peak VRAM/RSS, wall-clock); `404` if none.
- `GET /v1/epochs/current` — current epoch id and length.
- `GET /v1/epochs` — recent epochs.
- `GET /v1/health/eval-jobs` — recent eval-job health entries (id, submission id, level, status, attempts).
- `GET /v1/gpu/status` — GPU-lease summary (total GPUs, active leases, by status, by tier).

### `POST /v1/submissions`

Submit a two-script bundle directly to PRISM (miner authentication headers). In production, submissions
usually enter through the BASE proxy and the internal bridge route instead.

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

### `GET /v1/submissions/{submission_id}`

```json
{
  "id": "...",
  "hotkey": "...",
  "epoch_id": 123,
  "status": "completed",
  "final_score": 0.72,
  "anti_cheat_multiplier": 1.0
}
```

`final_score` is the challenge-computed prequential bits-per-byte score (a lower bpb yields a higher
`final_score`); `q_arch`, `q_recipe`, `diversity_bonus`, and `penalty` are legacy fields retained for
response-schema stability. `status` is `pending`, `running`, `completed`, `failed`, `rejected`, or
`held` (rejected = failed a static gate, the two-script contract, the LLM hard gate, or duplicate
review; held = LLM-review quarantine).

### `GET /v1/architectures`

Architecture-lab leaderboard grouped by architecture family, ranked by `best_final_score` descending.
Optional `epoch_id` scopes to architectures with a completed submission in that epoch; omitting it
resolves to the most-recent non-empty epoch. `name` is the miner-declared, deterministically moderated
architecture name (may be `null`).

```json
{
  "epoch_id": 42,
  "architectures": [
    {
      "rank": 1,
      "architecture_id": "...",
      "arch_hash": "...",
      "name": "Rotary MoE v3",
      "owner_hotkey": "...",
      "best_final_score": 1.2345,
      "best_submission_id": "...",
      "variant_count": 3,
      "submission_count": 7,
      "updated_at": "..."
    }
  ]
}
```

## Internal BASE Routes

All internal routes require `Authorization: Bearer <shared-token>`.

### `GET /internal/v1/get_weights`

Standard BASE challenge contract. Returns normalized, dry-run hotkey weights (one per hotkey, from that
hotkey's best `final_score`). Weights are never written on-chain.

### `POST /internal/v1/bridge/submissions`

Receives BASE-verified submissions.

```text
Authorization: Bearer <shared-token>
X-Base-Verified-Hotkey: <hotkey>
X-Submission-Filename: project.zip
Content-Type: application/zip
```

The body can be raw ZIP bytes or JSON matching `SubmissionCreate`.

### `POST /internal/v1/worker/process-next`

Claims and processes one pending submission through the full pipeline: static gates, the LLM hard gate,
the forced-init re-execution, and prequential bits-per-byte scoring.
