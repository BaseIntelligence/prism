# Getting started

## The contract (recipe v2.1.0)

You do **not** ship a free-form `architecture.py` / `training.py` project.
Live recipe **2.1.0** accepts only a pin id plus your unified diff against that
pin:

```text
automodel.base          # required — pin id from GET /v1/recipe (live: automodel@v0.5.0)
automodel.patch         # required — unified diff vs that pin (git diff pin...HEAD)
prism.toml              # optional — entry / model-config knobs
```

**Workflow: fork pin → edit → `git diff` → submit**

1. Read the live pin from `GET /v1/recipe` (`automodel_pin_id`,
   `automodel_repo_url`, `automodel_git_commit`, `automodel_content_sha256`).
2. Check out that exact AutoModel commit (or extract the staged archive and
   verify `automodel_content_sha256` matches `/v1/recipe`).
3. Edit under the AutoModel layout — new model modules / configs are allowed;
   trainer / data-path edits get high scrutiny.
4. Produce a unified diff against the pin commit, e.g.
   `git diff <automodel_git_commit> > automodel.patch`.
5. Write `automodel.base` as a single line equal to `automodel_pin_id`, pack
   the ZIP, and `POST /v1/submissions` with your hotkey + **`X-Lium-Api-Key`**.

Models must stay **≤ 1B parameters**. Recipe-v10 pods expose four GPUs;
train from `ctx["train_stream"]` (rank 0 owns the stream under DDP). Miner
**model** code has **no network** (`unshare --net`, loopback up for rendezvous)
beyond the operator-owned dataset pull — do not call Hub downloads from
`build_model` / `train`. You may ship `requirements.txt` or `pyproject.toml`
for a network-on install phase before that sandbox.

**Legacy recipe 1.x is rejected on live.** Two-script ZIPs
(`architecture.py` + `training.py`), 1.3 source-tree ZIPs, and training-only
`arch_id` submissions return `400 unsupported_layout` or `400 recipe_version`.
Do not ship Megatron-Bridge or other non-AutoModel frameworks.

## Pay for your own GPU (required on live)

Create a [Lium](https://lium.io) account, fund it, and pass your API key on
every live submit:

```http
X-Lium-Api-Key: <your Lium API key>
```

The key is held in master memory for that submission and may also land in a
**short-TTL encrypted seal file** on the master host (never in Postgres, never
logged) so a control-plane restart can still stop your pod. Missing key on
live → `400 missing_lium_api_key`.

If the challenge process restarts mid-run, your submission is marked failed
promptly with `control_plane_restart` / `harness_detached`. Stop the Lium pod
if it is still billing, then resubmit with `X-Lium-Api-Key`. Poll
`GET /v1/submissions/{id}/events` and `GET /v1/submissions/{id}/logs?since=`.

## Telemetry hooks (still required)

The harness wrap still requires `prism_telemetry` reporting /
`finish_evaluation` under the AutoModel train entry. Patches that remove or
bypass those hooks fail review (`missing_telemetry_hooks`, zero score,
terminal).

## Pinned dataset

| Field | Value |
|-------|-------|
| Ref | `HuggingFaceFW/fineweb-edu@sample/10BT` |
| URL | `…/resolve/main/sample/10BT/010_00000.parquet` |
| Bytes | 2 152 798 864 |
| SHA-256 | `e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6` |

The harness re-verifies the hash on the file it actually fetched; a mismatch ends the
eval as `ChallengeInternal` — never a miner score. Always confirm live values via
`GET /v1/recipe`.

## Budget & caps

| Cap | Value |
|-----|-------|
| Train wall clock | 6.0 h per submission (`train_hours_cap`) |
| Hard step cap | 20 000 (`max_train_steps`) |
| Model parameters | ≤ **350 000 000** (`max_params`) |

Trust live `GET /v1/recipe` (`version`, `automodel_*`, `pin_hex`, caps) over any
marketing chart.

## Recipe pin

```bash
curl -sS "$GATEWAY/challenge/prism/v1/recipe"
```

Live recipe **2.0.0** advertises `version: "2.0.0"` and AutoModel pin fields
(`automodel_pin_id` = `automodel@v0.5.0`, `automodel_repo_url`,
`automodel_git_ref`, `automodel_git_commit`, `automodel_content_sha256`).

## Next

→ [Submit](submit.md)  
→ [Full guide](prism.md)  
→ [Scoring & competition](scoring.md)
