"""Prism master-side held-out RCE-safe load, claude-opus gate via the master gateway, and weights.

Covers VAL-PRISM-030..036 (architecture.md sections 5, 6, 7, 11):
  030 - trained_state loaded ONLY when the challenge manifest recorded that exact artifact as a
        regular file under the run dir (no traversal/symlink), with weights_only=True.
  031 - the llm_review gate routes through the master gateway /llm/v1 at temp 0; the gateway injects
        the model (the client sends a placeholder), so no provider model is pinned in code.
  032 - an LLM reject is TERMINAL and pre-eval (no container eval / GPU lease / checkpoint publish).
  033 - a gateway failure / unparseable verdict fails CLOSED (hold); oversized source rejected.
  034 - the validator/eval runtime holds NO provider key (gateway token only); secrets redacted.
  035 - get_weights = two-tier split of prism's emission between the cross-epoch best architecture's
        owner (architecture_weight) and the best training-variant owner on it (training_weight),
        renormalized to sum 1 ({} when no crown exists or its all-time best is non-positive).
  036 - /internal/v1/get_weights keeps the {challenge_slug, epoch, weights{hotkey: float}} shape.

External systems are mocked: the chat client is stubbed (no real provider), the broker executor and
checkpoint publisher are monkeypatched, and the held-out runs the tiny CPU twin (no GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator import llm_review as llm
from prism_challenge.evaluator.container import (
    TRAINED_STATE_ARTIFACT,
    PrismContainerEvaluator,
    _redact_detail,
    _resolve_recorded_trained_state,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.llm_review import (
    GATEWAY_MODEL_PLACEHOLDER,
    LlmReviewConfig,
    review_code,
    review_plagiarism,
)
from prism_challenge.repository import PrismRepository
from prism_challenge.weights import get_weights

GATEWAY_URL = "http://base-master:18080/llm/v1"
GATEWAY_TOKEN = "**********************"
PROVIDER_KEY = "sk-or-raw-provider-key-must-not-leak"
EPOCH_SECONDS = 60

_ALLOW_ARGS: dict[str, dict[str, Any]] = {
    "SubmitMermaid": {"mermaid": "flowchart LR\n  A[arch] --> B[train]", "notes": "ok"},
    "SubmitVerdict": {
        "reason": "coherent from-scratch learner; no escapes",
        "verdict": True,
        "violations": [],
        "confidence": 0.9,
        "rule_ids": [],
        "evidence": [],
    },
}


def _fake_chat_class(captured: dict[str, Any], *, verdict: bool = True) -> type:
    args_by_tool = dict(_ALLOW_ARGS)
    args_by_tool["SubmitVerdict"] = {**_ALLOW_ARGS["SubmitVerdict"], "verdict": verdict}

    class _FakeMessage:
        def __init__(self, tool_name: str) -> None:
            self.tool_calls = [{"name": tool_name, "args": args_by_tool[tool_name]}]

    class _FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self._tool: str | None = None

        def bind_tools(self, tools: list[Any], tool_choice: str, strict: bool) -> _FakeChat:
            self._tool = tool_choice
            return self

        def invoke(self, messages: list[tuple[str, str]]) -> _FakeMessage:
            assert self._tool is not None
            return _FakeMessage(self._tool)

    return _FakeChat


# --- VAL-PRISM-030: RCE-safe trained_state resolution ---------------------------------------------


def test_recorded_trained_state_accepts_regular_file_under_run_dir(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / TRAINED_STATE_ARTIFACT).write_bytes(b"weights")

    resolved = _resolve_recorded_trained_state(artifacts, TRAINED_STATE_ARTIFACT)
    assert resolved == (artifacts / TRAINED_STATE_ARTIFACT).resolve()


def test_recorded_trained_state_rejects_unrecorded_or_wrong_name(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / TRAINED_STATE_ARTIFACT).write_bytes(b"weights")

    # Not recorded at all, recorded under a different name, or non-str -> never deserialized.
    assert _resolve_recorded_trained_state(artifacts, None) is None
    assert _resolve_recorded_trained_state(artifacts, "weights.pt") is None
    assert _resolve_recorded_trained_state(artifacts, 123) is None


def test_recorded_trained_state_rejects_traversal_name(tmp_path: Path) -> None:
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)
    outside = tmp_path / "trained_state.pt"
    outside.write_bytes(b"hostile")

    # A recorded name carrying traversal segments is not the exact challenge artifact -> rejected.
    assert _resolve_recorded_trained_state(artifacts, "../trained_state.pt") is None
    assert _resolve_recorded_trained_state(artifacts, "../../etc/passwd") is None


def test_recorded_trained_state_rejects_symlink_escape(tmp_path: Path) -> None:
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)
    outside = tmp_path / "outside_state.pt"
    outside.write_bytes(b"hostile-pickle")
    # The miner-writable artifacts dir holds trained_state.pt as a SYMLINK pointing outside the run
    # dir; resolving it lands outside `base`, so it must be refused (no symlink escape).
    (artifacts / TRAINED_STATE_ARTIFACT).symlink_to(outside)

    assert _resolve_recorded_trained_state(artifacts, TRAINED_STATE_ARTIFACT) is None


def test_recorded_trained_state_returns_none_for_missing_file(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # Recorded by name but no regular file present -> None (held-out skipped, run still scores).
    assert _resolve_recorded_trained_state(artifacts, TRAINED_STATE_ARTIFACT) is None


# --- VAL-PRISM-031 / VAL-LLM-CODE-008: gate routes via the master /llm/v1 gateway at temp 0 ------


def test_llm_gate_routes_through_master_gateway_with_token(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm, "_load_chat_openai", lambda: _fake_chat_class(captured))

    config = LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=GATEWAY_TOKEN)
    review = review_code("def build_model(ctx):\n    return None\n", config=config)

    assert review.approved is True
    # The gateway injects the model server-side, so the client sends only a placeholder (no
    # hardcoded provider model), targets /llm/v1, and authenticates with the scoped token.
    assert captured["model"] == GATEWAY_MODEL_PLACEHOLDER
    assert captured["base_url"] == GATEWAY_URL
    assert captured["temperature"] == 0.0
    assert captured["api_key"] == GATEWAY_TOKEN


def test_resolve_endpoint_fails_closed_without_gateway() -> None:
    # A gateway URL is configured but its scoped token is unresolvable: the gate MUST refuse (there
    # is NO direct-provider fallback).
    with pytest.raises(RuntimeError, match="gateway"):
        llm._resolve_endpoint(LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=None))

    # With no gateway URL configured at all, the gate also fails closed (no direct-provider path).
    with pytest.raises(RuntimeError, match="gateway"):
        llm._resolve_endpoint(LlmReviewConfig())


def test_gateway_configured_without_token_holds_no_direct_provider_call(monkeypatch) -> None:
    # End-to-end: review_code with a gateway URL but no resolvable token fails closed (HOLD) and
    # never constructs a chat client / makes a direct provider call.
    def must_not_build_client() -> None:
        raise AssertionError("no chat client may be built when the gateway token is unresolvable")

    monkeypatch.setattr(llm, "_load_chat_openai", must_not_build_client)

    review = review_code(
        "def build_model(ctx):\n    return None\n",
        config=LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=None),
    )

    assert review.approved is False
    assert review.held is True
    assert "llm_review_failed" in review.violations
    assert PROVIDER_KEY not in review.reason


def test_settings_gateway_token_only_sourced_from_secret_file(tmp_path: Path) -> None:
    secret = tmp_path / "base_gateway_token"
    secret.write_text(GATEWAY_TOKEN + "\n", encoding="utf-8")
    settings = PrismSettings(llm_gateway_url=GATEWAY_URL, llm_gateway_token_file=secret)
    assert settings.llm_gateway_token_value() == GATEWAY_TOKEN

    missing = PrismSettings(llm_gateway_token=None, llm_gateway_token_file=tmp_path / "nope")
    assert missing.llm_gateway_token_value() is None


# --- VAL-PRISM-032: an LLM reject is terminal and pre-eval ----------------------------------------


def test_llm_reject_is_terminal_no_eval_lease_or_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_load_chat_openai", lambda: _fake_chat_class({}, verdict=False))

    def fail_run(self, spec, timeout_seconds):  # noqa: ANN001
        raise AssertionError("container eval must not run after an LLM reject")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fail_run)

    name = "prism-reject-terminal.sqlite3"
    with TestClient(create_app(_pipeline_settings(tmp_path, name))) as client:
        submission_id = _submit(client, two_script_bundle(), nonce="reject-1")
        assert _process(client) == 200

    state = _db_state(tmp_path, name, submission_id)
    assert state["status"] == "rejected"
    assert state["gpu_leases"] == 0
    assert state["eval_jobs"] == 0
    assert state["scores"] == 0


# --- VAL-PRISM-033: gateway failure / parse error holds; oversized rejected pre-call --------------


def test_gateway_failure_fails_closed_to_hold(monkeypatch) -> None:
    def boom(config, *, system, prompt):  # noqa: ANN001
        raise TimeoutError("master gateway upstream 503 (transient)")

    monkeypatch.setattr(llm, "_invoke_review_flow", boom)

    review = review_code(
        "ok learner",
        config=LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=GATEWAY_TOKEN),
    )
    assert review.approved is False
    assert review.held is True
    assert "llm_review_failed" in review.violations


def test_unparseable_verdict_fails_closed_to_hold(monkeypatch) -> None:
    # A verdict outside the bool vocabulary is a parse failure -> hold (never coerced to allow).
    monkeypatch.setattr(llm, "_load_chat_openai", lambda: _make_fake_chat_with_verdict("banana"))

    review = review_code(
        "ok",
        config=LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=GATEWAY_TOKEN),
    )
    assert review.approved is False
    assert review.held is True
    assert "parse" in review.reason.lower()


def test_oversized_source_rejected_before_any_model_call(monkeypatch) -> None:
    def must_not_call(config, *, system, prompt):  # noqa: ANN001
        raise AssertionError("no model call may be issued for oversized source")

    monkeypatch.setattr(llm, "_invoke_review_flow", must_not_call)

    big = "x = 1  # " + ("A" * 5000) + "\n"
    review = review_code(
        big, config=LlmReviewConfig(gateway_token=GATEWAY_TOKEN, max_source_chars=500)
    )
    assert review.approved is False
    assert review.held is False  # oversized is a terminal reject, not a hold
    assert "large" in review.reason.lower()


# --- VAL-PRISM-034: no provider key on the validator; token/keys redacted -------------------------


def test_eval_container_env_carries_no_provider_key(tmp_path: Path) -> None:
    settings = PrismSettings(
        base_eval_artifact_root=tmp_path / "artifacts",
        llm_gateway_token=GATEWAY_TOKEN,
    )
    ctx = PrismContext(vocab_size=128, sequence_length=16, seed=1337, max_parameters=5_000_000)
    evaluator = PrismContainerEvaluator(settings=settings, ctx=ctx)

    env = evaluator._env("sub-1", "codehash", "archhash", "base_gpu")
    blob = "\n".join(f"{key}={value}" for key, value in env.items())
    assert PROVIDER_KEY not in blob
    assert GATEWAY_TOKEN not in blob
    assert not any("API_KEY" in key for key in env)


def test_failed_closed_reason_redacts_secrets(monkeypatch) -> None:
    def leak(config, *, system, prompt):  # noqa: ANN001
        raise RuntimeError(f"upstream rejected Authorization: Bearer {GATEWAY_TOKEN}")

    monkeypatch.setattr(llm, "_invoke_review_flow", leak)

    review = review_code(
        "ok",
        config=LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=GATEWAY_TOKEN),
    )
    assert review.held is True
    assert GATEWAY_TOKEN not in review.reason
    assert "[REDACTED]" in review.reason


def test_plagiarism_failed_closed_reason_redacts_secrets(monkeypatch) -> None:
    def leak(config, *, system, prompt):  # noqa: ANN001
        raise RuntimeError(f"upstream rejected Authorization: Bearer {GATEWAY_TOKEN}")

    monkeypatch.setattr(llm, "_invoke_review_flow", leak)

    review = review_plagiarism(
        current_code="def f():\n    return 1\n",
        candidate_code="def g():\n    return 2\n",
        comparison_report={"overlap": 0.0},
        config=LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=GATEWAY_TOKEN),
    )
    assert review.copied is True
    assert "llm_review_failed" in review.violations
    assert GATEWAY_TOKEN not in review.reason
    assert "[REDACTED]" in review.reason


def test_redact_detail_scrubs_sensitive_lines() -> None:
    detail = (
        "step 1 ok\n"
        f"Authorization: Bearer {GATEWAY_TOKEN}\n"
        "OPENROUTER_API_KEY=sk-or-leak\n"
        "step 2 ok\n"
    )
    redacted = _redact_detail(detail)
    assert GATEWAY_TOKEN not in redacted
    assert "sk-or-leak" not in redacted
    assert "step 1 ok" in redacted
    assert "step 2 ok" in redacted


# --- VAL-PRISM-035: two-tier emission split (architecture owner + training owner) -----------------


async def _new_repository(tmp_path: Path, name: str) -> PrismRepository:
    database = Database(tmp_path / name)
    await database.init()
    return PrismRepository(database, epoch_seconds=EPOCH_SECONDS)


async def test_get_weights_empty_store_returns_empty(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-empty.sqlite3")
    # No architecture has ever scored -> no crown -> BURN ({}).
    assert await get_weights(repository, EPOCH_SECONDS) == {}


async def test_get_weights_splits_between_distinct_owners(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-split.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="train-owner",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights["arch-owner"] == pytest.approx(0.60)
    assert weights["train-owner"] == pytest.approx(0.40)
    assert sum(weights.values()) == pytest.approx(1.0)


async def test_get_weights_same_owner_takes_full_pool(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-same.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="solo",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="solo",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights == {"solo": pytest.approx(1.0)}


async def test_get_weights_honors_db_configured_custom_split(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-custom.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="train-owner",
        q_recipe=0.8,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await repository.store_runtime_config(
        config_key="reward_pools",
        value={"architecture": 0.7, "training": 0.3},
        updated_by="ops",
    )

    runtime_config = await repository.runtime_config(PrismSettings(), official=True)
    weights = await get_weights(
        repository,
        EPOCH_SECONDS,
        architecture_weight=runtime_config.reward_pools.architecture,
        training_weight=runtime_config.reward_pools.training,
    )

    assert weights["arch-owner"] == pytest.approx(0.70)
    assert weights["train-owner"] == pytest.approx(0.30)
    assert sum(weights.values()) == pytest.approx(1.0)


async def test_get_weights_crown_is_cross_epoch(tmp_path: Path) -> None:
    # An OLDER, higher-scoring architecture keeps the crown over a NEWER, lower-scoring one: the
    # crown is global/all-time, not scoped to the current epoch.
    repository = await _new_repository(tmp_path, "weights-crossepoch.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-old",
        owner_hotkey="old-owner",
        q_arch_best=0.95,
        created_at="2023-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-old",
        architecture_id="arch-old",
        owner_hotkey="old-train",
        q_recipe=0.9,
        created_at="2023-01-01T00:00:00+00:00",
    )
    await _seed_architecture(
        repository,
        architecture_id="arch-new",
        owner_hotkey="new-owner",
        q_arch_best=0.50,
        created_at="2025-06-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-new",
        architecture_id="arch-new",
        owner_hotkey="new-train",
        q_recipe=0.4,
        created_at="2025-06-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert set(weights) == {"old-owner", "old-train"}
    assert weights["old-owner"] == pytest.approx(0.60)
    assert weights["old-train"] == pytest.approx(0.40)


async def test_get_weights_no_architecture_families_burns(tmp_path: Path) -> None:
    repository = await _new_repository(tmp_path, "weights-burn.sqlite3")
    assert await get_weights(repository, EPOCH_SECONDS) == {}


async def test_get_weights_nonpositive_crown_burns(tmp_path: Path) -> None:
    # A crown holder whose all-time best is non-positive is not a real learner -> BURN.
    repository = await _new_repository(tmp_path, "weights-zero-crown.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.0,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _seed_training_variant(
        repository,
        variant_id="var-1",
        architecture_id="arch-1",
        owner_hotkey="train-owner",
        q_recipe=0.0,
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert await get_weights(repository, EPOCH_SECONDS) == {}


async def test_get_weights_missing_training_variant_gives_arch_owner_full(tmp_path: Path) -> None:
    # Crowned architecture exists but has NO training variant -> arch owner takes the whole pool.
    repository = await _new_repository(tmp_path, "weights-no-variant.sqlite3")
    await _seed_architecture(
        repository,
        architecture_id="arch-1",
        owner_hotkey="arch-owner",
        q_arch_best=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )

    weights = await get_weights(repository, EPOCH_SECONDS)

    assert weights == {"arch-owner": pytest.approx(1.0)}


# --- VAL-PRISM-036: /internal/v1/get_weights response shape unchanged -----------------------------


def test_get_weights_endpoint_shape_under_internal_auth(tmp_path: Path) -> None:
    with TestClient(create_app(_pipeline_settings(tmp_path, "weights-shape.sqlite3"))) as client:
        assert client.get("/internal/v1/get_weights").status_code == 401
        response = client.get(
            "/internal/v1/get_weights",
            headers={"Authorization": "Bearer secret", "X-Base-Challenge-Slug": "prism"},
        )
    assert response.status_code == 200
    body = response.json()
    assert {"challenge_slug", "epoch", "weights"} <= set(body)
    assert body["challenge_slug"] == "prism"
    assert isinstance(body["epoch"], int)
    assert isinstance(body["weights"], dict)
    assert all(isinstance(value, float) for value in body["weights"].values())


# --- helpers --------------------------------------------------------------------------------------


def _make_fake_chat_with_verdict(verdict: Any) -> type:
    args_by_tool = {
        "SubmitMermaid": _ALLOW_ARGS["SubmitMermaid"],
        "SubmitVerdict": {**_ALLOW_ARGS["SubmitVerdict"], "verdict": verdict},
    }

    class _FakeMessage:
        def __init__(self, tool_name: str) -> None:
            self.tool_calls = [{"name": tool_name, "args": args_by_tool[tool_name]}]

    class _FakeChat:
        def __init__(self, **kwargs: Any) -> None:
            self._tool: str | None = None

        def bind_tools(self, tools: list[Any], tool_choice: str, strict: bool) -> _FakeChat:
            self._tool = tool_choice
            return self

        def invoke(self, messages: list[tuple[str, str]]) -> _FakeMessage:
            assert self._tool is not None
            return _FakeMessage(self._tool)

    return _FakeChat


def _pipeline_settings(tmp_path: Path, name: str, **overrides: Any) -> PrismSettings:
    base: dict[str, Any] = dict(
        database_url=f"sqlite+aiosqlite:///{tmp_path / name}",
        shared_token="secret",
        allow_insecure_signatures=True,
        llm_review_enabled=True,
        llm_gateway_url=GATEWAY_URL,
        llm_gateway_token=GATEWAY_TOKEN,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        distributed_contract_policy="off",
    )
    base.update(overrides)
    return PrismSettings(**base)


def _submit(client: TestClient, code: str, *, nonce: str) -> str:
    payload = {"code": code, "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body, nonce=nonce), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client: TestClient) -> int:
    return client.post(
        "/internal/v1/worker/process-next", headers={"Authorization": "Bearer secret"}
    ).status_code


def _db_state(tmp_path: Path, name: str, submission_id: str) -> dict[str, Any]:
    async def fetch() -> dict[str, Any]:
        database = Database(tmp_path / name)
        async with database.connect() as conn:
            submission = list(
                await conn.execute_fetchall(
                    "SELECT status FROM submissions WHERE id=?", (submission_id,)
                )
            )[0]
            leases = await conn.execute_fetchall(
                "SELECT 1 FROM gpu_leases WHERE submission_id=?", (submission_id,)
            )
            jobs = await conn.execute_fetchall(
                "SELECT 1 FROM eval_jobs WHERE submission_id=? AND level != 'l1'", (submission_id,)
            )
            scores = await conn.execute_fetchall(
                "SELECT 1 FROM scores WHERE submission_id=?", (submission_id,)
            )
        return {
            "status": str(submission["status"]),
            "gpu_leases": len(list(leases)),
            "eval_jobs": len(list(jobs)),
            "scores": len(list(scores)),
        }

    return anyio.run(fetch)


async def _seed_architecture(
    repository: PrismRepository,
    *,
    architecture_id: str,
    owner_hotkey: str,
    q_arch_best: float,
    created_at: str,
    family_hash: str | None = None,
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO architecture_families("
            "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
            "owner_submission_id, canonical_submission_id, q_arch_best, display_name, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                architecture_id,
                family_hash or f"fh-{architecture_id}",
                f"fp-{architecture_id}",
                f"bp-{architecture_id}",
                owner_hotkey,
                f"sub-{architecture_id}",
                f"sub-{architecture_id}",
                q_arch_best,
                f"arch-{architecture_id}",
                created_at,
                created_at,
            ),
        )


async def _seed_training_variant(
    repository: PrismRepository,
    *,
    variant_id: str,
    architecture_id: str,
    owner_hotkey: str,
    q_recipe: float,
    created_at: str,
    is_current_best: int = 1,
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO training_variants("
            "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
            "metric_mean, metric_std, is_current_best, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                variant_id,
                architecture_id,
                f"th-{variant_id}",
                owner_hotkey,
                f"sub-{variant_id}",
                q_recipe,
                q_recipe,
                0.0,
                is_current_best,
                created_at,
                created_at,
            ),
        )
