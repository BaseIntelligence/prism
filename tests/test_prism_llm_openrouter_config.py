"""prism llm_review gateway config contract (milestone llm-yunwu; VAL-LLM-CODE-008).

The prism safety gate routes ONLY through the master LLM gateway (``/llm/v1``) with a scoped token;
the gateway injects the provider key AND the model server-side. There is NO ``openrouter_*`` /
legacy ``CHUTES_*`` config, NO hardcoded provider base URL or model in code, and NO direct-provider
fallback. These tests pin the removal + the gateway-only wiring.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import llm_review as llm
from prism_challenge.evaluator.llm_review import (
    GATEWAY_MODEL_PLACEHOLDER,
    LlmReviewConfig,
    review_code,
)
from prism_challenge.queue import PrismWorker
from prism_challenge.runtime_config import resolve_runtime_policy, runtime_policy_defaults

GATEWAY_URL = "http://base-master:18080/llm/v1"
GATEWAY_TOKEN = "scoped-gateway-token"
GATEWAY_SECRET_PATH = Path("/run/secrets/base_gateway_token")


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PRISM_LLM_REVIEW_ENABLED",
        "PRISM_LLM_GATEWAY_URL",
        "BASE_LLM_GATEWAY_URL",
        "PRISM_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN",
        "PRISM_GATEWAY_TOKEN_FILE",
        "BASE_GATEWAY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_prism_settings_default_to_gateway_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    settings = PrismSettings()

    assert settings.llm_review_enabled is True
    # Fail-closed default: a deploy without a wired gateway must NOT silently allow submissions.
    assert settings.llm_review_required is True
    # The scoped gateway token is sourced from the docker secret; no provider key/base URL/model.
    assert settings.llm_gateway_token_file == GATEWAY_SECRET_PATH
    assert settings.llm_review_temperature == 0.0
    assert not hasattr(settings, "openrouter_base_url")
    assert not hasattr(settings, "openrouter_model")
    assert not hasattr(settings, "openrouter_api_key")
    assert not hasattr(settings, "openrouter_api_key_file")
    assert not hasattr(settings, "openrouter_api_key_value")


def test_gateway_token_only_sourced_from_secret_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_llm_env(monkeypatch)
    secret = tmp_path / "base_gateway_token"
    secret.write_text("scoped-secret\n", encoding="utf-8")

    settings = PrismSettings(llm_gateway_token_file=secret)
    assert settings.llm_gateway_token_value() == "scoped-secret"

    # With no inline token and a missing file, nothing is resolved (fails closed).
    missing = PrismSettings(llm_gateway_token=None, llm_gateway_token_file=tmp_path / "nope")
    assert missing.llm_gateway_token_value() is None


def test_hf_token_only_sourced_from_secret_file(tmp_path: Path) -> None:
    secret = tmp_path / "hf_token"
    secret.write_text("hf_secret\n", encoding="utf-8")
    settings = PrismSettings(hf_token_file=secret)
    assert settings.hf_token_value() == "hf_secret"

    # FineWeb-Edu is public (anonymous works), so a missing token file resolves
    # to None instead of failing: the prep download simply runs unauthenticated.
    missing = PrismSettings(hf_token=None, hf_token_file=tmp_path / "nope")
    assert missing.hf_token_value() is None


def test_legacy_chutes_env_alias_no_longer_recognized(monkeypatch: pytest.MonkeyPatch) -> None:
    # The legacy PRISM_CHUTES_* / PRISM_OPENROUTER_* env aliases are gone (extra=ignore), so they
    # cannot re-introduce a provider base URL / model onto settings.
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PRISM_CHUTES_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("PRISM_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    settings = PrismSettings()
    assert not hasattr(settings, "openrouter_model")
    assert not hasattr(settings, "openrouter_base_url")


def test_llm_review_config_defaults_are_gateway_only() -> None:
    config = LlmReviewConfig()

    assert config.enabled is True
    assert config.gateway_url is None
    assert config.gateway_token is None
    assert config.temperature == 0.0
    # No hardcoded provider base URL / model / api key remains on the config.
    assert not hasattr(config, "base_url")
    assert not hasattr(config, "model")
    assert not hasattr(config, "api_key")
    assert not hasattr(config, "api_key_file")


def test_worker_llm_config_maps_gateway_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_llm_env(monkeypatch)
    secret = tmp_path / "base_gateway_token"
    secret.write_text("scoped-worker\n", encoding="utf-8")
    settings = PrismSettings(llm_gateway_url=GATEWAY_URL, llm_gateway_token_file=secret)

    config = PrismWorker._llm_config(SimpleNamespace(settings=settings))

    assert config.enabled is True
    assert config.gateway_url == GATEWAY_URL
    assert config.gateway_token == "scoped-worker"
    assert config.temperature == 0.0


def test_runtime_policy_llm_review_has_no_provider_base_url_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    settings = PrismSettings()
    defaults = runtime_policy_defaults(settings)

    policy = defaults["llm_review_policy"]
    assert policy["enabled"] is True
    # The gateway injects provider + model, so the runtime policy pins neither.
    assert "base_url" not in policy
    assert "model" not in policy

    model = resolve_runtime_policy(settings, [])
    assert model.llm_review_policy.enabled is True
    assert not hasattr(model.llm_review_policy, "base_url")
    assert not hasattr(model.llm_review_policy, "model")


def test_invoke_review_flow_targets_gateway_with_placeholder_model() -> None:
    captured: dict[str, Any] = {}

    class _FakeMessage:
        def __init__(self, tool_name: str) -> None:
            self.tool_calls = [
                {
                    "name": tool_name,
                    "args": _ARGS_BY_TOOL[tool_name],
                }
            ]

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

    monkeypatch_target = llm
    original = monkeypatch_target._load_chat_openai
    monkeypatch_target._load_chat_openai = lambda: _FakeChat  # type: ignore[assignment]
    try:
        review = review_code(
            "def build_model(ctx):\n    return None\n",
            config=LlmReviewConfig(gateway_url=GATEWAY_URL, gateway_token=GATEWAY_TOKEN),
        )
    finally:
        monkeypatch_target._load_chat_openai = original  # type: ignore[assignment]

    # The gate targets the master gateway /llm/v1 with the scoped token; the model is a placeholder
    # (the gateway overwrites it server-side), never a hardcoded provider model.
    assert captured["base_url"] == GATEWAY_URL
    assert captured["model"] == GATEWAY_MODEL_PLACEHOLDER
    assert captured["temperature"] == 0.0
    assert captured["api_key"] == GATEWAY_TOKEN
    assert review.approved is True


_ARGS_BY_TOOL: dict[str, dict[str, Any]] = {
    "SubmitMermaid": {"mermaid": "flowchart LR\n  A[Source] --> B[Review]", "notes": "ok"},
    "SubmitVerdict": {
        "reason": "defines build_model; no escapes detected",
        "verdict": True,
        "violations": [],
        "confidence": 0.9,
        "rule_ids": [],
        "evidence": [],
    },
}
