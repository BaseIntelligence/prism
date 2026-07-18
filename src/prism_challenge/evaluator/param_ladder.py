"""Dual parameter ladder for PRISM Research Lab emission admission (VAL-RESLAB-003+).

Two admitted stages replace the legacy single 150M emission cap:

| Stage      | Cap           | Role                                               |
| ---------- | ------------- | -------------------------------------------------- |
| ``explore``  | 124_000_000 | Cheap continuous exploration; may provisional-crown |
| ``promote``  | 350_000_000 | Same package pin re-eval; confirms or revokes       |

Stage labels ride on scores/manifests as ``param_ladder_stage`` (values ``explore`` /
``promote``). Provisional-crown eligibility is explore-stage only; promote-path hooks
(confirm/revoke state machine) live on top of these types — full weights wiring can
complete in the weights feature when split cleanly.

Official Comparison default pin follows the **promote** ceiling so dual-pin honesty is
documentable (explore runs may still pin 124M explicitly).
"""

from __future__ import annotations

from typing import Any, Literal

ParamLadderStage = Literal["explore", "promote"]

EXPLORE_MAX_PARAMETERS = 124_000_000
PROMOTE_MAX_PARAMETERS = 350_000_000

# Canonical stage label strings (scores / manifests / Official pin).
STAGE_EXPLORE: ParamLadderStage = "explore"
STAGE_PROMOTE: ParamLadderStage = "promote"

PARAM_LADDER_STAGE_FIELD = "param_ladder_stage"
PARAM_LADDER_CAP_FIELD = "param_ladder_cap"
PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD = "provisional_crown_eligible"

# All admitted stage labels in fixed order (explore first).
PARAM_LADDER_STAGES: tuple[ParamLadderStage, ...] = (STAGE_EXPLORE, STAGE_PROMOTE)

# Map stage → hard max_parameters.
PARAM_LADDER_CAPS: dict[ParamLadderStage, int] = {
    STAGE_EXPLORE: EXPLORE_MAX_PARAMETERS,
    STAGE_PROMOTE: PROMOTE_MAX_PARAMETERS,
}

# Official Comparison default pin uses the promote ceiling; explore pins remain explicit.
OFFICIAL_DEFAULT_PARAM_CAP = PROMOTE_MAX_PARAMETERS
OFFICIAL_DEFAULT_PARAM_STAGE: ParamLadderStage = STAGE_PROMOTE

# Backward-compat alias: single historical constant that once meant "the" emission cap.
# Paths still importing a singular name should prefer stage-aware helpers below.
LEGACY_SINGLE_PARAM_CAP = 150_000_000


def max_parameters_for_stage(stage: ParamLadderStage | str) -> int:
    """Return the hard param cap for an admitted ladder stage.

    Raises ``ValueError`` on unknown stage labels (fail closed).
    """
    key = str(stage).strip().lower()
    if key not in PARAM_LADDER_CAPS:
        raise ValueError(
            f"unknown param ladder stage {stage!r}; expected one of {list(PARAM_LADDER_STAGES)}"
        )
    return PARAM_LADDER_CAPS[key]  # type: ignore[index]


def normalize_param_ladder_stage(
    stage: ParamLadderStage | str | None,
    *,
    default: ParamLadderStage = STAGE_EXPLORE,
) -> ParamLadderStage:
    """Coerce a caller/config stage label to a canonical admitted stage.

    ``None`` / blank → ``default`` (explore for continuous emission admission).
    """
    if stage is None:
        return default
    key = str(stage).strip().lower()
    if not key:
        return default
    if key not in PARAM_LADDER_CAPS:
        raise ValueError(
            f"unknown param ladder stage {stage!r}; expected one of {list(PARAM_LADDER_STAGES)}"
        )
    return key  # type: ignore[return-value]


def resolve_max_parameters(
    *,
    stage: ParamLadderStage | str | None = None,
    max_parameters: int | None = None,
    default_stage: ParamLadderStage = STAGE_EXPLORE,
) -> tuple[ParamLadderStage, int]:
    """Resolve ``(stage, cap)`` from optional stage + optional numeric override.

    Rules:
    - Explicit ``max_parameters`` wins for the numeric cap; stage still defaults to
      ``default_stage`` unless provided (allows both lab overrides and dual labels).
    - When only stage is set, the stage's locked cap is used.
    - When neither is set, return ``(default_stage, cap(default_stage))``.
    """
    resolved_stage = normalize_param_ladder_stage(stage, default=default_stage)
    if max_parameters is not None:
        cap = int(max_parameters)
        if cap <= 0:
            raise ValueError(f"max_parameters must be positive, got {max_parameters!r}")
        return resolved_stage, cap
    return resolved_stage, max_parameters_for_stage(resolved_stage)


def stage_for_param_count(
    param_count: int,
    *,
    prefer: ParamLadderStage | str | None = None,
) -> ParamLadderStage:
    """Pick the smallest ladder stage whose cap admits ``param_count``.

    Counts above the promote ceiling raise ``ValueError`` (fail closed for admission).
    When ``prefer`` is set and its cap already admits the count, that stage is kept
    (so an explicit promote-stage pin stays labeled promote even under 124M).
    """
    count = int(param_count)
    if count < 0:
        raise ValueError(f"param_count must be non-negative, got {param_count!r}")
    if prefer is not None:
        preferred = normalize_param_ladder_stage(prefer)
        if count <= max_parameters_for_stage(preferred):
            return preferred
    if count <= EXPLORE_MAX_PARAMETERS:
        return STAGE_EXPLORE
    if count <= PROMOTE_MAX_PARAMETERS:
        return STAGE_PROMOTE
    raise ValueError(f"param_count {count:,} exceeds promote ladder cap {PROMOTE_MAX_PARAMETERS:,}")


def is_within_stage_cap(param_count: int, stage: ParamLadderStage | str) -> bool:
    """True when ``param_count`` is ≤ the stage's hard cap."""
    return int(param_count) <= max_parameters_for_stage(stage)


def provisional_crown_eligible(
    *,
    stage: ParamLadderStage | str,
    param_count: int | None = None,
    score_valid: bool = True,
) -> bool:
    """Explore-stage provisional crown eligibility gate (VAL-RESLAB-004 hook).

    A qualifying explore-stage run may provisionally occupy architecture/training crowns.
    Promote-stage scores are confirmed/revoked on the promote path (not provisional).
    Full revoke/confirm state machine may complete in the weights feature; this helper is
    the admission/type surface tests lock here.
    """
    if not score_valid:
        return False
    resolved = normalize_param_ladder_stage(stage)
    if resolved != STAGE_EXPLORE:
        return False
    if param_count is not None and not is_within_stage_cap(param_count, STAGE_EXPLORE):
        return False
    return True


def promote_path_decision(
    *,
    provisional_stage: ParamLadderStage | str,
    promote_stage: ParamLadderStage | str,
    promote_valid: bool,
    promote_beats_provisional: bool | None = None,
) -> Literal["confirm", "revoke", "ineligible"]:
    """Promote-path state-machine hook (VAL-RESLAB-005 surface).

    - ``confirm``: promote-stage valid and (if compared) beats or ties provisional
    - ``revoke``: promote-stage invalid or loses fair compare vs provisional
    - ``ineligible``: stages are not the expected explore→promote pair

    Weights feature may expand this into durable crown transitions; admission/types
    tests lock the decision labels here.
    """
    provisional = normalize_param_ladder_stage(provisional_stage)
    promote = normalize_param_ladder_stage(promote_stage)
    if provisional != STAGE_EXPLORE or promote != STAGE_PROMOTE:
        return "ineligible"
    if not promote_valid:
        return "revoke"
    if promote_beats_provisional is False:
        return "revoke"
    return "confirm"


def ladder_labels(
    stage: ParamLadderStage | str,
    *,
    param_count: int | None = None,
    score_valid: bool = True,
    max_parameters: int | None = None,
) -> dict[str, Any]:
    """Stage labels for scores / manifests / Official pin honesty.

    Always includes ``param_ladder_stage`` and ``param_ladder_cap``. When
    ``param_count`` is known, also records provisional-crown eligibility.
    """
    resolved = normalize_param_ladder_stage(stage)
    cap = int(max_parameters) if max_parameters is not None else max_parameters_for_stage(resolved)
    payload: dict[str, Any] = {
        PARAM_LADDER_STAGE_FIELD: resolved,
        PARAM_LADDER_CAP_FIELD: cap,
    }
    if param_count is not None:
        payload["model_params"] = int(param_count)
        payload[PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD] = provisional_crown_eligible(
            stage=resolved,
            param_count=int(param_count),
            score_valid=score_valid,
        )
    else:
        payload[PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD] = provisional_crown_eligible(
            stage=resolved,
            score_valid=score_valid,
        )
    return payload


def dual_ladder_summary() -> dict[str, Any]:
    """Machine-readable dual-ladder constants for pin / docs / tests."""
    return {
        "stages": list(PARAM_LADDER_STAGES),
        "explore_max_parameters": EXPLORE_MAX_PARAMETERS,
        "promote_max_parameters": PROMOTE_MAX_PARAMETERS,
        "official_default_param_cap": OFFICIAL_DEFAULT_PARAM_CAP,
        "official_default_param_stage": OFFICIAL_DEFAULT_PARAM_STAGE,
        "legacy_single_param_cap_removed": LEGACY_SINGLE_PARAM_CAP,
        "provisional_crown_stage": STAGE_EXPLORE,
        "promote_path_stage": STAGE_PROMOTE,
    }
