from __future__ import annotations

from typing import SupportsFloat, cast

from .repository import PrismRepository

# Versioned score-owner policy recorded for aggregation provenance (VAL-WEIGHT-094).
# One positive canonical score owner is selected per hotkey for the emission map;
# non-positive scores contribute zero and do not receive hidden renormalization
# advantages beyond the explicit architecture/training pool split.
SCORE_OWNER_POLICY_VERSION = "score-owner.architecture-training.v1"


async def get_weights(
    repository: PrismRepository,
    epoch_seconds: int,
    *,
    architecture_weight: float = 0.60,
    training_weight: float = 0.40,
) -> dict[str, float]:
    """Split prism's emission between the best architecture and best training-script owners.

    Two-tier, cross-epoch (persistent crown): the global all-time best architecture's owner takes
    the ``architecture_weight`` share and the owner of the best training variant on that winning
    architecture takes the ``training_weight`` share. ``epoch_seconds`` is retained for signature
    stability but no longer scopes the ranking (the crown is global, not per-epoch).

    Non-positive architecture scores contribute an empty map (explicit zero to the master after
    a synthetic zero push, or skip; never a hidden redistribution across non-owners).
    """
    best_arch = await repository.best_architecture()
    # BURN: no architecture has ever scored, or the crown holder's all-time best is non-positive ->
    # emit nothing so the master burns prism's share rather than rewarding a non-learner.
    if best_arch is None or float(cast(SupportsFloat, best_arch["q_arch_best"])) <= 0.0:
        return {}

    weights: dict[str, float] = {}
    arch_owner = str(best_arch["owner_hotkey"])
    weights[arch_owner] = weights.get(arch_owner, 0.0) + architecture_weight

    best_training = await repository.best_training_variant(str(best_arch["id"]))
    # Missing-training fallback: the crowned architecture has no training variant, so its owner is
    # the only recipient and the renormalization below lifts the lone share to 1.0. When a variant
    # exists its owner takes the training share even at a non-positive q_recipe (it is still a real
    # training script owner). A shared owner naturally accumulates both shares.
    if best_training is not None:
        training_owner = str(best_training["owner_hotkey"])
        weights[training_owner] = weights.get(training_owner, 0.0) + training_weight

    return _renormalize(weights)


def _renormalize(weights: dict[str, float]) -> dict[str, float]:
    positive = {hotkey: weight for hotkey, weight in weights.items() if weight > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {hotkey: weight / total for hotkey, weight in positive.items()}
