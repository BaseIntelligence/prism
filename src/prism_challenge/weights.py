from __future__ import annotations

from datetime import UTC, datetime
from typing import SupportsFloat, cast

from .repository import PrismRepository, epoch_id_for


async def get_weights(
    repository: PrismRepository,
    epoch_seconds: int,
    *,
    architecture_weight: float = 0.60,
    training_weight: float = 0.40,
) -> dict[str, float]:
    epoch_id = epoch_id_for(datetime.now(UTC), epoch_seconds)
    rows = await repository.score_rows(epoch_id)
    return _normalize(rows)


def _normalize(rows: list[dict[str, object]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        hotkey = str(row["hotkey"])
        raw_score = row.get("score", row.get("final_score", 0.0))
        score = max(0.0, float(cast(SupportsFloat, raw_score)))
        best[hotkey] = best.get(hotkey, 0.0) + score
    total = sum(best.values())
    if total <= 0:
        return {}
    return {hotkey: score / total for hotkey, score in best.items() if score > 0}
